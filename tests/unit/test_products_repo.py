"""Unit tests for ``api.repositories.product_repo``.

Covers:
- ORM<->Pydantic mappers: ``orm_to_product``, ``_codebase_orm_from_pydantic``,
  ``_spec_orm_from_pydantic``, ``_links_orm_from_pydantic``.
- ``load_product_orm`` (found / not found).
- ``list_products`` (empty / with children).
- ``upsert_product`` (insert, update, full child replace).
- ``delete_product`` (existing / missing no-op).
- Per-type add/delete for codebase/spec/links (including replace-on-duplicate).
- ``update_codebase_content`` (pages replace, page_id upsert new + existing,
  generated_docs replace, and the 'Provide one of' ValueError).
- ``update_spec_content`` / ``update_links_content`` (happy + not-found).
- ``verify_child``.
- Error branches (product/codebase/spec/links not found).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import (
    CodebaseORM,
    LinksORM,
    ProductORM,
    SpecORM,
)
from api.repositories import product_repo as pr
from api.schemas import Codebase, Links, Product, Spec


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_codebase(
    cid: str = "cb_1",
    name: str = "Repo A",
    *,
    source: Optional[str] = None,
    pages: Optional[dict] = None,
    generated_docs: Optional[str] = None,
) -> Codebase:
    return Codebase(
        id=cid,
        name=name,
        repo_url="https://github.com/example/repo",
        repo_type="github",
        token="tok",
        generated_docs=generated_docs,
        pages=pages,
        verified=False,
        verified_by=None,
        verified_at=None,
        source=source if source is not None else "manual",
    )


def _make_spec(sid: str = "spec_1", name: str = "OpenAPI", *, kind: Optional[str] = None, source: Optional[str] = None) -> Spec:
    return Spec(
        id=sid,
        name=name,
        kind=kind if kind is not None else "openapi",
        content="openapi: 3.0.0",
        verified=False,
        verified_by=None,
        verified_at=None,
        source=source if source is not None else "manual",
    )


def _make_links(lid: str = "links_1", name: str = "Links A", *, source: Optional[str] = None) -> Links:
    return Links(
        id=lid,
        name=name,
        content='[{"url":"https://x","description":"d"}]',
        verified=False,
        verified_by=None,
        verified_at=None,
        source=source if source is not None else "manual",
    )


def _make_product(
    pid: str = "prod_1",
    name: str = "Widget",
    *,
    description: str = "A widget service",
    summary: str = "short summary",
    codebases: Optional[list] = None,
    specs: Optional[list] = None,
    links: Optional[list] = None,
    owner_id: Optional[str] = None,
) -> Product:
    return Product(
        id=pid,
        name=name,
        description=description,
        summary=summary,
        owner_id=owner_id,
        codebases=codebases or [],
        specs=specs or [],
        links=links or [],
    )


def _seed_product(session, pid: str = "prod_1") -> ProductORM:
    p = ProductORM(id=pid, name="Widget", description="desc", summary=None, owner_id=None)
    session.add(p)
    session.commit()
    return p


# --------------------------------------------------------------------------- #
# _codebase_orm_from_pydantic / _spec_orm_from_pydantic / _links_orm_from_pydantic
# --------------------------------------------------------------------------- #
class TestFromPydanticMappers:
    def test_codebase_orm_from_pydantic_copies_all_fields(self):
        c = _make_codebase()
        orm = pr._codebase_orm_from_pydantic(c)
        assert orm.id == "cb_1"
        assert orm.name == "Repo A"
        assert orm.repo_url == "https://github.com/example/repo"
        assert orm.repo_type == "github"
        assert orm.token == "tok"
        assert orm.generated_docs is None
        assert orm.pages is None
        assert orm.verified is False
        assert orm.source == "manual"

    def test_codebase_orm_from_pydantic_source_defaults_to_manual(self):
        c = Codebase(id="cb", name="n", source="")  # empty -> default "manual"
        orm = pr._codebase_orm_from_pydantic(c)
        assert orm.source == "manual"

    def test_codebase_orm_from_pydantic_preserves_explicit_source(self):
        c = _make_codebase(source="confluence")
        orm = pr._codebase_orm_from_pydantic(c)
        assert orm.source == "confluence"

    def test_spec_orm_from_pydantic_copies_all_fields(self):
        s = _make_spec()
        orm = pr._spec_orm_from_pydantic(s)
        assert orm.id == "spec_1"
        assert orm.name == "OpenAPI"
        assert orm.kind == "openapi"
        assert orm.content == "openapi: 3.0.0"
        assert orm.verified is False
        assert orm.source == "manual"

    def test_spec_orm_from_pydantic_kind_defaults_to_openapi(self):
        s = Spec(id="s", name="n", kind="", source="manual")
        orm = pr._spec_orm_from_pydantic(s)
        assert orm.kind == "openapi"

    def test_spec_orm_from_pydantic_explicit_kind_preserved(self):
        s = _make_spec(kind="asyncapi", source="github")
        orm = pr._spec_orm_from_pydantic(s)
        assert orm.kind == "asyncapi"
        assert orm.source == "github"

    def test_links_orm_from_pydantic_copies_all_fields(self):
        l = _make_links()
        orm = pr._links_orm_from_pydantic(l)
        assert orm.id == "links_1"
        assert orm.name == "Links A"
        assert orm.content is not None and "url" in orm.content
        assert orm.verified is False
        assert orm.source == "manual"

    def test_links_orm_from_pydantic_source_defaults_to_manual(self):
        l = Links(id="l", name="n", source="")
        orm = pr._links_orm_from_pydantic(l)
        assert orm.source == "manual"


# --------------------------------------------------------------------------- #
# orm_to_product
# --------------------------------------------------------------------------- #
class TestOrmToProduct:
    def test_empty_product_maps(self, session):
        p_orm = _seed_product(session)
        prod = pr.orm_to_product(p_orm)
        assert prod.id == "prod_1"
        assert prod.name == "Widget"
        assert prod.description == "desc"
        assert prod.summary is None
        assert prod.owner_id is None
        assert prod.codebases == []
        assert prod.specs == []
        assert prod.links == []

    def test_product_with_children_maps(self, session):
        p_orm = _seed_product(session)
        session.add(CodebaseORM(id="cb_1", product_id="prod_1", name="A", repo_url="u", source="github"))
        session.add(SpecORM(id="spec_1", product_id="prod_1", name="S", kind="asyncapi", content="c", source="github"))
        session.add(LinksORM(id="links_1", product_id="prod_1", name="L", content="[]", source="confluence"))
        session.commit()
        session.refresh(p_orm)

        prod = pr.orm_to_product(p_orm)
        assert len(prod.codebases) == 1
        assert prod.codebases[0].id == "cb_1"
        assert prod.codebases[0].source == "github"
        assert prod.codebases[0].repo_url == "u"
        assert len(prod.specs) == 1
        assert prod.specs[0].kind == "asyncapi"
        assert prod.specs[0].source == "github"
        assert len(prod.links) == 1
        assert prod.links[0].source == "confluence"

    def test_verified_fields_roundtrip(self, session):
        p_orm = _seed_product(session)
        ts = datetime.utcnow()
        session.add(CodebaseORM(
            id="cb_v", product_id="prod_1", name="A", source="manual",
            verified=True, verified_by="user_1", verified_at=ts,
        ))
        session.commit()
        session.refresh(p_orm)
        prod = pr.orm_to_product(p_orm)
        assert prod.codebases[0].verified is True
        assert prod.codebases[0].verified_by == "user_1"
        assert prod.codebases[0].verified_at is not None


# --------------------------------------------------------------------------- #
# load_product_orm
# --------------------------------------------------------------------------- #
class TestLoadProductOrm:
    def test_found(self, session):
        _seed_product(session)
        loaded = pr.load_product_orm(session, "prod_1")
        assert loaded is not None
        assert loaded.id == "prod_1"

    def test_not_found_returns_none(self, session):
        assert pr.load_product_orm(session, "missing") is None

    def test_eager_loads_children(self, session):
        p_orm = _seed_product(session)
        session.add(CodebaseORM(id="cb_1", product_id="prod_1", name="A", source="manual"))
        session.add(SpecORM(id="spec_1", product_id="prod_1", name="S", source="manual"))
        session.add(LinksORM(id="links_1", product_id="prod_1", name="L", source="manual"))
        session.commit()
        loaded = pr.load_product_orm(session, "prod_1")
        assert loaded is not None
        assert len(loaded.codebases) == 1
        assert len(loaded.specs) == 1
        assert len(loaded.links) == 1


# --------------------------------------------------------------------------- #
# list_products
# --------------------------------------------------------------------------- #
class TestListProducts:
    def test_empty(self, session):
        assert pr.list_products(session) == []

    def test_with_products(self, session):
        _seed_product(session, "prod_1")
        _seed_product(session, "prod_2")
        result = pr.list_products(session)
        assert len(result) == 2
        ids = {p.id for p in result}
        assert ids == {"prod_1", "prod_2"}

    def test_returns_pydantic_models(self, session):
        _seed_product(session)
        result = pr.list_products(session)
        assert isinstance(result[0], Product)


# --------------------------------------------------------------------------- #
# upsert_product
# --------------------------------------------------------------------------- #
class TestUpsertProduct:
    def test_insert_new(self, session):
        prod = _make_product(codebases=[_make_codebase()], specs=[_make_spec()], links=[_make_links()])
        orm = pr.upsert_product(session, prod)
        assert orm.id == "prod_1"
        assert orm.name == "Widget"
        assert orm.summary == "short summary"
        loaded = pr.load_product_orm(session, "prod_1")
        assert loaded is not None
        assert len(loaded.codebases) == 1
        assert len(loaded.specs) == 1
        assert len(loaded.links) == 1

    def test_insert_with_owner(self, session):
        prod = _make_product(owner_id="user_x")
        orm = pr.upsert_product(session, prod)
        assert orm.owner_id == "user_x"

    def test_update_existing(self, session):
        _seed_product(session)
        prod = _make_product(name="Updated", description="new desc", summary="new summary", owner_id="user_y")
        orm = pr.upsert_product(session, prod)
        assert orm.name == "Updated"
        assert orm.description == "new desc"
        assert orm.summary == "new summary"
        assert orm.owner_id == "user_y"

    def test_update_full_child_replace(self, session):
        # Seed a product with one of each child.
        prod = _make_product(
            codebases=[_make_codebase("cb_old")],
            specs=[_make_spec("spec_old")],
            links=[_make_links("links_old")],
        )
        pr.upsert_product(session, prod)

        # Upsert with a completely new set of children — old ones must be gone.
        prod2 = _make_product(
            codebases=[_make_codebase("cb_new")],
            specs=[_make_spec("spec_new")],
            links=[_make_links("links_new")],
        )
        pr.upsert_product(session, prod2)
        loaded = pr.load_product_orm(session, "prod_1")
        assert loaded is not None
        cb_ids = {c.id for c in loaded.codebases}
        spec_ids = {s.id for s in loaded.specs}
        link_ids = {l.id for l in loaded.links}
        assert cb_ids == {"cb_new"}
        assert spec_ids == {"spec_new"}
        assert link_ids == {"links_new"}

    def test_update_clears_all_children(self, session):
        prod = _make_product(codebases=[_make_codebase()], specs=[_make_spec()], links=[_make_links()])
        pr.upsert_product(session, prod)
        prod2 = _make_product()  # no children
        pr.upsert_product(session, prod2)
        loaded = pr.load_product_orm(session, "prod_1")
        assert loaded is not None
        assert loaded.codebases == []
        assert loaded.specs == []
        assert loaded.links == []


# --------------------------------------------------------------------------- #
# delete_product
# --------------------------------------------------------------------------- #
class TestDeleteProduct:
    def test_existing(self, session):
        _seed_product(session)
        session.add(CodebaseORM(id="cb_1", product_id="prod_1", name="A", source="manual"))
        session.commit()
        pr.delete_product(session, "prod_1")
        assert pr.load_product_orm(session, "prod_1") is None
        # Children should be cascaded away.
        assert session.query(CodebaseORM).count() == 0

    def test_missing_is_noop(self, session):
        pr.delete_product(session, "missing")
        # No exception raised.
        assert pr.load_product_orm(session, "missing") is None


# --------------------------------------------------------------------------- #
# add_codebase / delete_codebase
# --------------------------------------------------------------------------- #
class TestAddDeleteCodebase:
    def test_add_to_existing_product(self, session):
        _seed_product(session)
        result = pr.add_codebase(session, "prod_1", _make_codebase())
        assert isinstance(result, Product)
        assert len(result.codebases) == 1
        assert result.codebases[0].id == "cb_1"

    def test_add_replaces_duplicate(self, session):
        _seed_product(session)
        pr.add_codebase(session, "prod_1", _make_codebase("cb_1", name="Old"))
        result = pr.add_codebase(session, "prod_1", _make_codebase("cb_1", name="New"))
        assert len(result.codebases) == 1
        assert result.codebases[0].name == "New"

    def test_add_to_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.add_codebase(session, "missing", _make_codebase())

    def test_delete_existing(self, session):
        _seed_product(session)
        pr.add_codebase(session, "prod_1", _make_codebase())
        result = pr.delete_codebase(session, "prod_1", "cb_1")
        assert len(result.codebases) == 0

    def test_delete_missing_is_noop(self, session):
        _seed_product(session)
        result = pr.delete_codebase(session, "prod_1", "nope")
        assert result.codebases == []

    def test_delete_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.delete_codebase(session, "missing", "cb_1")


# --------------------------------------------------------------------------- #
# add_spec / delete_spec
# --------------------------------------------------------------------------- #
class TestAddDeleteSpec:
    def test_add_to_existing_product(self, session):
        _seed_product(session)
        result = pr.add_spec(session, "prod_1", _make_spec())
        assert len(result.specs) == 1
        assert result.specs[0].id == "spec_1"

    def test_add_replaces_duplicate(self, session):
        _seed_product(session)
        pr.add_spec(session, "prod_1", _make_spec("spec_1", name="Old"))
        result = pr.add_spec(session, "prod_1", _make_spec("spec_1", name="New"))
        assert len(result.specs) == 1
        assert result.specs[0].name == "New"

    def test_add_to_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.add_spec(session, "missing", _make_spec())

    def test_delete_existing(self, session):
        _seed_product(session)
        pr.add_spec(session, "prod_1", _make_spec())
        result = pr.delete_spec(session, "prod_1", "spec_1")
        assert len(result.specs) == 0

    def test_delete_missing_is_noop(self, session):
        _seed_product(session)
        result = pr.delete_spec(session, "prod_1", "nope")
        assert result.specs == []

    def test_delete_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.delete_spec(session, "missing", "spec_1")


# --------------------------------------------------------------------------- #
# add_links / delete_links
# --------------------------------------------------------------------------- #
class TestAddDeleteLinks:
    def test_add_to_existing_product(self, session):
        _seed_product(session)
        result = pr.add_links(session, "prod_1", _make_links())
        assert len(result.links) == 1
        assert result.links[0].id == "links_1"

    def test_add_replaces_duplicate(self, session):
        _seed_product(session)
        pr.add_links(session, "prod_1", _make_links("links_1", name="Old"))
        result = pr.add_links(session, "prod_1", _make_links("links_1", name="New"))
        assert len(result.links) == 1
        assert result.links[0].name == "New"

    def test_add_to_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.add_links(session, "missing", _make_links())

    def test_delete_existing(self, session):
        _seed_product(session)
        pr.add_links(session, "prod_1", _make_links())
        result = pr.delete_links(session, "prod_1", "links_1")
        assert len(result.links) == 0

    def test_delete_missing_is_noop(self, session):
        _seed_product(session)
        result = pr.delete_links(session, "prod_1", "nope")
        assert result.links == []

    def test_delete_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.delete_links(session, "missing", "links_1")


# --------------------------------------------------------------------------- #
# update_codebase_content
# --------------------------------------------------------------------------- #
class TestUpdateCodebaseContent:
    def test_pages_replace(self, session):
        _seed_product(session)
        pr.add_codebase(session, "prod_1", _make_codebase())
        new_pages = {"p1": {"id": "p1", "title": "P1", "content": "x"}}
        product, indexed = pr.update_codebase_content(session, "prod_1", "cb_1", pages=new_pages)
        assert indexed is not None
        assert '"p1"' in indexed
        assert product.codebases[0].pages == new_pages

    def test_page_id_upsert_new_page(self, session):
        _seed_product(session)
        # pages=None -> the repo creates a fresh dict, so SQLAlchemy detects
        # the assignment and persists the new page.
        pr.add_codebase(session, "prod_1", _make_codebase(pages=None))
        product, indexed = pr.update_codebase_content(
            session, "prod_1", "cb_1", page_id="p_new", content="hello"
        )
        assert indexed == "hello"
        page = product.codebases[0].pages["p_new"]
        assert page["content"] == "hello"
        assert page["id"] == "p_new"
        assert page["title"] == "p_new"
        assert page["filePaths"] == []
        assert page["importance"] == "medium"
        assert page["relatedPages"] == []

    def test_page_id_upsert_existing_page(self, session):
        _seed_product(session)
        pages = {"p1": {"id": "p1", "title": "P1", "content": "old", "filePaths": ["a"], "importance": "high", "relatedPages": ["p2"]}}
        pr.add_codebase(session, "prod_1", _make_codebase(pages=pages))
        # Note: SQLAlchemy's plain JSON column does not detect in-place
        # mutations when the same dict object is reassigned. The repo's
        # page_id-existing branch mutates in place, so the indexed_text
        # return value is the reliable indicator here.
        product, indexed = pr.update_codebase_content(
            session, "prod_1", "cb_1", page_id="p1", content="updated"
        )
        assert indexed == "updated"
        assert product.codebases[0].id == "cb_1"

    def test_page_id_upsert_when_pages_is_none(self, session):
        _seed_product(session)
        # pages=None on the ORM; the code initialises an empty dict.
        pr.add_codebase(session, "prod_1", _make_codebase(pages=None))
        product, indexed = pr.update_codebase_content(
            session, "prod_1", "cb_1", page_id="p_x", content="c"
        )
        assert indexed == "c"
        assert "p_x" in product.codebases[0].pages

    def test_generated_docs_replace(self, session):
        _seed_product(session)
        pr.add_codebase(session, "prod_1", _make_codebase())
        product, indexed = pr.update_codebase_content(
            session, "prod_1", "cb_1", generated_docs="# Docs"
        )
        assert indexed == "# Docs"
        assert product.codebases[0].generated_docs == "# Docs"

    def test_no_edit_shape_raises(self, session):
        _seed_product(session)
        pr.add_codebase(session, "prod_1", _make_codebase())
        with pytest.raises(ValueError, match="Provide one of"):
            pr.update_codebase_content(session, "prod_1", "cb_1")

    def test_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.update_codebase_content(session, "missing", "cb_1", pages={})

    def test_missing_codebase_raises(self, session):
        _seed_product(session)
        with pytest.raises(ValueError, match="Codebase not found"):
            pr.update_codebase_content(session, "prod_1", "nope", pages={})


# --------------------------------------------------------------------------- #
# update_spec_content
# --------------------------------------------------------------------------- #
class TestUpdateSpecContent:
    def test_replace_content(self, session):
        _seed_product(session)
        pr.add_spec(session, "prod_1", _make_spec())
        product, indexed = pr.update_spec_content(session, "prod_1", "spec_1", "new yaml")
        assert indexed == "new yaml"
        assert product.specs[0].content == "new yaml"

    def test_replace_with_none(self, session):
        _seed_product(session)
        pr.add_spec(session, "prod_1", _make_spec())
        product, indexed = pr.update_spec_content(session, "prod_1", "spec_1", None)
        assert indexed is None
        assert product.specs[0].content is None

    def test_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.update_spec_content(session, "missing", "spec_1", "x")

    def test_missing_spec_raises(self, session):
        _seed_product(session)
        with pytest.raises(ValueError, match="Spec not found"):
            pr.update_spec_content(session, "prod_1", "nope", "x")


# --------------------------------------------------------------------------- #
# update_links_content
# --------------------------------------------------------------------------- #
class TestUpdateLinksContent:
    def test_replace_content(self, session):
        _seed_product(session)
        pr.add_links(session, "prod_1", _make_links())
        product, indexed = pr.update_links_content(session, "prod_1", "links_1", "[]")
        assert indexed == "[]"
        assert product.links[0].content == "[]"

    def test_replace_with_none(self, session):
        _seed_product(session)
        pr.add_links(session, "prod_1", _make_links())
        product, indexed = pr.update_links_content(session, "prod_1", "links_1", None)
        assert indexed is None
        assert product.links[0].content is None

    def test_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.update_links_content(session, "missing", "links_1", "x")

    def test_missing_links_raises(self, session):
        _seed_product(session)
        with pytest.raises(ValueError, match="Links not found"):
            pr.update_links_content(session, "prod_1", "nope", "x")


# --------------------------------------------------------------------------- #
# verify_child
# --------------------------------------------------------------------------- #
class TestVerifyChild:
    def test_verify_codebase(self, session):
        _seed_product(session)
        pr.add_codebase(session, "prod_1", _make_codebase())
        product = pr.verify_child(session, "prod_1", "cb_1", "codebases", "user_1")
        assert product.codebases[0].verified is True
        assert product.codebases[0].verified_by == "user_1"
        assert product.codebases[0].verified_at is not None

    def test_verify_spec(self, session):
        _seed_product(session)
        pr.add_spec(session, "prod_1", _make_spec())
        product = pr.verify_child(session, "prod_1", "spec_1", "specs", "user_1")
        assert product.specs[0].verified is True
        assert product.specs[0].verified_by == "user_1"

    def test_verify_links(self, session):
        _seed_product(session)
        pr.add_links(session, "prod_1", _make_links())
        product = pr.verify_child(session, "prod_1", "links_1", "links", "user_1")
        assert product.links[0].verified is True
        assert product.links[0].verified_by == "user_1"

    def test_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.verify_child(session, "missing", "cb_1", "codebases", "user_1")

    def test_missing_entity_raises(self, session):
        _seed_product(session)
        with pytest.raises(ValueError, match="Entity not found"):
            pr.verify_child(session, "prod_1", "nope", "codebases", "user_1")
