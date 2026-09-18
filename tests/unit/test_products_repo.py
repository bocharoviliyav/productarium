"""Unit tests for ``api.repositories.product_repo``.

Covers:
- ORM<->Pydantic mappers: ``orm_to_product``, ``_codebase_orm_from_pydantic``,
  ``_spec_orm_from_pydantic``, ``_links_orm_from_pydantic``.
- ``load_product_orm`` (found / not found).
- ``list_products_light`` (empty / counters / visibility / pagination).
- ``upsert_product`` (insert, update, full child replace).
- ``delete_product`` (existing / missing no-op).
- Per-type add/delete for codebase/spec/links (including replace-on-duplicate).
- ``update_codebase_content`` (pages replace, page_id upsert new + existing,
  generated_docs replace, and the 'Provide one of' ValueError).
- ``update_spec_content`` / ``update_links_content`` (happy + not-found).
- ``verify_child`` / ``verify_page`` (per-page flags: set, reset-on-edit,
  wholesale-pages strip).
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
    DatabaseORM,
    LinksORM,
    ProductORM,
    SpecORM,
)
from api.repositories import product_repo as pr
from api.schemas import Codebase, Database, Links, Product, ProductListItem, Spec


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
        # P0-2: the payload token is WRITE-ONLY — the mapper persists only the
        # caller-resolved stored (encrypted) value; a raw "tok" never lands.
        assert orm.token is None
        stored = pr._resolved_stored_token(c.token, None)
        assert stored is not None and stored != "tok"  # Fernet ciphertext
        orm2 = pr._codebase_orm_from_pydantic(c, stored_token=stored)
        assert orm2.token == stored
        assert pr.get_codebase_token(orm2) == "tok"
        assert orm.generated_docs is None
        assert orm.pages is None
        # P0-5: verified is server-owned — not copied from the payload.
        assert orm.verified is None
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
        # P0-5: verified is server-owned — not copied from the payload
        # (column default applies at flush, so it is None on the fresh ORM).
        assert orm.verified is None
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
        # P0-5: verified is server-owned — not copied from the payload.
        assert orm.verified is None
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
# list_products_light
# --------------------------------------------------------------------------- #
class TestListProductsLight:
    def test_empty(self, session):
        assert pr.list_products_light(session) == ([], 0)

    def test_with_products(self, session):
        _seed_product(session, "prod_1")
        _seed_product(session, "prod_2")
        items, total = pr.list_products_light(session)
        assert total == 2
        assert {p.id for p in items} == {"prod_1", "prod_2"}

    def test_returns_light_pydantic_models(self, session):
        _seed_product(session)
        items, _ = pr.list_products_light(session)
        assert isinstance(items[0], ProductListItem)
        assert items[0].id == "prod_1"

    def test_counters_verified_vs_total(self, session):
        _seed_product(session)
        session.add(CodebaseORM(
            id="cb_v", product_id="prod_1", name="V", source="manual",
            verified=True, verified_by="admin",
        ))
        session.add(CodebaseORM(id="cb_u", product_id="prod_1", name="U", source="manual"))
        session.add(SpecORM(id="spec_1", product_id="prod_1", name="S", kind="openapi", source="manual"))
        session.add(LinksORM(id="links_1", product_id="prod_1", name="L", source="manual"))
        session.commit()
        items, total = pr.list_products_light(session)
        assert total == 1
        it = items[0]
        assert it.codebases_count == 2
        assert it.verified_codebases == 1
        assert it.specs_count == 1
        assert it.verified_specs == 0
        assert it.links_count == 1
        assert it.verified_links == 0

    def test_visibility_filter(self, session):
        _seed_product(session, "prod_1")
        _seed_product(session, "prod_2")
        items, total = pr.list_products_light(session, product_ids=["prod_2"])
        assert total == 1
        assert [i.id for i in items] == ["prod_2"]
        # Empty visibility -> no query at all.
        assert pr.list_products_light(session, product_ids=[]) == ([], 0)

    def test_pagination_and_ordering(self, session):
        for i in range(3):
            session.add(ProductORM(
                id=f"prod_{i}", name=f"W{i}", description="d",
                created_at=datetime(2024, 1, 1, 0, 0, i),
            ))
        session.commit()

        items, total = pr.list_products_light(session, limit=2)
        assert total == 3
        assert [i.id for i in items] == ["prod_0", "prod_1"]

        items, total = pr.list_products_light(session, limit=2, offset=2)
        assert total == 3
        assert [i.id for i in items] == ["prod_2"]

        # Offset beyond the end: empty page, total unchanged.
        items, total = pr.list_products_light(session, limit=2, offset=10)
        assert items == []
        assert total == 3


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


# --------------------------------------------------------------------------- #
# verify_page (per-page verification; flags bind to exact content)
# --------------------------------------------------------------------------- #
class TestVerifyPage:
    def _seed_codebase_with_pages(self, session):
        _seed_product(session)
        pages = {"p1": {"id": "p1", "title": "P1", "content": "old"}}
        pr.add_codebase(session, "prod_1", _make_codebase(pages=pages))

    def test_sets_flags_on_the_page(self, session):
        self._seed_codebase_with_pages(session)
        product = pr.verify_page(session, "prod_1", "cb_1", "codebases", "p1", "user_1")
        page = product.codebases[0].pages["p1"]
        assert page["verified"] is True
        assert page["verified_by"] == "user_1"
        assert page["verified_at"]

    def test_verify_database_page(self, session):
        _seed_product(session)
        session.add(DatabaseORM(
            id="db_1", product_id="prod_1", name="DB", source="manual",
            pages={"page_overview": {"id": "page_overview", "title": "Overview", "content": "x"}},
        ))
        session.commit()
        product = pr.verify_page(
            session, "prod_1", "db_1", "databases", "page_overview", "user_1"
        )
        page = product.databases[0].pages["page_overview"]
        assert page["verified"] is True
        assert page["verified_by"] == "user_1"

    def test_missing_page_raises(self, session):
        self._seed_codebase_with_pages(session)
        with pytest.raises(ValueError, match="Page not found"):
            pr.verify_page(session, "prod_1", "cb_1", "codebases", "nope", "user_1")

    def test_missing_product_raises(self, session):
        with pytest.raises(ValueError, match="Product not found"):
            pr.verify_page(session, "missing", "cb_1", "codebases", "p1", "user_1")

    def test_content_change_resets_flags(self, session):
        self._seed_codebase_with_pages(session)
        pr.verify_page(session, "prod_1", "cb_1", "codebases", "p1", "user_1")
        product, _ = pr.update_codebase_content(
            session, "prod_1", "cb_1", page_id="p1", content="changed"
        )
        page = product.codebases[0].pages["p1"]
        assert "verified" not in page
        assert "verified_by" not in page
        assert "verified_at" not in page

    def test_identical_content_edit_keeps_flags(self, session):
        self._seed_codebase_with_pages(session)
        pr.verify_page(session, "prod_1", "cb_1", "codebases", "p1", "user_1")
        product, _ = pr.update_codebase_content(
            session, "prod_1", "cb_1", page_id="p1", content="old"
        )
        assert product.codebases[0].pages["p1"]["verified"] is True

    def test_wholesale_pages_strip_client_flags(self, session):
        self._seed_codebase_with_pages(session)
        forged = {
            "p1": {
                "id": "p1", "title": "P1", "content": "x",
                "verified": True, "verified_by": "evil", "verified_at": "t",
            }
        }
        product, _ = pr.update_codebase_content(session, "prod_1", "cb_1", pages=forged)
        page = product.codebases[0].pages["p1"]
        assert "verified" not in page
        assert "verified_by" not in page


# --------------------------------------------------------------------------- #
# ?light=1 shaping: meta-only pages + the PUT round-trip merge guard
# --------------------------------------------------------------------------- #
class TestLightPayloadHelpers:
    _FULL_PAGES = {
        "p1": {
            "id": "p1", "title": "P1", "parent": "root", "content": "body",
            "importance": "high", "filePaths": ["a.py"], "relatedPages": ["p2"],
            "provenance": {"judge": {"verdict": "ok"}},
            "verified": True, "verified_by": "u1", "verified_at": "2026-01-01",
        }
    }

    def test_strip_keeps_meta_drops_bodies(self):
        prod = _make_product(
            codebases=[_make_codebase(pages=self._FULL_PAGES, generated_docs="# blob")]
        )
        light = pr.strip_page_content(prod)
        page = light.codebases[0].pages["p1"]
        assert set(page) == {
            "id", "title", "parent", "importance", "filePaths", "relatedPages",
            "verified", "verified_by", "verified_at",
        }
        assert light.codebases[0].generated_docs is None
        # Copy semantics: the source product is untouched.
        assert prod.codebases[0].pages["p1"]["content"] == "body"
        assert prod.codebases[0].generated_docs == "# blob"

    def test_strip_keeps_pageless_blob_and_specs(self):
        prod = _make_product(
            codebases=[_make_codebase(generated_docs="# docs")],
            specs=[_make_spec()],
        )
        light = pr.strip_page_content(prod)
        assert light.codebases[0].generated_docs == "# docs"
        assert light.codebases[0].pages is None
        assert light.specs[0].content == "openapi: 3.0.0"

    def test_merge_fills_missing_content_from_storage(self, session):
        _seed_product(session)
        session.add(CodebaseORM(
            id="cb_1", product_id="prod_1", name="A", source="manual",
            generated_docs="# stored", pages=self._FULL_PAGES,
        ))
        session.commit()
        light = pr.strip_page_content(
            pr.orm_to_product(pr.load_product_orm(session, "prod_1"))
        )
        merged = pr.merge_stored_page_bodies(session, "prod_1", light)
        page = merged.codebases[0].pages["p1"]
        assert page["content"] == "body"  # restored from storage
        assert page["verified"] is True  # fresh meta wins
        assert merged.codebases[0].generated_docs == "# stored"

    def test_merge_respects_explicit_empty_content(self, session):
        _seed_product(session)
        session.add(CodebaseORM(
            id="cb_1", product_id="prod_1", name="A", source="manual",
            pages=self._FULL_PAGES,
        ))
        session.commit()
        prod = pr.orm_to_product(pr.load_product_orm(session, "prod_1"))
        prod.codebases[0] = prod.codebases[0].model_copy(update={
            "pages": {"p1": {"id": "p1", "title": "P1", "content": ""}},
        })
        merged = pr.merge_stored_page_bodies(session, "prod_1", prod)
        assert merged.codebases[0].pages["p1"]["content"] == ""

    def test_merge_noop_when_bodies_present(self, session):
        _seed_product(session)
        session.add(CodebaseORM(
            id="cb_1", product_id="prod_1", name="A", source="manual",
            pages=self._FULL_PAGES,
        ))
        session.commit()
        full = pr.orm_to_product(pr.load_product_orm(session, "prod_1"))
        assert pr.merge_stored_page_bodies(session, "prod_1", full) is full


class TestUpsertDatabaseVerifiedOwnership:
    def test_upsert_preserves_stored_verified_forces_false_for_new(self, session):
        _seed_product(session)
        verified_at = datetime(2026, 1, 1)
        session.add(DatabaseORM(
            id="db_1", product_id="prod_1", name="DB1", source="manual",
            verified=True, verified_by="user_owner", verified_at=verified_at,
        ))
        session.commit()

        product = _make_product()
        product.databases = [
            # Existing id: the client payload's verified triple is IGNORED,
            # the stored one survives (round-trip PUT cannot re-grant or
            # tamper with verification).
            Database(
                id="db_1", name="DB1 renamed",
                verified=True, verified_by="user_evil",
                verified_at=datetime(2026, 6, 6), source="manual",
            ),
            # New id: verified claims are forced off.
            Database(
                id="db_2", name="DB2",
                verified=True, verified_by="user_evil", source="manual",
            ),
        ]
        pr.upsert_product(session, product)

        row1 = session.get(DatabaseORM, "db_1")
        assert row1.verified is True
        assert row1.verified_by == "user_owner"
        assert row1.verified_at == verified_at
        row2 = session.get(DatabaseORM, "db_2")
        assert row2.verified is False
        assert row2.verified_by is None
        assert row2.verified_at is None
