"""Unit tests for ``api.routers.products`` and ``api.db``.

Router coverage (via ``build_test_client``):
- GET    /api/products                     (empty + with product)
- POST   /api/products                     (create)
- GET    /api/products/{id}                (200 + 404)
- PUT    /api/products/{id}                (update)
- DELETE /api/products/{id}
- POST   /api/products/{id}/codebases      (add + 404)
- DELETE /api/products/{id}/codebases/{cid} (delete + 404)
- PUT    /api/products/{id}/codebases/{cid} (pages, page_id+content, generated_docs, 400, 404)
- POST/DELETE/PUT specs
- POST/DELETE/PUT links
- POST verify endpoints (owner ok, admin ok, 403 non-owner, 404 missing product/entity)

DB coverage:
- ``_safe_url``: postgres url with password, sqlite url, unparseable.
- ``_build_database_url``: postgres + non-postgres fallback.
- ``init_db``: idempotency + failure (monkeypatch create_all to raise).
"""

from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime
from typing import Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import CodebaseORM, LinksORM, ProductORM, SpecORM, UserORM
from api.schemas import Codebase, Links, Product, Spec
from api.routers import products as products_router_module
from tests.conftest import build_test_client


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _product_payload(pid: str = "prod_1") -> dict:
    return {
        "id": pid,
        "name": "Widget",
        "description": "A widget",
        "summary": "sum",
        "owner_id": None,
        "codebases": [],
        "specs": [],
        "links": [],
    }


def _codebase_payload(cid: str = "cb_1") -> dict:
    return {
        "id": cid,
        "name": "Repo A",
        "repo_url": "https://github.com/x/y",
        "repo_type": "github",
        "token": "tok",
        "generated_docs": None,
        "pages": None,
        "verified": False,
        "verified_by": None,
        "verified_at": None,
        "source": "manual",
    }


def _spec_payload(sid: str = "spec_1") -> dict:
    return {
        "id": sid,
        "name": "OpenAPI",
        "kind": "openapi",
        "content": "openapi: 3.0.0",
        "verified": False,
        "verified_by": None,
        "verified_at": None,
        "source": "manual",
    }


def _links_payload(lid: str = "links_1") -> dict:
    return {
        "id": lid,
        "name": "Links A",
        "content": "[]",
        "verified": False,
        "verified_by": None,
        "verified_at": None,
        "source": "manual",
    }


def _seed_product(db_mod, pid: str = "prod_1", owner_id: Optional[str] = None) -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(ProductORM(id=pid, name="Widget", description="desc", summary=None, owner_id=owner_id))
        s.commit()
    finally:
        s.close()


def _seed_codebase(db_mod, pid: str = "prod_1", cid: str = "cb_1") -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(CodebaseORM(id=cid, product_id=pid, name="Repo A", source="manual"))
        s.commit()
    finally:
        s.close()


def _seed_spec(db_mod, pid: str = "prod_1", sid: str = "spec_1") -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(SpecORM(id=sid, product_id=pid, name="S", kind="openapi", source="manual"))
        s.commit()
    finally:
        s.close()


def _seed_links(db_mod, pid: str = "prod_1", lid: str = "links_1") -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(LinksORM(id=lid, product_id=pid, name="L", source="manual"))
        s.commit()
    finally:
        s.close()


def _make_client(db_mod):
    return build_test_client(db_mod, [products_router_module], auth_none=True)


def _disable_reindex(monkeypatch):
    """Stub out the cognee re-index so PUT endpoints don't touch cognee/asyncio."""
    monkeypatch.setattr(products_router_module, "_reindex", lambda *a, **kw: None)


# --------------------------------------------------------------------------- #
# GET /api/products
# --------------------------------------------------------------------------- #
class TestListProducts:
    def test_empty(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.get("/api/products")
        assert r.status_code == 200
        assert r.json() == []

    def test_with_product(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.get("/api/products")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["id"] == "prod_1"


# --------------------------------------------------------------------------- #
# POST /api/products
# --------------------------------------------------------------------------- #
class TestCreateProduct:
    def test_create(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.post("/api/products", json=_product_payload())
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "prod_1"
        assert body["name"] == "Widget"

    def test_create_with_children(self, isolated_db):
        payload = _product_payload()
        payload["codebases"] = [_codebase_payload()]
        payload["specs"] = [_spec_payload()]
        payload["links"] = [_links_payload()]
        app, client = _make_client(isolated_db)
        r = client.post("/api/products", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert len(body["codebases"]) == 1
        assert len(body["specs"]) == 1
        assert len(body["links"]) == 1


# --------------------------------------------------------------------------- #
# GET /api/products/{id}
# --------------------------------------------------------------------------- #
class TestGetProduct:
    def test_found(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.get("/api/products/prod_1")
        assert r.status_code == 200
        assert r.json()["id"] == "prod_1"

    def test_not_found(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.get("/api/products/missing")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# PUT /api/products/{id}
# --------------------------------------------------------------------------- #
class TestUpdateProduct:
    def test_update(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        payload = _product_payload()
        payload["name"] = "Updated"
        r = client.put("/api/products/prod_1", json=payload)
        assert r.status_code == 200
        assert r.json()["name"] == "Updated"

    def test_update_upserts_missing(self, isolated_db):
        # PUT uses upsert, so a non-existent product is created.
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_new", json=_product_payload("prod_new"))
        assert r.status_code == 200
        assert r.json()["id"] == "prod_new"


# --------------------------------------------------------------------------- #
# DELETE /api/products/{id}
# --------------------------------------------------------------------------- #
class TestDeleteProduct:
    def test_delete(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/prod_1")
        assert r.status_code == 200
        assert "deleted" in r.json()["message"].lower()
        # Confirm gone.
        assert client.get("/api/products/prod_1").status_code == 404

    def test_delete_missing_is_noop(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/missing")
        assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Codebases
# --------------------------------------------------------------------------- #
class TestCodebaseEndpoints:
    def test_add(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/prod_1/codebases", json=_codebase_payload())
        assert r.status_code == 200
        assert len(r.json()["codebases"]) == 1

    def test_add_missing_product_404(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/missing/codebases", json=_codebase_payload())
        assert r.status_code == 404

    def test_delete(self, isolated_db):
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/prod_1/codebases/cb_1")
        assert r.status_code == 200
        assert r.json()["codebases"] == []

    def test_delete_missing_product_404(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/missing/codebases/cb_1")
        assert r.status_code == 404

    def test_update_pages(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/codebases/cb_1", json={"pages": {"p1": {"id": "p1", "title": "P1", "content": "x"}}})
        assert r.status_code == 200
        assert r.json()["codebases"][0]["pages"]["p1"]["content"] == "x"

    def test_update_page_id_new(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/codebases/cb_1", json={"page_id": "p_new", "content": "hello"})
        assert r.status_code == 200
        page = r.json()["codebases"][0]["pages"]["p_new"]
        assert page["content"] == "hello"

    def test_update_generated_docs(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/codebases/cb_1", json={"generated_docs": "# Docs"})
        assert r.status_code == 200
        assert r.json()["codebases"][0]["generated_docs"] == "# Docs"

    def test_update_empty_body_400(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/codebases/cb_1", json={})
        assert r.status_code == 400
        assert "Provide one of" in r.json()["detail"]

    def test_update_missing_product_404(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/missing/codebases/cb_1", json={"pages": {}})
        assert r.status_code == 404

    def test_update_missing_codebase_404(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/codebases/nope", json={"pages": {}})
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
class TestSpecEndpoints:
    def test_add(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/prod_1/specs", json=_spec_payload())
        assert r.status_code == 200
        assert len(r.json()["specs"]) == 1

    def test_add_missing_product_404(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/missing/specs", json=_spec_payload())
        assert r.status_code == 404

    def test_delete(self, isolated_db):
        _seed_product(isolated_db)
        _seed_spec(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/prod_1/specs/spec_1")
        assert r.status_code == 200
        assert r.json()["specs"] == []

    def test_delete_missing_product_404(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/missing/specs/spec_1")
        assert r.status_code == 404

    def test_update(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_spec(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/specs/spec_1", json={"content": "new yaml"})
        assert r.status_code == 200
        assert r.json()["specs"][0]["content"] == "new yaml"

    def test_update_missing_product_404(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/missing/specs/spec_1", json={"content": "x"})
        assert r.status_code == 404

    def test_update_missing_spec_404(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/specs/nope", json={"content": "x"})
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #
class TestLinksEndpoints:
    def test_add(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/prod_1/links", json=_links_payload())
        assert r.status_code == 200
        assert len(r.json()["links"]) == 1

    def test_add_missing_product_404(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/missing/links", json=_links_payload())
        assert r.status_code == 404

    def test_delete(self, isolated_db):
        _seed_product(isolated_db)
        _seed_links(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/prod_1/links/links_1")
        assert r.status_code == 200
        assert r.json()["links"] == []

    def test_delete_missing_product_404(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/missing/links/links_1")
        assert r.status_code == 404

    def test_update(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_links(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/links/links_1", json={"content": "[]"})
        assert r.status_code == 200
        assert r.json()["links"][0]["content"] == "[]"

    def test_update_missing_product_404(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/missing/links/links_1", json={"content": "x"})
        assert r.status_code == 404

    def test_update_missing_links_404(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/links/nope", json={"content": "x"})
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Verify endpoints (require get_current_user override)
# --------------------------------------------------------------------------- #
class TestVerifyEndpoints:
    def _client_with_user(self, db_mod, user):
        app, client = build_test_client(db_mod, [products_router_module], auth_none=True)
        app.dependency_overrides[products_router_module.get_current_user] = lambda: user
        return app, client

    def test_owner_can_verify_codebase(self, isolated_db, admin_user):
        _seed_product(isolated_db, owner_id="user_admin1")
        _seed_codebase(isolated_db)
        admin_user.id = "user_admin1"
        app, client = self._client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/prod_1/codebases/cb_1/verify")
        assert r.status_code == 200
        assert r.json()["codebases"][0]["verified"] is True

    def test_admin_can_verify_spec(self, isolated_db, admin_user):
        _seed_product(isolated_db, owner_id="someone_else")
        _seed_spec(isolated_db)
        app, client = self._client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/prod_1/specs/spec_1/verify")
        assert r.status_code == 200
        assert r.json()["specs"][0]["verified"] is True

    def test_non_owner_non_admin_403(self, isolated_db):
        _seed_product(isolated_db, owner_id="someone_else")
        _seed_codebase(isolated_db)
        user = UserORM(id="user_regular", username="regular", role="user", provider="local", created_at=datetime.utcnow())
        app, client = self._client_with_user(isolated_db, user)
        r = client.post("/api/products/prod_1/codebases/cb_1/verify")
        assert r.status_code == 403

    def test_verify_missing_product_404(self, isolated_db, admin_user):
        app, client = self._client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/missing/codebases/cb_1/verify")
        assert r.status_code == 404

    def test_verify_missing_entity_404(self, isolated_db, admin_user):
        _seed_product(isolated_db)
        app, client = self._client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/prod_1/codebases/nope/verify")
        assert r.status_code == 404

    def test_verify_links(self, isolated_db, admin_user):
        _seed_product(isolated_db)
        _seed_links(isolated_db)
        app, client = self._client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/prod_1/links/links_1/verify")
        assert r.status_code == 200
        assert r.json()["links"][0]["verified"] is True


# --------------------------------------------------------------------------- #
# api.db — _safe_url
# --------------------------------------------------------------------------- #
class TestSafeUrl:
    def test_postgres_url_strips_password(self):
        from api.db import _safe_url
        url = "postgresql+psycopg://cognee:secret@localhost:5432/cognee_db"
        safe = _safe_url(url)
        assert "secret" not in safe
        assert "***" in safe
        assert "cognee" in safe
        assert "localhost" in safe

    def test_sqlite_url_returned_as_is(self):
        from api.db import _safe_url
        url = "sqlite:///:memory:"
        assert _safe_url(url) == "sqlite:///:memory:"

    def test_url_without_credentials_returned_as_is(self):
        from api.db import _safe_url
        url = "postgresql+psycopg://localhost:5432/cognee_db"
        assert _safe_url(url) == url

    def test_unparseable_returns_placeholder(self):
        from api.db import _safe_url
        # The URL has "://" and "@", but the @ is before the :// so the
        # post-:// part has no "@" and split('@', 1) returns a single element,
        # causing the creds, rest = ... unpacking to raise ValueError -> the
        # except branch returns the placeholder.
        assert _safe_url("@://") == "<unparseable db url>"

    def test_plain_string_returned_as_is(self):
        from api.db import _safe_url
        # No "://" separator -> the function returns the url unchanged (the
        # outer condition is False so no split is attempted).
        assert _safe_url("not-a-url") == "not-a-url"


# --------------------------------------------------------------------------- #
# api.db — _build_database_url
# --------------------------------------------------------------------------- #
class TestBuildDatabaseUrl:
    def test_postgres_provider(self, monkeypatch):
        import api.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PROVIDER", "postgres")
        monkeypatch.setattr(db_mod, "DB_USERNAME", "u")
        monkeypatch.setattr(db_mod, "DB_PASSWORD", "p")
        monkeypatch.setattr(db_mod, "DB_HOST", "h")
        monkeypatch.setattr(db_mod, "DB_PORT", "5433")
        monkeypatch.setattr(db_mod, "DB_NAME", "n")
        url = db_mod._build_database_url()
        assert url == "postgresql+psycopg://u:p@h:5433/n"

    def test_postgresql_alias(self, monkeypatch):
        import api.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PROVIDER", "postgresql")
        monkeypatch.setattr(db_mod, "DB_USERNAME", "u")
        monkeypatch.setattr(db_mod, "DB_PASSWORD", "p")
        monkeypatch.setattr(db_mod, "DB_HOST", "h")
        monkeypatch.setattr(db_mod, "DB_PORT", "5432")
        monkeypatch.setattr(db_mod, "DB_NAME", "n")
        assert db_mod._build_database_url().startswith("postgresql+psycopg://")

    def test_non_postgres_falls_back_to_sqlite(self, monkeypatch):
        import api.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PROVIDER", "sqlite")
        assert db_mod._build_database_url() == "sqlite:///:memory:"


# --------------------------------------------------------------------------- #
# api.db — init_db idempotency + failure
# --------------------------------------------------------------------------- #
class TestInitDb:
    def test_idempotent(self, isolated_db):
        # isolated_db already called init_db once. Calling again should be a
        # no-op returning True without re-running create_all.
        assert isolated_db.init_db() is True
        assert isolated_db._db_ready is True

    def test_failure_returns_false(self, monkeypatch):
        import api.db as db_mod
        # Reset the ready flag so the failure path is exercised.
        monkeypatch.setattr(db_mod, "_db_ready", False)

        def _boom(*a, **kw):
            raise RuntimeError("db unreachable")

        monkeypatch.setattr(db_mod.Base.metadata, "create_all", _boom)
        assert db_mod.init_db() is False
        assert db_mod._db_ready is False

    def test_success_sets_ready_flag(self, monkeypatch):
        import api.db as db_mod
        monkeypatch.setattr(db_mod, "_db_ready", False)
        # Use the real engine from isolated_db so create_all succeeds.
        called = {"n": 0}
        original = db_mod.Base.metadata.create_all

        def _wrapper(*a, **kw):
            called["n"] += 1
            return original(*a, **kw)

        monkeypatch.setattr(db_mod.Base.metadata, "create_all", _wrapper)
        assert db_mod.init_db() is True
        assert db_mod._db_ready is True
        assert called["n"] == 1
