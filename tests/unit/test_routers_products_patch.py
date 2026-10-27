"""Unit tests for PATCH /api/products/{id} (SPEC-002 partial update).

Covers: partial semantics (absent fields unchanged), name trim/empty-400,
owner change (existing user ok, unknown 400), 404 unknown product, 403 for
read-only viewers, ``?light=1`` shaping, and the EntityBusyError -> 409
contract. PUT semantics stay pinned by tests/integration/test_products_router.py.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.docgen.jobs import EntityBusyError
from api.models import CodebaseORM, ProductORM, UserORM
from api.routers import products as products_router_module
from tests.conftest import build_test_client


def _seed_product(
    db_mod,
    pid: str = "prod_1",
    *,
    owner_id: Optional[str] = None,
    name: str = "Widget",
    description: str = "desc",
) -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(ProductORM(
            id=pid, name=name, description=description, summary=None,
            owner_id=owner_id,
        ))
        s.commit()
    finally:
        s.close()


def _seed_user(db_mod, uid: str, username: str, role: str = "user") -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(UserORM(
            id=uid, username=username, role=role, provider="local",
            created_at=datetime.utcnow(),
        ))
        s.commit()
    finally:
        s.close()


def _make_client(db_mod):
    # Default auth override: a fixed admin (rw on every product).
    return build_test_client(db_mod, [products_router_module])


class TestPatchProduct:
    def test_name_only_leaves_description(self, isolated_db):
        _seed_product(isolated_db, name="Widget", description="desc")
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1", json={"name": "Renamed"})
        assert r.status_code == 200
        assert r.json()["name"] == "Renamed"
        assert r.json()["description"] == "desc"

    def test_description_only_leaves_name(self, isolated_db):
        _seed_product(isolated_db, name="Widget", description="desc")
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1", json={"description": "new desc"})
        assert r.status_code == 200
        assert r.json()["name"] == "Widget"
        assert r.json()["description"] == "new desc"

    def test_empty_body_is_noop(self, isolated_db):
        _seed_product(isolated_db, name="Widget", description="desc")
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1", json={})
        assert r.status_code == 200
        assert r.json()["name"] == "Widget"
        assert r.json()["description"] == "desc"

    def test_name_trimmed(self, isolated_db):
        _seed_product(isolated_db)
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1", json={"name": "  Padded  "})
        assert r.status_code == 200
        assert r.json()["name"] == "Padded"

    def test_blank_name_400(self, isolated_db):
        _seed_product(isolated_db)
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1", json={"name": "   "})
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()

    def test_empty_description_clears(self, isolated_db):
        _seed_product(isolated_db, description="desc")
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1", json={"description": ""})
        assert r.status_code == 200
        assert r.json()["description"] == ""

    def test_owner_change(self, isolated_db):
        _seed_product(isolated_db, owner_id="user_a")
        _seed_user(isolated_db, "user_b", "bob")
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1", json={"owner_id": "user_b"})
        assert r.status_code == 200
        assert r.json()["owner_id"] == "user_b"
        assert r.json()["name"] == "Widget"  # other fields untouched

    def test_unknown_owner_400(self, isolated_db):
        _seed_product(isolated_db, owner_id="user_a")
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1", json={"owner_id": "ghost"})
        assert r.status_code == 400
        assert "owner" in r.json()["detail"].lower()

    def test_missing_product_404(self, isolated_db):
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/missing", json={"name": "X"})
        assert r.status_code == 404

    def test_readonly_viewer_403(self, isolated_db):
        _seed_product(isolated_db, owner_id="someone_else")
        app, client = _make_client(isolated_db)
        viewer = UserORM(
            id="user_viewer", username="viewer", role="viewer_global",
            provider="local", created_at=datetime.utcnow(),
        )
        app.dependency_overrides[products_router_module.get_current_user] = lambda: viewer
        r = client.patch("/api/products/prod_1", json={"name": "Nope"})
        assert r.status_code == 403

    def test_light_strips_page_bodies(self, isolated_db):
        _seed_product(isolated_db)
        s = isolated_db.SessionLocal()
        try:
            s.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="Repo A", source="manual",
                pages={"p1": {"id": "p1", "title": "P1", "content": "the body"}},
            ))
            s.commit()
        finally:
            s.close()
        _app, client = _make_client(isolated_db)
        r = client.patch("/api/products/prod_1?light=1", json={"name": "Renamed"})
        assert r.status_code == 200
        assert "content" not in r.json()["codebases"][0]["pages"]["p1"]
        # The stored body survives the partial update.
        full = client.get("/api/products/prod_1").json()
        assert full["codebases"][0]["pages"]["p1"]["content"] == "the body"

    def test_entity_busy_409(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _app, client = _make_client(isolated_db)

        def _busy(*a, **kw):
            raise EntityBusyError("locked by a running job")

        monkeypatch.setattr(products_router_module.product_repo, "patch_product", _busy)
        r = client.patch("/api/products/prod_1", json={"name": "X"})
        assert r.status_code == 409
