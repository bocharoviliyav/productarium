#!/usr/bin/env python3
"""Integration tests for ``api.routers.product_mcp`` (Wave C contract 2).

Product ↔ MCP server bindings over an isolated SQLite DB:

- GET empty list; POST create (embedded server name/transport/status).
- 404s for unknown product / server / binding.
- Duplicate (product, server) binding -> 409 (documented deviation).
- PUT enabled + ``allowed_tools`` presence semantics (absent keeps, explicit
  null = all tools) + validation 400s.
- DELETE -> 200, then 404.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _user_orm():
    from api.models import UserORM

    return UserORM(
        id="user_1",
        username="alice",
        role="user",
        provider="local",
        created_at=datetime.utcnow(),
    )


def _build_client(db_mod, product_mcp_mod):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.auth import deps as auth_deps

    app = FastAPI()
    app.include_router(product_mcp_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[product_mcp_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_current_user] = _user_orm
    return app, TestClient(app)


def _seed(db_mod):
    from api.models import McpServerORM, ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id="prod_1", name="Acme"))
        db.add(
            McpServerORM(
                id="mcp_a",
                name="alpha",
                transport="http",
                url="http://a/mcp",
                enabled=True,
                status="ok",
            )
        )
        db.add(
            McpServerORM(
                id="mcp_b",
                name="beta",
                transport="stdio",
                command="/bin/srv",
                enabled=True,
                status="unknown",
            )
        )
        db.commit()


def _mod():
    from api.routers import product_mcp

    return product_mcp


class TestProductMcpBindings:
    def test_list_empty(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        resp = client.get("/api/products/prod_1/mcp")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_binding_embeds_server_fields(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        resp = client.post(
            "/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_a"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"].startswith("pmb_")
        assert body["mcp_server_id"] == "mcp_a"
        assert body["name"] == "alpha"
        assert body["transport"] == "http"
        assert body["enabled"] is True
        assert body["allowed_tools"] is None
        assert body["status"] == "ok"

    def test_create_binding_with_allowlist(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        resp = client.post(
            "/api/products/prod_1/mcp",
            json={"mcp_server_id": "mcp_b", "enabled": False,
                  "allowed_tools": ["echo", "fetch"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["allowed_tools"] == ["echo", "fetch"]
        assert body["status"] == "unknown"

    def test_create_unknown_server_404(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        resp = client.post(
            "/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_nope"}
        )
        assert resp.status_code == 404

    def test_create_unknown_product_404(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        resp = client.post(
            "/api/products/prod_nope/mcp", json={"mcp_server_id": "mcp_a"}
        )
        assert resp.status_code == 404

    def test_list_unknown_product_404(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        assert client.get("/api/products/prod_nope/mcp").status_code == 404

    def test_duplicate_binding_409(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        assert (
            client.post("/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_a"}).status_code
            == 200
        )
        resp = client.post(
            "/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_a"}
        )
        assert resp.status_code == 409

    def test_two_servers_bound_independently(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        client.post("/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_a"})
        assert (
            client.post("/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_b"}).status_code
            == 200
        )
        listing = client.get("/api/products/prod_1/mcp").json()
        assert {b["mcp_server_id"] for b in listing} == {"mcp_a", "mcp_b"}


class TestProductMcpUpdate:
    def _binding_id(self, client):
        resp = client.post("/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_a"})
        return resp.json()["id"]

    def test_update_enabled(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        bid = self._binding_id(client)
        resp = client.put(
            f"/api/products/prod_1/mcp/{bid}", json={"enabled": False}
        )
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False
        assert resp.json()["allowed_tools"] is None  # untouched

    def test_update_sets_allowlist(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        bid = self._binding_id(client)
        resp = client.put(
            f"/api/products/prod_1/mcp/{bid}", json={"allowed_tools": ["echo"]}
        )
        assert resp.status_code == 200
        assert resp.json()["allowed_tools"] == ["echo"]

    def test_explicit_null_resets_allowlist(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        bid = self._binding_id(client)
        client.put(f"/api/products/prod_1/mcp/{bid}", json={"allowed_tools": ["echo"]})
        resp = client.put(f"/api/products/prod_1/mcp/{bid}", json={"allowed_tools": None})
        assert resp.status_code == 200
        assert resp.json()["allowed_tools"] is None

    def test_absent_fields_keep_values(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        bid = self._binding_id(client)
        client.put(f"/api/products/prod_1/mcp/{bid}", json={"allowed_tools": ["echo"]})
        resp = client.put(f"/api/products/prod_1/mcp/{bid}", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["allowed_tools"] == ["echo"]
        assert body["enabled"] is True

    @pytest.mark.parametrize(
        "tools,expected",
        [
            ([""], 400),               # empty entry (semantic)
            (["x" * 200], 400),        # entry too long (semantic)
            (["ok", ""], 400),         # one bad entry (semantic)
            # too many entries: rejected by the Pydantic schema-level
            # Field(max_length) guard -> FastAPI 422 (codebase convention:
            # 422 schema violations, 400 semantic violations)
            ([f"t{i}" for i in range(130)], 422),
        ],
    )
    def test_invalid_allowlist_rejected(self, isolated_db, tools, expected):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        bid = self._binding_id(client)
        resp = client.put(
            f"/api/products/prod_1/mcp/{bid}", json={"allowed_tools": tools}
        )
        assert resp.status_code == expected

    def test_update_unknown_binding_404(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        assert (
            client.put(
                "/api/products/prod_1/mcp/pmb_nope", json={"enabled": False}
            ).status_code
            == 404
        )

    def test_update_unknown_product_404(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        assert (
            client.put(
                "/api/products/prod_nope/mcp/pmb_x", json={"enabled": False}
            ).status_code
            == 404
        )


class TestProductMcpDelete:
    def test_delete_binding_200_then_404(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        bid = client.post(
            "/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_a"}
        ).json()["id"]

        resp = client.delete(f"/api/products/prod_1/mcp/{bid}")
        assert resp.status_code == 200
        assert "message" in resp.json()

        assert client.delete(f"/api/products/prod_1/mcp/{bid}").status_code == 404
        assert client.get("/api/products/prod_1/mcp").json() == []

    def test_delete_unknown_binding_404(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _mod())
        assert client.delete("/api/products/prod_1/mcp/pmb_nope").status_code == 404
