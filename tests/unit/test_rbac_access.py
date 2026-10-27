#!/usr/bin/env python3
"""Unit tests for the SPEC-001 two-level permission model (docs | manage).

Matrix over cheap endpoint probes (no LLM, no job worker):
- docs probes   — PUT codebases/specs/links content, PUT databases doc-shape,
  verify (entity + page);
- manage probes — POST codebases (artifact CRUD), PUT databases metadata
  shape; edit probe — PATCH product (SPEC-023: owner/admin only).

Principals: admin, owner, rw grant, ro grant, manager (ro baseline / owner /
rw grant), viewer_global (pure read), stranger (invisible -> 404).
Also covers the product grants endpoints GET/PUT/DELETE
/api/products/{id}/grants (configure right, SPEC-023) and the grants field in
the Product payload.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --- Helpers ----------------------------------------------------------------
def _user(uid: str, role: str):
    from api.models import UserORM

    return UserORM(
        id=uid, username=uid, role=role, provider="local",
        created_at=datetime.utcnow(),
    )


def _build_client(db_mod, router_mods, user):
    """App over the isolated DB with get_current_user overridden by ``user``."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    for mod in router_mods:
        app.include_router(mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    first = router_mods[0]
    # get_db / get_current_user are the same imported objects in every router
    # module, so one override each covers the whole dependency graph.
    app.dependency_overrides[first.get_db] = _get_test_db
    app.dependency_overrides[first.get_current_user] = lambda: user
    return app, TestClient(app)


def _seed(db_mod):
    """prod_1 (owner user_owner) with one codebase/spec/links/database row.

    Grants on prod_1: user_grantee_rw -> rw, user_grantee_ro -> ro.
    prod_2 owned by user_manager (manager-owned product).
    """
    from api.models import (
        CodebaseORM,
        DatabaseORM,
        LinksORM,
        ProductGrantORM,
        ProductORM,
        SpecORM,
        UserORM,
    )

    users = [
        _user("user_owner", "user"),
        _user("user_grantee_rw", "user"),
        _user("user_grantee_ro", "user"),
        _user("user_stranger", "user"),
        _user("user_target", "user"),
        _user("user_admin", "admin"),
        _user("user_manager", "manager"),
        _user("user_viewer", "viewer_global"),
    ]
    with db_mod.SessionLocal() as db:
        for u in users:
            db.add(UserORM(**{c: getattr(u, c) for c in (
                "id", "username", "role", "provider", "created_at")}))
        db.add(ProductORM(id="prod_1", name="P1", description="d",
                          owner_id="user_owner"))
        db.add(ProductORM(id="prod_2", name="P2", description="d",
                          owner_id="user_manager"))
        db.add(CodebaseORM(
            id="cb_1", product_id="prod_1", name="svc",
            generated_docs="docs",
            pages={"p1": {"id": "p1", "title": "T", "content": "c"}},
        ))
        db.add(SpecORM(id="spec_1", product_id="prod_1", name="spec",
                       kind="openapi", content="openapi: 3.1.0"))
        db.add(LinksORM(id="links_1", product_id="prod_1", name="links",
                        content="[]"))
        db.add(DatabaseORM(id="db_1", product_id="prod_1", name="Main DB",
                           dsn_masked=None, source="manual",
                           generated_docs="db docs"))
        db.add(ProductGrantORM(
            product_id="prod_1", user_id="user_grantee_rw",
            level="rw", granted_by="user_owner",
        ))
        db.add(ProductGrantORM(
            product_id="prod_1", user_id="user_grantee_ro",
            level="ro", granted_by="user_owner",
        ))
        db.commit()


def _routers():
    from api.routers import databases as databases_mod
    from api.routers import products as products_mod

    return [products_mod, databases_mod]


# --- product_access (pure function) -------------------------------------------
class TestProductAccessFlags:
    def test_flags_matrix(self, isolated_db):
        from api.auth.deps import product_access
        from api.models import ProductGrantORM, ProductORM

        _seed(isolated_db)
        with isolated_db.SessionLocal() as db:
            prod = db.get(ProductORM, "prod_1")
            grant_rw = ProductGrantORM(level="rw")
            grant_ro = ProductGrantORM(level="ro")

            a = product_access(_user("user_admin", "admin"), prod)
            assert (a.level, a.can_docs, a.can_manage,
                    a.can_edit, a.can_configure) == ("rw", True, True, True, True)
            a = product_access(_user("user_owner", "user"), prod)
            assert (a.level, a.can_docs, a.can_manage,
                    a.can_edit, a.can_configure) == ("rw", True, True, True, True)
            a = product_access(
                _user("user_grantee_rw", "user"), prod, grant_rw)
            assert (a.level, a.can_docs, a.can_manage,
                    a.can_edit, a.can_configure) == ("rw", True, True, False, False)
            # ro grant: docs rights WITHOUT manage.
            a = product_access(
                _user("user_grantee_ro", "user"), prod, grant_ro)
            assert (a.level, a.can_docs, a.can_manage,
                    a.can_edit, a.can_configure) == ("ro", True, False, False, False)
            # manager baseline ro on a foreign product: docs, no manage, but
            # configure yes (SPEC-023: manager is global).
            a = product_access(_user("user_manager", "manager"), prod)
            assert (a.level, a.can_docs, a.can_manage,
                    a.can_edit, a.can_configure) == ("ro", True, False, False, True)
            # viewer_global: pure read — access but NO docs rights.
            a = product_access(_user("user_viewer", "viewer_global"), prod)
            assert (a.level, a.can_docs, a.can_manage,
                    a.can_edit, a.can_configure) == ("ro", False, False, False, False)
            # stranger: invisible.
            a = product_access(_user("user_stranger", "user"), prod)
            assert (a.level, a.can_docs, a.can_manage,
                    a.can_edit, a.can_configure) == (None, False, False, False, False)
            assert product_access(_user("x", "user"), None).level is None


# --- Endpoint matrix ----------------------------------------------------------
class TestDocsManageMatrix:
    """docs probes vs manage probes for every principal on prod_1."""

    CASES = [
        # (role, uid, read, docs, manage, edit)
        ("admin", "user_admin", 200, 200, 200, 200),
        ("user", "user_owner", 200, 200, 200, 200),      # owner
        ("user", "user_grantee_rw", 200, 200, 200, 403),  # rw grant
        ("user", "user_grantee_ro", 200, 200, 403, 403),  # ro grant: docs only
        ("manager", "user_manager", 200, 200, 403, 403),  # ro baseline
        ("viewer_global", "user_viewer", 200, 403, 403, 403),  # pure read
        ("user", "user_stranger", 404, 404, 404, 404),   # invisible
    ]

    @pytest.mark.parametrize("role,uid,read_code,docs_code,manage_code,edit_code", CASES)
    def test_matrix(self, isolated_db, role, uid, read_code, docs_code,
                    manage_code, edit_code):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _routers(), _user(uid, role))

        # read
        r = client.get("/api/products/prod_1")
        assert r.status_code == read_code, (uid, "read", r.text)

        # docs: edit codebase docs / spec content / links content / db doc shape
        r = client.put("/api/products/prod_1/codebases/cb_1",
                       json={"generated_docs": "edited"})
        assert r.status_code == docs_code, (uid, "docs-codebase", r.text)
        r = client.put("/api/products/prod_1/specs/spec_1",
                       json={"content": "openapi: 3.1.1"})
        assert r.status_code == docs_code, (uid, "docs-spec", r.text)
        r = client.put("/api/products/prod_1/links/links_1",
                       json={"content": "[]"})
        assert r.status_code == docs_code, (uid, "docs-links", r.text)
        r = client.put("/api/products/prod_1/databases/db_1",
                       json={"generated_docs": "edited"})
        assert r.status_code == docs_code, (uid, "docs-db", r.text)
        # docs: verify (entity + page)
        r = client.post("/api/products/prod_1/codebases/cb_1/verify")
        assert r.status_code == docs_code, (uid, "verify", r.text)
        r = client.post("/api/products/prod_1/codebases/cb_1/pages/p1/verify")
        assert r.status_code == docs_code, (uid, "verify-page", r.text)

        # manage: artifact CRUD + db metadata shape
        r = client.post("/api/products/prod_1/codebases",
                        json={"id": "cb_new", "name": "new"})
        assert r.status_code == manage_code, (uid, "manage-add", r.text)
        r = client.put("/api/products/prod_1/databases/db_1",
                       json={"name": "Renamed"})
        assert r.status_code == manage_code, (uid, "manage-db-meta", r.text)

        # edit: product settings (SPEC-023) — owner or admin only
        r = client.patch("/api/products/prod_1", json={"name": "P1x"})
        assert r.status_code == edit_code, (uid, "edit-patch", r.text)

    def test_manager_owner_full_rights(self, isolated_db):
        """The manager OWNS prod_2: docs AND manage (and sees it)."""
        _seed(isolated_db)
        app, client = _build_client(
            isolated_db, _routers(), _user("user_manager", "manager")
        )
        assert client.get("/api/products/prod_2").status_code == 200
        assert client.patch(
            "/api/products/prod_2", json={"name": "P2x"}
        ).status_code == 200

    def test_rw_grant_on_foreign_product(self, isolated_db):
        from api.models import ProductGrantORM

        _seed(isolated_db)
        with isolated_db.SessionLocal() as s:
            s.add(ProductGrantORM(product_id="prod_2", user_id="user_owner",
                                  level="rw", granted_by="user_manager"))
            s.commit()
        app, client = _build_client(
            isolated_db, _routers(), _user("user_owner", "user")
        )
        assert client.get("/api/products/prod_2").status_code == 200
        # rw grant still manages artifacts, but never product settings (SPEC-023).
        assert client.post(
            "/api/products/prod_2/codebases", json={"id": "cb_x", "name": "x"}
        ).status_code == 200
        assert client.patch(
            "/api/products/prod_2", json={"name": "P2x"}
        ).status_code == 403


# --- Product payload grants ----------------------------------------------------
class TestProductGrantsField:
    def test_payload_carries_grants(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(
            isolated_db, _routers(), _user("user_grantee_ro", "user")
        )
        r = client.get("/api/products/prod_1")
        assert r.status_code == 200
        grants = {g["user_id"]: g["level"] for g in r.json()["grants"]}
        assert grants == {"user_grantee_rw": "rw", "user_grantee_ro": "ro"}

    def test_client_supplied_grants_ignored_on_put(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(
            isolated_db, _routers(), _user("user_owner", "user")
        )
        r = client.put(
            "/api/products/prod_1",
            json={
                "id": "prod_1", "name": "P1", "description": "d",
                "owner_id": "user_owner",
                "codebases": [], "specs": [], "links": [],
                "grants": [{"user_id": "user_stranger", "level": "rw"}],
            },
        )
        assert r.status_code == 200, r.text
        grants = {g["user_id"] for g in r.json()["grants"]}
        assert "user_stranger" not in grants


# --- Grants endpoints -----------------------------------------------------------
class TestProductGrantsEndpoints:
    """GET/PUT/DELETE /api/products/{product_id}/grants (right: configure,
    SPEC-023: owner / manager / admin — grants do NOT delegate it)."""

    CASES = [
        ("admin", "user_admin", 200),
        ("user", "user_owner", 200),           # owner
        ("user", "user_grantee_rw", 403),      # rw grant is NOT configure
        ("user", "user_grantee_ro", 403),      # docs only
        ("manager", "user_manager", 200),      # manager is global
        ("user", "user_stranger", 404),        # invisible
    ]

    @pytest.mark.parametrize("role,uid,code", CASES)
    def test_get_grants(self, isolated_db, role, uid, code):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _routers(), _user(uid, role))
        r = client.get("/api/products/prod_1/grants")
        assert r.status_code == code, (uid, r.text)
        if code == 200:
            rows = {g["user_id"]: g["level"] for g in r.json()}
            assert rows == {"user_grantee_rw": "rw", "user_grantee_ro": "ro"}
            # P-6: usernames resolve for the granted users only — the
            # response never enumerates the rest of the user table.
            assert {g["user_id"]: g["username"] for g in r.json()} == {
                "user_grantee_rw": "user_grantee_rw",
                "user_grantee_ro": "user_grantee_ro",
            }

    def test_owner_grant_and_revoke_roundtrip(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(
            isolated_db, _routers(), _user("user_owner", "user")
        )
        r = client.put("/api/products/prod_1/grants",
                       json={"user_id": "user_target", "level": "ro"})
        assert r.status_code == 200, r.text
        assert r.json()["level"] == "ro"

        # The grantee can now read the product and edit docs.
        app2, client2 = _build_client(
            isolated_db, _routers(), _user("user_target", "user")
        )
        assert client2.get("/api/products/prod_1").status_code == 200
        assert client2.put(
            "/api/products/prod_1/specs/spec_1", json={"content": "x"}
        ).status_code == 200

        # Revoke -> invisible again; DELETE is idempotent.
        assert client.delete(
            "/api/products/prod_1/grants", params={"user_id": "user_target"}
        ).status_code == 200
        assert client.delete(
            "/api/products/prod_1/grants", params={"user_id": "user_target"}
        ).status_code == 200
        assert client2.get("/api/products/prod_1").status_code == 404

    def test_manager_owner_can_grant_on_own_product(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(
            isolated_db, _routers(), _user("user_manager", "manager")
        )
        r = client.put("/api/products/prod_2/grants",
                       json={"user_id": "user_target", "level": "rw"})
        assert r.status_code == 200, r.text

    def test_put_validation(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(
            isolated_db, _routers(), _user("user_admin", "admin")
        )
        base = "/api/products/prod_1/grants"
        assert client.put(base, json={
            "user_id": "user_target", "level": "xx"}).status_code == 400
        assert client.put(base, json={
            "user_id": "user_admin", "level": "ro"}).status_code == 400
        assert client.put(base, json={
            "user_id": "user_owner", "level": "ro"}).status_code == 400
        assert client.put(base, json={
            "user_id": "user_ghost", "level": "ro"}).status_code == 404
        assert client.put("/api/products/prod_ghost/grants", json={
            "user_id": "user_target", "level": "ro"}).status_code == 404


# --- Product ↔ MCP binding guards (P-2) -----------------------------------------
class TestProductMcpBindingGuards:
    """Binding reads need product access (ro); mutations need manage."""

    def test_matrix(self, isolated_db):
        import api.routers.product_mcp as pm
        from api.models import McpServerORM

        _seed(isolated_db)
        with isolated_db.SessionLocal() as db:
            db.add(McpServerORM(
                id="mcp_1", name="srv", transport="http",
                url="http://localhost:9000/sse", enabled=True,
            ))
            db.commit()

        def _client(uid, role):
            _, client = _build_client(isolated_db, [pm], _user(uid, role))
            return client

        # GET: any product access (ro grant) — empty list is fine.
        assert _client("user_grantee_ro", "user").get(
            "/api/products/prod_1/mcp").status_code == 200
        # POST: manage right — an ro grant is 403, an rw grant is 200.
        r = _client("user_grantee_ro", "user").post(
            "/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_1"})
        assert r.status_code == 403, r.text
        r = _client("user_grantee_rw", "user").post(
            "/api/products/prod_1/mcp", json={"mcp_server_id": "mcp_1"})
        assert r.status_code == 200, r.text
        bid = r.json()["id"]
        # PUT/DELETE are manage-gated too.
        assert _client("user_grantee_ro", "user").put(
            f"/api/products/prod_1/mcp/{bid}", json={"enabled": False}
        ).status_code == 403
        assert _client("user_grantee_ro", "user").delete(
            f"/api/products/prod_1/mcp/{bid}").status_code == 403
        # Stranger: invisible -> 404 (never 403 — no existence leak).
        assert _client("user_stranger", "user").get(
            "/api/products/prod_1/mcp").status_code == 404
