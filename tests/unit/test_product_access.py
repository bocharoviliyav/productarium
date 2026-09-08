#!/usr/bin/env python3
"""Unit tests for the P0-2 product access matrix (roles + per-product grants).

Covers the require_product_access dependency through the products router:
- 401 unauthenticated (AUTH_PROVIDER=local, no cookie);
- 404 invisible product (plain user without grant — indistinguishable from a
  missing product on purpose);
- 403 visible but read-only (ro grant, viewer_global, manager baseline);
- 200 owner / rw grant / manager-rw-grant / admin;
- list visibility (plain user sees owned + granted only).
"""

from __future__ import annotations

import sys
import uuid
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


def _build_client(db_mod, router_mod, *, user=None):
    """App over the isolated DB; override get_current_user with ``user``.

    ``router_mod.get_current_user`` IS ``api.auth.deps.get_current_user``
    (same imported object), so one override also covers the inner dependency
    of require_product_access. When ``user`` is None the real dependency runs
    (used for the 401 case).
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[router_mod.get_db] = _get_test_db
    if user is not None:
        app.dependency_overrides[router_mod.get_current_user] = lambda: user
    return app, TestClient(app)


def _seed(db_mod):
    """prod_1 (owned by user_owner), prod_2 (owned by user_other).

    Grants on prod_1: grantee_rw -> rw, grantee_ro -> ro.
    """
    from api.models import ProductGrantORM, ProductORM, UserORM

    users = [
        _user("user_owner", "user"),
        _user("user_other", "user"),
        _user("user_grantee_rw", "user"),
        _user("user_grantee_ro", "user"),
        _user("user_stranger", "user"),
        _user("user_admin", "admin"),
        _user("user_manager", "manager"),
        _user("user_viewer", "viewer_global"),
    ]
    with db_mod.SessionLocal() as db:
        for u in users:
            db.add(UserORM(**{c: getattr(u, c) for c in (
                "id", "username", "role", "provider", "created_at")}))
        db.add(ProductORM(id="prod_1", name="P1", description="d", owner_id="user_owner"))
        db.add(ProductORM(id="prod_2", name="P2", description="d", owner_id="user_other"))
        db.add(ProductGrantORM(
            product_id="prod_1", user_id="user_grantee_rw",
            level="rw", granted_by="user_owner",
        ))
        db.add(ProductGrantORM(
            product_id="prod_1", user_id="user_grantee_ro",
            level="ro", granted_by="user_owner",
        ))
        db.commit()


def _probe_rw(client):
    """POST a codebase (no repo_url — manual) as the write probe."""
    return client.post(
        "/api/products/prod_1/codebases",
        json={"id": f"cb_{uuid.uuid4().hex[:8]}", "name": "probe"},
    )


# --- resolve_product_access (pure function) ----------------------------------
class TestResolveProductAccess:
    def test_matrix(self, isolated_db):
        from api.auth.deps import resolve_product_access
        from api.models import ProductGrantORM, ProductORM

        _seed(isolated_db)
        with isolated_db.SessionLocal() as db:
            prod = db.get(ProductORM, "prod_1")
            grant_rw = ProductGrantORM(level="rw")
            grant_ro = ProductGrantORM(level="ro")

            assert resolve_product_access(_user("user_admin", "admin"), prod) == "rw"
            assert resolve_product_access(_user("user_viewer", "viewer_global"), prod) == "ro"
            # owner
            assert resolve_product_access(_user("user_owner", "user"), prod) == "rw"
            # plain user with grants
            assert resolve_product_access(_user("user_grantee_rw", "user"), prod, grant_rw) == "rw"
            assert resolve_product_access(_user("user_grantee_ro", "user"), prod, grant_ro) == "ro"
            # plain user without grant
            assert resolve_product_access(_user("user_stranger", "user"), prod) is None
            # manager: rw on own, ro baseline elsewhere, rw-grant upgrades
            own = ProductORM(id="prod_m", name="M", owner_id="user_manager")
            assert resolve_product_access(_user("user_manager", "manager"), own) == "rw"
            assert resolve_product_access(_user("user_manager", "manager"), prod) == "ro"
            assert resolve_product_access(_user("user_manager", "manager"), prod, grant_rw) == "rw"


# --- Endpoint matrix ----------------------------------------------------------
class TestAccessMatrixEndpoints:
    """GET (ro probe) + POST codebases (rw probe) for each principal."""

    CASES = [
        # (role, user_id, expect_get, expect_post)
        ("admin", "user_admin", 200, 200),
        ("user", "user_owner", 200, 200),          # owner
        ("user", "user_grantee_rw", 200, 200),     # rw grant
        ("user", "user_grantee_ro", 200, 403),     # ro grant
        ("viewer_global", "user_viewer", 200, 403),
        ("manager", "user_manager", 200, 403),     # manager baseline ro on prod_1
        ("user", "user_stranger", 404, 404),       # invisible
    ]

    @pytest.mark.parametrize("role,uid,get_code,post_code", CASES)
    def test_matrix(self, isolated_db, role, uid, get_code, post_code):
        from api.routers import products as products_mod

        _seed(isolated_db)
        app, client = _build_client(isolated_db, products_mod, user=_user(uid, role))

        r = client.get("/api/products/prod_1")
        assert r.status_code == get_code, (uid, r.text)
        r = _probe_rw(client)
        assert r.status_code == post_code, (uid, r.text)

    def test_missing_product_404_for_everyone(self, isolated_db):
        from api.routers import products as products_mod

        _seed(isolated_db)
        app, client = _build_client(isolated_db, products_mod, user=_user("user_admin", "admin"))
        assert client.get("/api/products/prod_ghost").status_code == 404

    def test_unauthenticated_401(self, isolated_db, monkeypatch):
        """No session cookie + AUTH_PROVIDER=local -> 401 (P0-9: no fallback)."""
        import api.auth.deps as deps_mod
        from api.routers import products as products_mod

        _seed(isolated_db)
        monkeypatch.setattr(deps_mod, "AUTH_PROVIDER", "local")
        app, client = _build_client(isolated_db, products_mod, user=None)
        r = client.get("/api/products/prod_1")
        assert r.status_code == 401

    def test_stale_session_user_401(self, isolated_db, monkeypatch):
        """A valid-looking session cookie for a DELETED user -> 401 (P0-9)."""
        import api.auth.deps as deps_mod
        from api.auth.tokens import SESSION_COOKIE_NAME, create_session_token
        from api.routers import products as products_mod

        _seed(isolated_db)
        monkeypatch.setattr(deps_mod, "AUTH_PROVIDER", "local")

        # Token for a user id that has no row (deleted / forged claim).
        ghost = _user("user_ghost_deleted", "admin")
        token = create_session_token(ghost)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(products_mod.router)

        def _get_test_db():
            s = isolated_db.SessionLocal()
            try:
                yield s
            finally:
                s.close()

        app.dependency_overrides[products_mod.get_db] = _get_test_db
        client = TestClient(app)
        r = client.get(
            "/api/products/prod_1",
            cookies={SESSION_COOKIE_NAME: token},
        )
        assert r.status_code == 401


class TestCreateProductRoles:
    def test_plain_user_cannot_create(self, isolated_db):
        from api.routers import products as products_mod

        _seed(isolated_db)
        app, client = _build_client(isolated_db, products_mod, user=_user("user_stranger", "user"))
        r = client.post(
            "/api/products",
            json={"id": "prod_new", "name": "N", "description": "d",
                  "codebases": [], "specs": [], "links": []},
        )
        assert r.status_code == 403

    def test_manager_creates_and_becomes_owner(self, isolated_db):
        from api.routers import products as products_mod

        _seed(isolated_db)
        app, client = _build_client(isolated_db, products_mod, user=_user("user_manager", "manager"))
        r = client.post(
            "/api/products",
            json={"id": "prod_new", "name": "N", "description": "d",
                  "codebases": [], "specs": [], "links": []},
        )
        assert r.status_code == 200, r.text
        assert r.json()["owner_id"] == "user_manager"


# --- List visibility (P1-16, bare-list shape) --------------------------------
class TestListVisibility:
    """GET /api/products is visibility-filtered per user; the response stays
    a bare JSON array with the filtered total in the X-Total-Count header."""

    def test_plain_user_sees_owned_and_granted_only(self, isolated_db):
        from api.routers import products as products_mod

        _seed(isolated_db)
        # stranger: no products owned, no grants -> empty list
        app, client = _build_client(
            isolated_db, products_mod, user=_user("user_stranger", "user")
        )
        r = client.get("/api/products")
        assert r.status_code == 200
        assert r.json() == []
        assert r.headers["X-Total-Count"] == "0"

        # rw grantee: sees exactly the granted product
        app, client = _build_client(
            isolated_db, products_mod, user=_user("user_grantee_rw", "user")
        )
        r = client.get("/api/products")
        assert r.status_code == 200
        assert [p["id"] for p in r.json()] == ["prod_1"]
        assert r.headers["X-Total-Count"] == "1"

        # owner: sees the owned product
        app, client = _build_client(
            isolated_db, products_mod, user=_user("user_owner", "user")
        )
        r = client.get("/api/products")
        assert [p["id"] for p in r.json()] == ["prod_1"]
        assert r.headers["X-Total-Count"] == "1"

    def test_privileged_roles_see_all(self, isolated_db):
        from api.routers import products as products_mod

        _seed(isolated_db)
        for uid, role in (
            ("user_admin", "admin"),
            ("user_viewer", "viewer_global"),
            ("user_manager", "manager"),
        ):
            app, client = _build_client(
                isolated_db, products_mod, user=_user(uid, role)
            )
            r = client.get("/api/products")
            assert r.status_code == 200
            assert sorted(p["id"] for p in r.json()) == ["prod_1", "prod_2"]
            assert r.headers["X-Total-Count"] == "2"
