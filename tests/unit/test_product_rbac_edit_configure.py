#!/usr/bin/env python3
"""Unit tests for the SPEC-023 product-settings permission matrix.

- edit      — PATCH /api/products/{id} (name/description/owner transfer):
  owner || admin only.
- configure — GET/PUT/DELETE /api/products/{id}/schedules and .../grants
  (schedules + product users): owner || manager || admin (manager is global,
  like Grafana).

Neither right is reachable through grants: rw stays artifact-CRUD + docs,
viewer_global stays pure read, strangers stay invisible (404).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


def _user(uid: str, role: str):
    from api.models import UserORM

    return UserORM(
        id=uid, username=uid, role=role, provider="local",
        created_at=datetime.utcnow(),
    )


def _routers():
    from api.routers import products as products_mod
    from api.routers import schedules as schedules_mod

    return [products_mod, schedules_mod]


def _build_client(db_mod, router_mods, user):
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
    """prod_1 (owner user_owner) with one codebase row.

    Grants on prod_1: user_grantee_rw -> rw, user_grantee_ro -> ro.
    """
    from api.models import (
        CodebaseORM,
        ProductGrantORM,
        ProductORM,
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
        db.add(CodebaseORM(id="cb_1", product_id="prod_1", name="svc"))
        db.add(ProductGrantORM(
            product_id="prod_1", user_id="user_grantee_rw",
            level="rw", granted_by="user_owner",
        ))
        db.add(ProductGrantORM(
            product_id="prod_1", user_id="user_grantee_ro",
            level="ro", granted_by="user_owner",
        ))
        db.commit()


# --- edit: PATCH /api/products/{id} ------------------------------------------
class TestEditMatrix:
    CASES = [
        ("admin", "user_admin", 200),
        ("user", "user_owner", 200),        # owner
        ("manager", "user_manager", 403),   # manager is NOT an editor
        ("user", "user_grantee_rw", 403),   # rw grant is not edit
        ("user", "user_grantee_ro", 403),
        ("viewer_global", "user_viewer", 403),
        ("user", "user_stranger", 404),     # invisible
    ]

    @pytest.mark.parametrize("role,uid,code", CASES)
    def test_patch_product(self, isolated_db, role, uid, code):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _routers(), _user(uid, role))
        r = client.patch("/api/products/prod_1", json={"name": "P1x"})
        assert r.status_code == code, (uid, r.text)

    def test_owner_transfer_by_edit_right(self, isolated_db):
        _seed(isolated_db)
        _, owner = _build_client(
            isolated_db, _routers(), _user("user_owner", "user"))
        r = owner.patch("/api/products/prod_1",
                        json={"owner_id": "user_target"})
        assert r.status_code == 200, r.text
        assert r.json()["owner_id"] == "user_target"

        # The rw grantee cannot transfer ownership back.
        _, rw = _build_client(
            isolated_db, _routers(), _user("user_grantee_rw", "user"))
        assert rw.patch("/api/products/prod_1",
                        json={"owner_id": "user_owner"}).status_code == 403


# --- configure: schedules + product users ------------------------------------
class TestConfigureMatrix:
    CASES = [
        ("admin", "user_admin", 200),
        ("user", "user_owner", 200),        # owner
        ("manager", "user_manager", 200),   # manager is global
        ("user", "user_grantee_rw", 403),   # rw grant is not configure
        ("user", "user_grantee_ro", 403),
        ("viewer_global", "user_viewer", 403),
        ("user", "user_stranger", 404),     # invisible
    ]

    @pytest.mark.parametrize("role,uid,code", CASES)
    def test_schedules_and_grants(self, isolated_db, role, uid, code):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _routers(), _user(uid, role))

        base = "/api/products/prod_1/schedules"
        body = {"entity_type": "codebase", "entity_id": "cb_1",
                "cron_expr": "0 2 * * *", "enabled": True}
        assert client.get(base).status_code == code
        assert client.put(base, json=body).status_code == code
        assert client.delete(
            base, params={"entity_type": "codebase", "entity_id": "cb_1"}
        ).status_code == code

        gbase = "/api/products/prod_1/grants"
        assert client.get(gbase).status_code == code
        assert client.put(gbase, json={
            "user_id": "user_target", "level": "ro"}).status_code == code
        assert client.delete(
            gbase, params={"user_id": "user_target"}).status_code == code

    def test_manager_can_configure_foreign_product(self, isolated_db):
        """Manager on a product they do NOT own: schedules + users work."""
        _seed(isolated_db)
        _, client = _build_client(
            isolated_db, _routers(), _user("user_manager", "manager"))
        r = client.put("/api/products/prod_1/schedules", json={
            "entity_type": "codebase", "entity_id": "cb_1",
            "cron_expr": "0 2 * * *", "enabled": True,
        })
        assert r.status_code == 200, r.text
        r = client.put("/api/products/prod_1/grants", json={
            "user_id": "user_target", "level": "rw"})
        assert r.status_code == 200, r.text


# --- rw grant keeps its old rights -------------------------------------------
class TestRwGrantStillManagesArtifacts:
    def test_crud_and_docs_yes_settings_no(self, isolated_db):
        _seed(isolated_db)
        _, client = _build_client(
            isolated_db, _routers(), _user("user_grantee_rw", "user"))
        # manage: artifact CRUD still rw.
        assert client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_new", "name": "new"},
        ).status_code == 200
        # docs: page edit still allowed.
        assert client.put(
            "/api/products/prod_1/codebases/cb_1",
            json={"generated_docs": "edited"},
        ).status_code == 200
        # settings: closed by SPEC-023.
        assert client.patch(
            "/api/products/prod_1", json={"name": "Nope"}
        ).status_code == 403
