#!/usr/bin/env python3
"""SPEC-019 regression: static /api/admin/grants routes vs dynamic /{group}.

FastAPI matches routes in declaration order; the grants routes used to be
declared AFTER ``GET/PUT /{group}``, so ``/api/admin/grants`` was swallowed
by the dynamic handler (404 "Unknown admin group: grants"). These tests pin
the real handlers: GET/PUT reach them (success + 400 validation), DELETE
stays functional.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


def _seed(db_mod):
    """One product, an admin, a grantable user and one ro grant."""
    from api.models import ProductGrantORM, ProductORM, UserORM

    with db_mod.SessionLocal() as db:
        db.add(UserORM(id="user_admin1", username="admin", role="admin",
                       provider="local", created_at=datetime.utcnow()))
        db.add(UserORM(id="user_plain", username="plain", role="user",
                       provider="local", created_at=datetime.utcnow()))
        db.add(ProductORM(id="prod_1", name="P1", description="d",
                          owner_id="user_admin1"))
        db.add(ProductGrantORM(product_id="prod_1", user_id="user_plain",
                               level="ro", granted_by="user_admin1"))
        db.commit()


def _client(db_mod):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.auth import deps as auth_deps
    from api.models import UserORM
    from api.routers import admin as admin_mod

    app = FastAPI()
    app.include_router(admin_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    def _admin_override():
        return UserORM(id="user_admin1", username="admin", role="admin",
                       provider="local", created_at=datetime.utcnow())

    app.dependency_overrides[admin_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.require_admin] = _admin_override
    return TestClient(app)


class TestAdminGrantsRouteOrder:
    def test_get_grants_hits_real_handler(self, isolated_db):
        _seed(isolated_db)
        r = _client(isolated_db).get("/api/admin/grants")
        assert r.status_code == 200, r.text
        rows = {g["user_id"]: g["level"] for g in r.json()}
        assert rows == {"user_plain": "ro"}

    def test_put_grants_hits_real_handler(self, isolated_db):
        _seed(isolated_db)
        r = _client(isolated_db).put(
            "/api/admin/grants",
            json={"product_id": "prod_1", "user_id": "user_plain", "level": "rw"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["level"] == "rw"

    def test_put_grants_validation_400_not_unknown_group(self, isolated_db):
        _seed(isolated_db)
        r = _client(isolated_db).put(
            "/api/admin/grants",
            json={"product_id": "prod_1", "user_id": "user_plain", "level": "xx"},
        )
        assert r.status_code == 400, r.text
        assert "Unknown admin group" not in r.text

    def test_delete_grants_still_works(self, isolated_db):
        _seed(isolated_db)
        client = _client(isolated_db)
        params = {"product_id": "prod_1", "user_id": "user_plain"}
        assert client.delete("/api/admin/grants", params=params).status_code == 200
        # Idempotent delete + the row is really gone.
        assert client.delete("/api/admin/grants", params=params).status_code == 200
        assert client.get("/api/admin/grants").json() == []
