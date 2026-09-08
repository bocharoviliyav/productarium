#!/usr/bin/env python3
"""Unit tests for the server-owned verified triple (P0-5).

The verified/verified_by/verified_at fields on codebases/specs/links must be
set ONLY by the verify endpoints (owner/admin). A client re-sending an entity
via POST (create/replace) or PUT (full product replace) can neither forge
verification nor reset it.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _user(uid: str, role: str):
    from api.models import UserORM

    return UserORM(
        id=uid, username=uid, role=role, provider="local",
        created_at=datetime.utcnow(),
    )


def _build_client(db_mod, user):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import products as products_mod

    app = FastAPI()
    app.include_router(products_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[products_mod.get_db] = _get_test_db
    app.dependency_overrides[products_mod.get_current_user] = lambda: user
    return app, TestClient(app)


def _seed(db_mod):
    from api.models import ProductORM, UserORM

    with db_mod.SessionLocal() as db:
        db.add(UserORM(id="user_owner", username="owner", role="user",
                       provider="local", created_at=datetime.utcnow()))
        db.add(ProductORM(id="prod_1", name="P1", description="d",
                          owner_id="user_owner"))
        db.commit()


def _product_payload(**overrides):
    payload = {
        "id": "prod_1",
        "name": "P1",
        "description": "d",
        "owner_id": "user_owner",
        "codebases": [],
        "specs": [],
        "links": [],
    }
    payload.update(overrides)
    return payload


class TestVerifiedServerOwned:
    def test_create_cannot_forge_verified(self, isolated_db):
        """POST codebase with verified=True in the payload stays unverified."""
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))

        r = client.post(
            "/api/products/prod_1/codebases",
            json={
                "id": "cb_1", "name": "repo",
                "repo_url": "https://github.com/acme/repo.git",
                "verified": True, "verified_by": "user_owner",
                "verified_at": "2024-01-01T00:00:00Z",
            },
        )
        assert r.status_code == 200, r.text
        cb = r.json()["codebases"][0]
        assert cb["verified"] is False
        assert cb["verified_by"] is None

    def test_verify_endpoint_sets_triple(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo"},
        )

        r = client.post("/api/products/prod_1/codebases/cb_1/verify")
        assert r.status_code == 200, r.text
        cb = r.json()["codebases"][0]
        assert cb["verified"] is True
        assert cb["verified_by"] == "user_owner"
        assert cb["verified_at"] is not None

    def test_full_replace_preserves_verified(self, isolated_db):
        """PUT /products/{id} re-sending verified=False cannot reset it."""
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo"},
        )
        client.post("/api/products/prod_1/codebases/cb_1/verify")

        # Client tries to un-verify (and change the name while at it).
        r = client.put(
            "/api/products/prod_1",
            json=_product_payload(codebases=[
                {"id": "cb_1", "name": "repo-renamed", "verified": False,
                 "verified_by": None, "verified_at": None},
            ]),
        )
        assert r.status_code == 200, r.text
        cb = r.json()["codebases"][0]
        assert cb["name"] == "repo-renamed"          # client-owned field applied
        assert cb["verified"] is True                 # server-owned preserved
        assert cb["verified_by"] == "user_owner"
        assert cb["verified_at"] is not None

    def test_full_replace_cannot_forge_new_verified(self, isolated_db):
        """A NEW child sent with verified=True via PUT stays unverified."""
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))

        r = client.put(
            "/api/products/prod_1",
            json=_product_payload(specs=[
                {"id": "spec_1", "name": "spec", "kind": "openapi",
                 "verified": True, "verified_by": "user_owner"},
            ]),
        )
        assert r.status_code == 200, r.text
        spec = r.json()["specs"][0]
        assert spec["verified"] is False
        assert spec["verified_by"] is None

    def test_re_add_same_id_preserves_verified(self, isolated_db):
        """POST codebase with an existing id replaces content but keeps state."""
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo"},
        )
        client.post("/api/products/prod_1/codebases/cb_1/verify")

        r = client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo-v2", "verified": False},
        )
        assert r.status_code == 200, r.text
        cb = r.json()["codebases"][0]
        assert cb["name"] == "repo-v2"
        assert cb["verified"] is True

    def test_non_owner_cannot_verify(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_stranger", "user"))
        with isolated_db.SessionLocal() as db:
            from api.models import CodebaseORM

            db.add(CodebaseORM(id="cb_1", product_id="prod_1", name="repo"))
            db.commit()

        r = client.post("/api/products/prod_1/codebases/cb_1/verify")
        assert r.status_code == 403
