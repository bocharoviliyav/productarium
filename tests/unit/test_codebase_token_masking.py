#!/usr/bin/env python3
"""Unit tests for write-only codebase git tokens (P0-2).

Invariants:
- tokens are accepted on create/update but NEVER serialized back (no ``token``
  key in any response — only the boolean ``has_token``);
- the stored value is Fernet ciphertext, not the plaintext;
- :func:`get_codebase_token` decrypts for internal consumers (docgen);
- an empty/absent token on re-send keeps the stored one (merge rule);
- ``migrate_plaintext_tokens`` encrypts legacy plaintext rows in place.
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


SECRET = "ghp_supersecret_token_123"


class TestTokenMasking:
    def test_token_absent_from_responses_and_encrypted_at_rest(self, isolated_db):
        from api.config.settings import is_encrypted_secret
        from api.models import CodebaseORM

        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))

        r = client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo",
                  "repo_url": "https://github.com/acme/repo.git",
                  "token": SECRET},
        )
        assert r.status_code == 200, r.text
        cb = r.json()["codebases"][0]
        assert "token" not in cb                      # never serialized
        assert cb["has_token"] is True

        with isolated_db.SessionLocal() as db:
            row = db.get(CodebaseORM, "cb_1")
            assert row.token != SECRET                 # ciphertext at rest
            assert is_encrypted_secret(row.token)      # Fernet gAAAA… prefix

        # GET product: still masked.
        r = client.get("/api/products/prod_1")
        cb = r.json()["codebases"][0]
        assert "token" not in cb
        assert cb["has_token"] is True

    def test_no_token_flag_false(self, isolated_db):
        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo",
                  "repo_url": "https://github.com/acme/repo.git"},
        )
        cb = client.get("/api/products/prod_1").json()["codebases"][0]
        assert cb["has_token"] is False

    def test_get_codebase_token_decrypts(self, isolated_db):
        from api.models import CodebaseORM
        from api.repositories.product_repo import get_codebase_token

        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo", "token": SECRET},
        )
        with isolated_db.SessionLocal() as db:
            row = db.get(CodebaseORM, "cb_1")
            assert get_codebase_token(row) == SECRET
            assert get_codebase_token(CodebaseORM(id="cb_x")) is None

    def test_empty_token_on_re_add_keeps_stored(self, isolated_db):
        from api.models import CodebaseORM
        from api.repositories.product_repo import get_codebase_token

        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo", "token": SECRET},
        )
        # Re-send the same codebase with an empty token (frontend write-only
        # field left blank): the stored token must survive.
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo", "token": ""},
        )
        cb = client.get("/api/products/prod_1").json()["codebases"][0]
        assert cb["has_token"] is True
        with isolated_db.SessionLocal() as db:
            assert get_codebase_token(db.get(CodebaseORM, "cb_1")) == SECRET

    def test_token_rotation(self, isolated_db):
        from api.models import CodebaseORM
        from api.repositories.product_repo import get_codebase_token

        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo", "token": "old-token"},
        )
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo", "token": "new-token"},
        )
        with isolated_db.SessionLocal() as db:
            assert get_codebase_token(db.get(CodebaseORM, "cb_1")) == "new-token"

    def test_full_product_replace_keeps_token(self, isolated_db):
        from api.models import CodebaseORM
        from api.repositories.product_repo import get_codebase_token

        _seed(isolated_db)
        app, client = _build_client(isolated_db, _user("user_owner", "user"))
        client.post(
            "/api/products/prod_1/codebases",
            json={"id": "cb_1", "name": "repo", "token": SECRET},
        )
        r = client.put(
            "/api/products/prod_1",
            json={
                "id": "prod_1", "name": "P1", "description": "d",
                "owner_id": "user_owner",
                "codebases": [{"id": "cb_1", "name": "repo"}],  # no token key
                "specs": [], "links": [],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["codebases"][0]["has_token"] is True
        with isolated_db.SessionLocal() as db:
            assert get_codebase_token(db.get(CodebaseORM, "cb_1")) == SECRET


class TestMigration:
    def test_migrate_plaintext_tokens(self, isolated_db):
        """Legacy plaintext rows are encrypted in place; ciphertext skipped."""
        from api.config.settings import is_encrypted_secret
        from api.models import CodebaseORM
        from api.repositories.product_repo import migrate_plaintext_tokens

        _seed(isolated_db)
        with isolated_db.SessionLocal() as db:
            db.add(CodebaseORM(id="cb_plain", product_id="prod_1",
                               name="legacy", token="legacy-plaintext"))
            db.commit()

        migrated = migrate_plaintext_tokens(isolated_db.SessionLocal())
        assert migrated == 1
        with isolated_db.SessionLocal() as db:
            row = db.get(CodebaseORM, "cb_plain")
            assert is_encrypted_secret(row.token)
        # Idempotent: second pass finds nothing to migrate.
        assert migrate_plaintext_tokens(isolated_db.SessionLocal()) == 0

    def test_migration_non_fatal_on_bad_db(self, monkeypatch):
        from api.repositories import product_repo

        # A failing SessionLocal must not raise (startup contract: non-fatal).
        def _boom():
            raise RuntimeError("db down")

        monkeypatch.setattr(
            "api.db.SessionLocal", _boom, raising=False
        )
        assert product_repo.migrate_plaintext_tokens(None) == 0
