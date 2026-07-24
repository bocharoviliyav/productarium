#!/usr/bin/env python3
"""Unit tests for the admin + public routers (plan sections D + H).

Covers:
- admin settings roundtrip (secrets encrypted on save, redacted + hasKey on read)
- admin users list + promote/demote
- admin API token create (plaintext returned once) / list / delete
- public knowledge export with the verified-only filter (markdown + json)
- public ask returns 501 when the expert agent is not yet present
- public push returns 501 when integrations are not yet present
- full API-token create -> verify roundtrip (create via admin, then call a
  public endpoint with the raw Bearer token)

Runs under pytest with an isolated in-memory SQLite DB per test
(``DB_PROVIDER=sqlite``). The routers' captured ``get_db`` is overridden to
yield sessions from the reloaded test engine so DB access is isolated without
touching the real Postgres. ``api.settings_store`` reads ``SessionLocal``
lazily inside each function, so it picks up the reloaded test engine
automatically.
"""

from __future__ import annotations

import importlib
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# --- Shared fixtures --------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Isolated in-memory SQLite DB per test + stable SETTINGS_SECRET_KEY."""
    monkeypatch.setenv("DB_PROVIDER", "sqlite")
    monkeypatch.setenv("DB_HOST", str(tmp_path))
    monkeypatch.setenv("DB_NAME", str(tmp_path / "test.db"))
    monkeypatch.setenv("DB_USERNAME", "")
    monkeypatch.setenv("DB_PASSWORD", "")
    monkeypatch.setenv("AUTH_PROVIDER", "local")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    from cryptography.fernet import Fernet
    monkeypatch.setenv("SETTINGS_SECRET_KEY", Fernet.generate_key().decode())
    yield


def _setup_db():
    """Set up an isolated, cross-thread SQLite DB and init the schema.

    api.db's default ``sqlite:///:memory:`` uses ``SingletonThreadPool`` (one
    connection per thread), so a request served by FastAPI's worker thread
    would see an empty DB separate from the main thread where ``init_db`` ran.
    We therefore rebind ``api.db.engine`` / ``SessionLocal`` to a
    ``StaticPool`` engine (one shared connection, ``check_same_thread=False``)
    so the worker thread, the get_db override, and ``api.settings_store`` (which
    imports ``SessionLocal`` lazily) all see the same schema/data.

    Does NOT reload api.auth.deps or the router modules, so their captured
    ``get_db`` stays the same object and a single dependency override covers
    every Depends(get_db).
    """
    import api.db as db
    importlib.reload(db)  # reset _db_ready + module-level engine/SessionLocal
    import api.settings_store as ss
    importlib.reload(ss)

    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        future=True,
    )
    db.engine = engine
    db.SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db.init_db()  # create_all on the rebound engine; migration is non-fatal
    return db


def _admin_user():
    """A fixed admin UserORM used to override require_admin."""
    from api.models import UserORM
    return UserORM(
        id="user_admin1",
        username="admin",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


def _api_token():
    """A fixed ApiTokenORM used to override require_api_token (verify-bypass)."""
    from api.models import ApiTokenORM
    return ApiTokenORM(
        id="tok_fixed",
        user_id="user_admin1",
        token_hash="x" * 64,
        name="fixed",
        created_at=datetime.utcnow(),
    )


def _build_app(db_mod) -> tuple[FastAPI, TestClient]:
    """Build a minimal app with both routers + a get_db override to the test DB."""
    from api.routers import admin as admin_mod
    from api.routers import public as public_mod
    from api.auth import deps as auth_deps

    app = FastAPI()
    app.include_router(admin_mod.router)
    app.include_router(public_mod.router)

    # The router/auth deps captured get_db at import time (the SAME object across
    # admin.py, public.py, and auth.deps). Override it once to yield sessions
    # from the reloaded test engine so all DB access is isolated.
    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    # Override on every reference of the captured get_db (all the same object,
    # but set on each to be defensive).
    app.dependency_overrides[admin_mod.get_db] = _get_test_db
    app.dependency_overrides[public_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    return app, TestClient(app)


# --- Admin settings roundtrip -----------------------------------------------
class TestAdminSettings:
    def test_models_settings_roundtrip_encrypt_redact(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_admin] = _admin_user

        # Save a mix of plaintext + secret settings for the models group.
        resp = client.put(
            "/api/admin/models",
            json={
                "models.expert.provider": "ollama",
                "models.expert.base_url": "http://x:11434/v1",
                "models.expert.api_key": "secret-xyz",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert set(body["saved"]) == {
            "models.expert.provider",
            "models.expert.base_url",
            "models.expert.api_key",
        }

        # GET back: secrets redacted + hasKey; plaintext fields visible.
        resp = client.get("/api/admin/models")
        assert resp.status_code == 200
        data = resp.json()
        settings = data["settings"]
        assert settings["models.expert.provider"] == {
            "value": "ollama", "encrypted": False, "hasKey": True,
        }
        assert settings["models.expert.base_url"] == {
            "value": "http://x:11434/v1", "encrypted": False, "hasKey": True,
        }
        # api_key is a secret: value redacted to null, encrypted=True, hasKey=True
        assert settings["models.expert.api_key"] == {
            "value": None, "encrypted": True, "hasKey": True,
        }
        # Resolved view also redacts the key but reports hasApiKey.
        resolved = data["resolved"]["expert"]
        assert resolved["provider"] == "ollama"
        assert resolved["base_url"] == "http://x:11434/v1"
        assert resolved["api_key"] is None
        assert resolved["hasApiKey"] is True

        # The secret actually decrypts back via the settings store.
        import api.settings_store as ss
        assert ss.get_setting("models.expert.api_key") == "secret-xyz"
        assert ss.get_setting("models.expert.provider") == "ollama"

    def test_put_ignores_keys_outside_group(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_admin] = _admin_user

        resp = client.put(
            "/api/admin/models",
            json={
                "models.expert.provider": "ollama",
                "git.github.token": "should-be-ignored",  # wrong group
            },
        )
        assert resp.status_code == 200
        assert resp.json()["saved"] == ["models.expert.provider"]
        # The git token was NOT written.
        import api.settings_store as ss
        assert ss.get_setting("git.github.token") is None

    def test_unknown_group_404(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_admin] = _admin_user
        assert client.get("/api/admin/nope").status_code == 404


# --- Admin users ------------------------------------------------------------
class TestAdminUsers:
    def test_list_and_promote(self):
        db_mod = _setup_db()
        from api.models import UserORM
        with db_mod.SessionLocal() as db:
            db.add(UserORM(id="user_bob", username="bob", role="user"))
            db.commit()

        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_admin] = _admin_user

        resp = client.get("/api/admin/users")
        assert resp.status_code == 200
        users = resp.json()["users"]
        assert any(u["id"] == "user_bob" and u["role"] == "user" for u in users)

        resp = client.put(
            "/api/admin/users", json={"user_id": "user_bob", "role": "admin"}
        )
        assert resp.status_code == 200
        out = resp.json()["user"]
        assert out["id"] == "user_bob"
        assert out["role"] == "admin"

        # Invalid role is rejected.
        resp = client.put(
            "/api/admin/users", json={"user_id": "user_bob", "role": "superuser"}
        )
        assert resp.status_code == 400

    def test_promote_missing_user_404(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_admin] = _admin_user
        resp = client.put(
            "/api/admin/users", json={"user_id": "ghost", "role": "admin"}
        )
        assert resp.status_code == 404


# --- Admin API tokens -------------------------------------------------------
class TestAdminApiTokens:
    def test_create_returns_plaintext_once_then_list_then_delete(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_admin] = _admin_user

        # Create: plaintext token returned once.
        resp = client.post("/api/admin/apitokens", json={"name": "ci"})
        assert resp.status_code == 201
        tok = resp.json()
        assert tok["name"] == "ci"
        assert tok["token"]  # plaintext present
        token_id = tok["id"]
        plaintext = tok["token"]

        # List: token present but plaintext NOT returned.
        resp = client.get("/api/admin/apitokens")
        assert resp.status_code == 200
        toks = resp.json()["tokens"]
        assert any(t["id"] == token_id for t in toks)
        assert all(t["token"] is None for t in toks), "plaintext must not be listed"

        # The stored hash matches sha256(plaintext) (what require_api_token uses).
        import hashlib
        from api.models import ApiTokenORM
        with db_mod.SessionLocal() as db:
            row = db.get(ApiTokenORM, token_id)
            assert row is not None
            assert row.token_hash == hashlib.sha256(plaintext.encode("utf-8")).hexdigest()

        # Delete.
        resp = client.delete(f"/api/admin/apitokens/{token_id}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        # Deleting again -> 404.
        assert client.delete(f"/api/admin/apitokens/{token_id}").status_code == 404

    def test_put_apitokens_rejected(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_admin] = _admin_user
        # PUT on the apitokens group is not a settings save -> 400 with guidance.
        assert client.put("/api/admin/apitokens", json={}).status_code == 400


# --- Public knowledge export (verified filter) ------------------------------
class TestPublicKnowledgeExport:
    def _seed_product_with_mixed_verified(self, db_mod):
        from api.models import ArtifactORM, KnowledgeNodeORM, ProductORM
        with db_mod.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme", summary="sum"))
            db.flush()
            db.add(ArtifactORM(
                id="art_v", product_id="prod_1", name="svc-v", type="spec",
                kind="openapi", verified=True, generated_docs="docs A",
                source="manual",
            ))
            db.add(ArtifactORM(
                id="art_u", product_id="prod_1", name="svc-u", type="spec",
                verified=False, generated_docs="docs B", source="manual",
            ))
            db.add(KnowledgeNodeORM(
                id="node_v", product_id="prod_1", title="Page V", slug="page-v",
                content_md="node X", verified=True, source="manual",
            ))
            db.add(KnowledgeNodeORM(
                id="node_u", product_id="prod_1", title="Page U", slug="page-u",
                content_md="node Y", verified=False, source="manual",
            ))
            db.commit()

    def test_json_export_only_verified(self):
        db_mod = _setup_db()
        self._seed_product_with_mixed_verified(db_mod)
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_api_token] = _api_token

        resp = client.get("/api/public/products/prod_1/knowledge?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["verified_only"] is True
        assert data["product"]["id"] == "prod_1"
        arts = [a["id"] for a in data["artifacts"]]
        nodes = [n["id"] for n in data["nodes"]]
        assert arts == ["art_v"]
        assert nodes == ["node_v"]
        assert data["artifacts"][0]["generated_docs"] == "docs A"
        assert data["nodes"][0]["content_md"] == "node X"

    def test_markdown_export_only_verified(self):
        db_mod = _setup_db()
        self._seed_product_with_mixed_verified(db_mod)
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_api_token] = _api_token

        resp = client.get("/api/public/products/prod_1/knowledge")
        assert resp.status_code == 200
        assert "text/markdown" in resp.headers.get("content-type", "")
        md = resp.text
        assert "Verified Knowledge" in md
        assert "docs A" in md
        assert "node X" in md
        # Unverified content is excluded.
        assert "docs B" not in md
        assert "node Y" not in md

    def test_product_not_found_404(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_api_token] = _api_token
        assert client.get("/api/public/products/ghost/knowledge").status_code == 404

    def test_missing_token_401(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        # No override on require_api_token -> real dep runs; no Bearer header -> 401.
        assert client.get("/api/public/products/prod_1/knowledge").status_code == 401


# --- Public ask / push degrade to 501 when deps missing ---------------------
# After merge, api.expert_agent and api.integrations both exist, so the 501
# fallback can only be exercised by simulating the module being unimportable
# (the real graceful path when a parallel module has not landed yet).
import importlib as _importlib
import sys as _sys


class TestPublicAskPush501:
    def _with_module_blocked(self, modname: str):
        """Context manager-ish: block `import <modname>` by injecting a
        broken module + raising ImportError on re-import, then restore."""
        saved = _sys.modules.get(modname)
        _sys.modules[modname] = None  # forces ImportError on `import modname`
        return saved

    def _restore(self, modname, saved):
        if saved is None:
            _sys.modules.pop(modname, None)
        else:
            _sys.modules[modname] = saved

    def test_ask_501_when_expert_agent_missing(self):
        db_mod = _setup_db()
        from api.models import ProductORM
        with db_mod.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme"))
            db.commit()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_api_token] = _api_token

        saved = self._with_module_blocked("api.expert_agent")
        try:
            resp = client.post(
                "/api/public/products/prod_1/ask", json={"query": "hi"}
            )
            assert resp.status_code == 501
            assert "Expert agent not available" in resp.json()["detail"]
        finally:
            self._restore("api.expert_agent", saved)
            _importlib.import_module("api.expert_agent")

    def test_push_501_when_integrations_missing(self):
        db_mod = _setup_db()
        from api.models import ProductORM
        with db_mod.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme"))
            db.commit()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        app.dependency_overrides[auth_deps.require_api_token] = _api_token

        saved = self._with_module_blocked("api.integrations")
        try:
            resp = client.post(
                "/api/public/products/prod_1/push",
                json={"target": "confluence"},
            )
            assert resp.status_code == 501
            assert "Integrations not available" in resp.json()["detail"]
        finally:
            self._restore("api.integrations", saved)
            _importlib.import_module("api.integrations")


# --- Full API-token create -> verify roundtrip ------------------------------
class TestApiTokenCreateVerifyRoundtrip:
    def test_create_then_use_bearer_to_access_public(self):
        db_mod = _setup_db()
        from api.models import ProductORM
        with db_mod.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme", summary="s"))
            db.add(_admin_user())  # so the token's user_id exists
            db.commit()

        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        # Admin create endpoint: bypass require_admin (we test token verify, not admin auth).
        app.dependency_overrides[auth_deps.require_admin] = _admin_user
        # Do NOT override require_api_token: the real dep must verify the Bearer token.

        # Create a token via the admin endpoint (uses overridden get_db -> test DB).
        resp = client.post("/api/admin/apitokens", json={"name": "ci"})
        assert resp.status_code == 201
        plaintext = resp.json()["token"]
        assert plaintext

        # Use the raw token as a Bearer credential against the public API.
        # The real require_api_token hashes it (sha256) and finds the row in the
        # test DB, then updates last_used_at.
        resp = client.get(
            "/api/public/products/prod_1/knowledge?format=json",
            headers={"Authorization": f"Bearer {plaintext}"},
        )
        assert resp.status_code == 200
        assert resp.json()["product"]["id"] == "prod_1"

        # A wrong token is rejected.
        resp = client.get(
            "/api/public/products/prod_1/knowledge",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 401

        # last_used_at was updated by require_api_token.
        from api.models import ApiTokenORM
        with db_mod.SessionLocal() as db:
            row = db.query(ApiTokenORM).first()
            assert row is not None
            assert row.last_used_at is not None
