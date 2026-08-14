#!/usr/bin/env python3
"""Unit tests for the Productarium auth flows (first-run setup, login,
password change, password reset by token, admin user creation).

Runs under pytest with an isolated in-memory SQLite DB per test, mirroring
the harness in test_admin_public.py.
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


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
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
    import api.db as db

    importlib.reload(db)
    import api.config.settings as ss

    importlib.reload(ss)
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        future=True,
    )
    db.engine = engine
    db.SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db.init_db()
    return db


def _build_app(db_mod) -> tuple[FastAPI, TestClient]:
    from api.auth import deps as auth_deps
    from api.auth import router as auth_router
    from api.routers import admin as admin_mod

    app = FastAPI()
    app.include_router(auth_router.router)
    app.include_router(admin_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[auth_router.get_db] = _get_test_db
    app.dependency_overrides[admin_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    return app, TestClient(app)


def _admin_override():
    from api.models import UserORM

    return UserORM(
        id="user_admin1",
        username="admin",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


# --- First-run setup --------------------------------------------------------
class TestSetup:
    def test_setup_status_required_when_no_users(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        resp = client.get("/api/auth/setup-status")
        assert resp.status_code == 200
        assert resp.json()["setup_required"] is True
        assert resp.json()["auth_provider"] == "local"

    def test_setup_creates_first_admin_and_signs_in(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        resp = client.post(
            "/api/auth/setup",
            json={"username": "root", "password": "secret-123"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["username"] == "root"
        assert body["role"] == "admin"
        # Session cookie set.
        assert "productarium_session" in resp.cookies

        # setup-status now reports not required.
        assert client.get("/api/auth/setup-status").json()["setup_required"] is False

        # Login works with the new credentials.
        login = client.post(
            "/api/auth/login", json={"username": "root", "password": "secret-123"}
        )
        assert login.status_code == 200
        assert login.json()["username"] == "root"

    def test_setup_rejected_when_users_exist(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        # First setup succeeds.
        assert client.post(
            "/api/auth/setup", json={"username": "root", "password": "secret-123"}
        ).status_code == 200
        # Second setup is rejected (409).
        resp = client.post(
            "/api/auth/setup", json={"username": "other", "password": "secret-123"}
        )
        assert resp.status_code == 409


# --- Login ------------------------------------------------------------------
class TestLogin:
    def test_login_wrong_password_401(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        client.post("/api/auth/setup", json={"username": "root", "password": "secret-123"})
        resp = client.post(
            "/api/auth/login", json={"username": "root", "password": "wrong"}
        )
        assert resp.status_code == 401

    def test_login_unknown_user_401(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        client.post("/api/auth/setup", json={"username": "root", "password": "secret-123"})
        resp = client.post(
            "/api/auth/login", json={"username": "ghost", "password": "x"}
        )
        assert resp.status_code == 401


# --- Change password --------------------------------------------------------
class TestChangePassword:
    def test_change_password_then_login_with_new(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        # Create the admin via setup.
        client.post("/api/auth/setup", json={"username": "root", "password": "secret-123"})
        from api.auth import deps as auth_deps
        from api.models import UserORM

        # Resolve the persisted admin id to drive get_current_user override.
        with db_mod.SessionLocal() as db:
            admin = db.query(UserORM).filter(UserORM.username == "root").first()
            admin_id = admin.id

        app.dependency_overrides[auth_deps.get_current_user] = lambda: UserORM(
            id=admin_id, username="root", role="admin", provider="local",
            created_at=datetime.utcnow(),
        )

        resp = client.post(
            "/api/auth/change-password",
            json={"old_password": "secret-123", "new_password": "new-secret-456"},
        )
        assert resp.status_code == 200, resp.text

        # Old password no longer works; new one does.
        assert client.post(
            "/api/auth/login", json={"username": "root", "password": "secret-123"}
        ).status_code == 401
        assert client.post(
            "/api/auth/login", json={"username": "root", "password": "new-secret-456"}
        ).status_code == 200

    def test_change_password_wrong_old_401(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        client.post("/api/auth/setup", json={"username": "root", "password": "secret-123"})
        from api.auth import deps as auth_deps
        from api.models import UserORM

        with db_mod.SessionLocal() as db:
            admin_id = db.query(UserORM).filter(UserORM.username == "root").first().id
        app.dependency_overrides[auth_deps.get_current_user] = lambda: UserORM(
            id=admin_id, username="root", role="admin", provider="local",
            created_at=datetime.utcnow(),
        )
        resp = client.post(
            "/api/auth/change-password",
            json={"old_password": "wrong", "new_password": "new-secret-456"},
        )
        assert resp.status_code == 401


# --- Admin user creation + reset-by-token -----------------------------------
class TestAdminCreateUserAndReset:
    def test_admin_create_user_returns_temp_password_and_reset_token(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps

        app.dependency_overrides[auth_deps.require_admin] = _admin_override

        resp = client.post(
            "/api/admin/users",
            json={"username": "alice", "role": "user", "must_change_password": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["user"]["username"] == "alice"
        assert body["temp_password"]
        assert body["reset_token"]

        # The temp password works for login.
        assert client.post(
            "/api/auth/login",
            json={"username": "alice", "password": body["temp_password"]},
        ).status_code == 200

    def test_reset_password_by_token_then_login(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps

        app.dependency_overrides[auth_deps.require_admin] = _admin_override
        created = client.post(
            "/api/admin/users",
            json={"username": "bob", "role": "user"},
        ).json()
        reset_token = created["reset_token"]

        # Reset the password using the token.
        resp = client.post(
            "/api/auth/reset-password",
            json={"token": reset_token, "new_password": "bob-new-pw"},
        )
        assert resp.status_code == 200, resp.text

        # The new password works; the old temp password no longer does.
        assert client.post(
            "/api/auth/login", json={"username": "bob", "password": "bob-new-pw"}
        ).status_code == 200
        assert client.post(
            "/api/auth/login",
            json={"username": "bob", "password": created["temp_password"]},
        ).status_code == 401

    def test_reset_password_invalid_token_401(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        resp = client.post(
            "/api/auth/reset-password",
            json={"token": "not-a-real-token", "new_password": "whatever"},
        )
        assert resp.status_code == 401

    def test_admin_issue_reset_token_for_existing_user(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps
        from api.models import UserORM

        app.dependency_overrides[auth_deps.require_admin] = _admin_override
        created = client.post(
            "/api/admin/users", json={"username": "carol", "role": "user"}
        ).json()
        carol_id = created["user"]["id"]

        # Issue a fresh reset token.
        resp = client.post(f"/api/admin/users/{carol_id}/reset-token")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reset_token"]
        assert body["temp_password"]

        # The new temp password works; the previous one is invalidated.
        assert client.post(
            "/api/auth/login",
            json={"username": "carol", "password": body["temp_password"]},
        ).status_code == 200
        assert client.post(
            "/api/auth/login",
            json={"username": "carol", "password": created["temp_password"]},
        ).status_code == 401

        # And the fresh reset token resets the password.
        assert client.post(
            "/api/auth/reset-password",
            json={"token": body["reset_token"], "new_password": "carol-final"},
        ).status_code == 200
        assert client.post(
            "/api/auth/login", json={"username": "carol", "password": "carol-final"}
        ).status_code == 200

    def test_admin_create_duplicate_user_409(self):
        db_mod = _setup_db()
        app, client = _build_app(db_mod)
        from api.auth import deps as auth_deps

        app.dependency_overrides[auth_deps.require_admin] = _admin_override
        client.post("/api/admin/users", json={"username": "dup", "role": "user"})
        resp = client.post("/api/admin/users", json={"username": "dup", "role": "user"})
        assert resp.status_code == 409
