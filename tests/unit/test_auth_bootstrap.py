from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from api.auth.bootstrap import bootstrap_admin
from api.auth.local import hash_password, verify_password
from api.models import UserORM


class TestBootstrapNoneProvider:
    def test_skip_when_auth_none(self, monkeypatch):
        monkeypatch.setattr("api.auth.bootstrap.AUTH_PROVIDER", "none")
        assert bootstrap_admin() is False


class TestBootstrapNoEnv:
    def test_skip_when_env_unset(self, monkeypatch, isolated_db):
        monkeypatch.setattr("api.auth.bootstrap.AUTH_PROVIDER", "local")
        monkeypatch.delenv("BOOTSTRAP_ADMIN_USERNAME", raising=False)
        monkeypatch.delenv("BOOTSTRAP_ADMIN_PASSWORD", raising=False)
        assert bootstrap_admin() is False


class TestBootstrapCreateAdmin:
    def test_creates_admin_when_none_exists(self, monkeypatch, isolated_db):
        monkeypatch.setattr("api.auth.bootstrap.AUTH_PROVIDER", "local")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "bootadmin")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "bootpass123")
        result = bootstrap_admin()
        assert result is True
        with isolated_db.SessionLocal() as db:
            user = db.query(UserORM).filter(UserORM.username == "bootadmin").first()
            assert user is not None
            assert user.role == "admin"
            assert user.provider == "local"
            assert verify_password("bootpass123", user.password_hash)

    def test_creates_admin_with_keycloak_provider(self, monkeypatch, isolated_db):
        monkeypatch.setattr("api.auth.bootstrap.AUTH_PROVIDER", "keycloak")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "kcadmin")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "kcpass123")
        result = bootstrap_admin()
        assert result is True
        with isolated_db.SessionLocal() as db:
            user = db.query(UserORM).filter(UserORM.username == "kcadmin").first()
            assert user is not None
            assert user.role == "admin"


class TestBootstrapSkipExisting:
    def test_skip_when_admin_already_exists(self, monkeypatch, isolated_db):
        monkeypatch.setattr("api.auth.bootstrap.AUTH_PROVIDER", "local")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "bootadmin2")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "bootpass123")
        with isolated_db.SessionLocal() as db:
            db.add(UserORM(
                id="user_existing_admin",
                username="existingadmin",
                password_hash=hash_password("oldpass"),
                role="admin",
                provider="local",
            ))
            db.commit()
        assert bootstrap_admin() is False
        with isolated_db.SessionLocal() as db:
            # existing admin untouched
            user = db.query(UserORM).filter(UserORM.username == "existingadmin").first()
            assert verify_password("oldpass", user.password_hash)
            # new admin NOT created
            new = db.query(UserORM).filter(UserORM.username == "bootadmin2").first()
            assert new is None


class TestBootstrapPromoteExisting:
    def test_promotes_existing_user_to_admin(self, monkeypatch, isolated_db):
        monkeypatch.setattr("api.auth.bootstrap.AUTH_PROVIDER", "local")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "regularuser")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "newpass123")
        with isolated_db.SessionLocal() as db:
            db.add(UserORM(
                id="user_regular",
                username="regularuser",
                password_hash=hash_password("oldpass"),
                role="user",
                provider="local",
            ))
            db.commit()
        result = bootstrap_admin()
        assert result is True
        with isolated_db.SessionLocal() as db:
            user = db.query(UserORM).filter(UserORM.username == "regularuser").first()
            assert user.role == "admin"
            assert verify_password("newpass123", user.password_hash)
            assert not verify_password("oldpass", user.password_hash)


class TestBootstrapIdempotent:
    def test_second_call_is_noop(self, monkeypatch, isolated_db):
        monkeypatch.setattr("api.auth.bootstrap.AUTH_PROVIDER", "local")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "bootadmin3")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "bootpass123")
        assert bootstrap_admin() is True
        # Second call should find an admin already exists and return False
        assert bootstrap_admin() is False


class TestBootstrapNonFatal:
    def test_db_error_returns_false(self, monkeypatch, isolated_db):
        monkeypatch.setattr("api.auth.bootstrap.AUTH_PROVIDER", "local")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "erradmin")
        monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "errpass123")
        import api.db as db_mod

        original_session_local = db_mod.SessionLocal

        class _BadSession:
            def __enter__(self):
                raise RuntimeError("DB down")

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "SessionLocal", lambda: _BadSession())
        assert bootstrap_admin() is False
