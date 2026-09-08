"""Shared pytest fixtures for the Productarium test suite.

Provides:
- ``_isolated_env`` (autouse): every test gets an isolated SQLite DB + a stable
  ``SETTINGS_SECRET_KEY``, so no real Postgres is required and tests never touch
  the developer's data.
- ``isolated_db``: rebinds ``api.db`` (engine + ``SessionLocal`` + ``_db_ready``
  reset) to an in-memory ``StaticPool`` SQLite engine usable across the worker
  thread FastAPI's TestClient runs in, then runs ``init_db``. Returns the
  reloaded ``api.db`` module.
- ``mock_llm`` factory: returns an object whose ``generate(prompt)`` returns
  canned text (and ``stream`` yields ``ExpertStreamEvent`` content events).
- ``test_app`` / ``client``: build a FastAPI app + TestClient over the isolated
  DB with ``get_db`` overridden.
- ``admin_user`` / ``api_token_orm``: ORM rows for overriding the auth deps.

The duplicated per-module isolated-env fixtures in the existing test files are
kept (they predate this conftest); this module is the canonical source for new
tests and is safe because the autouse fixture is idempotent with them.
"""

from __future__ import annotations

import importlib
from datetime import datetime
from typing import Any, Iterator

import pytest


# --- Isolated environment (autouse) -----------------------------------------
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path_factory, monkeypatch):
    """Isolated SQLite DB + stable secret for every test.

    Uses ``tmp_path_factory`` (session-scoped temp dir) so DB files live under a
    per-test temp path without colliding. ``monkeypatch.setenv`` is
    automatically reverted by pytest.
    """
    tmp_path = tmp_path_factory.mktemp("iso")
    monkeypatch.setenv("DB_PROVIDER", "sqlite")
    monkeypatch.setenv("DB_HOST", str(tmp_path))
    monkeypatch.setenv("DB_NAME", str(tmp_path / "test.db"))
    monkeypatch.setenv("DB_USERNAME", "")
    monkeypatch.setenv("DB_PASSWORD", "")
    monkeypatch.setenv("AUTH_PROVIDER", "local")
    # Per-test managed state dir: anything the app writes outside the DB
    # (introspection disk cache, checkpointer fallback, …) stays inside the
    # test sandbox instead of the developer's ~/.productarium. Tests that
    # exercise the DEFAULT state-dir resolution delenv this explicitly.
    monkeypatch.setenv("PRODUCTARIUM_STATE_DIR", str(tmp_path / "state"))
    # Hermetic suite: the runtime checkpointer is Postgres-only and FATAL on
    # failure — tests opt into the in-memory saver instead (api/agents/runtime).
    monkeypatch.setenv("PRODUCTARIUM_CHECKPOINTER", "memory")
    # A stable-per-process Fernet key so encryption roundtrips are deterministic
    # within a test. cryptography is a hard dependency of the project.
    from cryptography.fernet import Fernet

    monkeypatch.setenv("SETTINGS_SECRET_KEY", Fernet.generate_key().decode())
    yield


# --- Isolated DB -------------------------------------------------------------
@pytest.fixture
def isolated_db():
    """Rebind ``api.db`` to an in-memory StaticPool SQLite engine + init schema.

    Returns the reloaded ``api.db`` module. The StaticPool + ``check_same_thread``
    config is required because FastAPI's TestClient serves requests in a worker
    thread that must share the in-memory DB connection.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import api.db as db

    importlib.reload(db)
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


@pytest.fixture
def session(isolated_db):
    """A short-lived Session from the isolated engine (committed data persists)."""
    s = isolated_db.SessionLocal()
    try:
        yield s
    finally:
        s.close()


# --- Mock LLM ----------------------------------------------------------------
class _MockLLM:
    """Minimal LLM stand-in: ``generate`` returns canned text, ``stream`` yields
    ``ExpertStreamEvent`` content events split into chunks."""

    def __init__(self, text: str = "mocked answer", chunk_size: int = 4):
        self.text = text
        self.chunk_size = chunk_size
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.text

    async def stream(self, prompt: str):
        from api.expert.types import EVENT_CONTENT, ExpertStreamEvent

        self.calls.append(prompt)
        text = self.text
        for i in range(0, len(text), self.chunk_size):
            yield ExpertStreamEvent(EVENT_CONTENT, text[i : i + self.chunk_size])


@pytest.fixture
def mock_llm():
    """Factory returning a fresh ``_MockLLM`` with the given canned text."""
    def _factory(text: str = "mocked answer", chunk_size: int = 4) -> _MockLLM:
        return _MockLLM(text=text, chunk_size=chunk_size)

    return _factory


# --- App + client ------------------------------------------------------------
@pytest.fixture
def admin_user():
    """A fixed admin UserORM for overriding ``require_admin``."""
    from api.models import UserORM

    return UserORM(
        id="user_admin1",
        username="admin",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def api_token_orm():
    """A fixed ApiTokenORM for overriding ``require_api_token`` (verify-bypass)."""
    from api.models import ApiTokenORM

    return ApiTokenORM(
        id="tok_fixed",
        user_id="user_admin1",
        token_hash="x" * 64,
        name="fixed",
        created_at=datetime.utcnow(),
    )


def build_test_client(
    db_mod, routers, *, auth_none: bool = True, default_admin_auth: bool = True
) -> tuple[Any, Any]:
    """Build a FastAPI app + TestClient over an isolated DB.

    Args:
        db_mod: the rebound ``api.db`` module (from ``isolated_db``).
        routers: iterable of router modules each exposing ``router``.
        auth_none: when True, ``AUTH_PROVIDER`` is left at the autouse default
            (``local``); callers needing unauthenticated access set it to
            ``none`` via their own ``monkeypatch`` on ``api.auth.deps``.
        default_admin_auth: when True, ``get_current_user`` is overridden with
            a fixed admin for every included router module (needed since the
            products/docgen/databases routers enforce router-level auth).
            Tests exercising REAL auth semantics (real cookies, 401 paths)
            pass False.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    for mod in routers:
        app.include_router(mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    # Override every captured get_db reference the routers hold.
    seen = set()
    for mod in routers:
        get_db = getattr(mod, "get_db", None)
        if get_db is not None and id(get_db) not in seen:
            app.dependency_overrides[get_db] = _get_test_db
            seen.add(id(get_db))

    # Default auth: a fixed admin user for every router module that uses
    # get_current_user, so router-level auth (products/docgen/databases)
    # doesn't 401 tests exercising CRUD/masking logic rather than authz.
    # Tests needing a specific (e.g. non-admin) user override the same key
    # AFTER building the client (last write wins); tests that monkeypatch
    # AUTH_PROVIDER="none" are unaffected; tests exercising real auth
    # semantics opt out via default_admin_auth=False.
    if default_admin_auth:
        from api.models import UserORM

        _default_admin = UserORM(
            id="user_admin1",
            username="admin",
            role="admin",
            provider="local",
            created_at=datetime.utcnow(),
        )
        for mod in routers:
            gcu = getattr(mod, "get_current_user", None)
            if gcu is not None and gcu not in app.dependency_overrides:
                app.dependency_overrides[gcu] = lambda: _default_admin
    return app, TestClient(app)
