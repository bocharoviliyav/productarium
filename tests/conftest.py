"""Shared pytest fixtures for the Productarium test suite.

Provides:
- ``_isolated_env`` (autouse): every test gets an isolated SQLite DB + a stable
  ``SETTINGS_SECRET_KEY`` + cognee connection-test skip, so no real Postgres /
  cognee / Ollama is required and tests never touch the developer's data.
- ``isolated_db``: rebinds ``api.db`` (engine + ``SessionLocal`` + ``_db_ready``
  reset) to an in-memory ``StaticPool`` SQLite engine usable across the worker
  thread FastAPI's TestClient runs in, then runs ``init_db``. Returns the
  reloaded ``api.db`` module.
- ``fake_cognee``: injects a stub ``cognee`` package into ``sys.modules`` so the
  cognee happy paths in ``api/cognee/*`` execute without the real dependency
  installed (it is optional locally).
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
import os
import sys
import types
from datetime import datetime
from typing import Any, Iterator

import pytest


# --- Isolated environment (autouse) -----------------------------------------
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path_factory, monkeypatch):
    """Isolated SQLite DB + stable secret + cognee/Ollama stubs for every test.

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
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
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


# --- Fake cognee -------------------------------------------------------------
class _FakeCogneeResult:
    """Minimal async-callable stand-in for cognee's async API."""

    def __init__(self, data: Any = None):
        self.data = data


@pytest.fixture
def fake_cognee(monkeypatch):
    """Inject a stub ``cognee`` package into ``sys.modules``.

    The real ``cognee`` dependency is optional locally. Several modules under
    ``api/cognee/`` import ``cognee`` lazily inside functions; this fixture
    installs an in-memory fake so those import paths execute. The fake exposes
    ``add``, ``cognify``, ``search`` (async) + ``DataPipeline`` + config helpers
    used by ``api/cognee/_runtime.py``.

    Returns the fake module so a test can assert on recorded calls.
    """
    calls: dict[str, list[Any]] = {"add": [], "cognify": [], "search": []}

    fake = types.ModuleType("cognee")

    async def _add(data):
        calls["add"].append(data)
        return ["fake_node"]

    async def _cognify():
        calls["cognify"].append(None)
        return ["fake_cognify"]

    async def _search(query, query_type=None, **kwargs):
        calls["search"].append(query)
        return ["fake search result"]

    fake.add = _add
    fake.cognify = _cognify
    fake.search = _search

    # ``cognee.modules.data.extraction`` etc. are accessed by _runtime config.
    # Provide a minimal getattr-fallback module so arbitrary attribute access
    # returns a stub rather than ImportError.
    class _AttrModule(types.ModuleType):
        def __getattr__(self, name):
            return types.ModuleType(f"cognee.{name}")

    fake.modules = _AttrModule("cognee.modules")
    fake.modules.data = _AttrModule("cognee.modules.data")
    fake.modules.data.extraction = _AttrModule("cognee.modules.data.extraction")
    fake.modules.pipelines = _AttrModule("cognee.modules.pipelines")

    # Config helpers used by _runtime.
    class _ConfigStub:
        def __init__(self):
            self._d: dict[str, Any] = {}

        def get(self, *keys):
            return None

        def set(self, key, value):
            self._d[key] = value

        def get_existing_config_without_default(self, *keys):
            return {}

    fake.get_config = _ConfigStub().get
    fake.set_config = _ConfigStub().set
    fake.get_existing_config_without_default = _ConfigStub().get_existing_config_without_default

    monkeypatch.setitem(sys.modules, "cognee", fake)
    # Pre-register submodules some code imports directly.
    for sub in (
        "cognee.modules.data",
        "cognee.modules.data.extraction",
        "cognee.modules.pipelines",
        "cognee.api",
        "cognee.api.v1",
    ):
        monkeypatch.setitem(sys.modules, sub, _AttrModule(sub))
    fake.calls = calls
    return fake


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


def build_test_client(db_mod, routers, *, auth_none: bool = True) -> tuple[Any, Any]:
    """Build a FastAPI app + TestClient over an isolated DB.

    Args:
        db_mod: the rebound ``api.db`` module (from ``isolated_db``).
        routers: iterable of router modules each exposing ``router``.
        auth_none: when True, ``AUTH_PROVIDER`` is left at the autouse default
            (``local``); callers needing unauthenticated access set it to
            ``none`` via their own ``monkeypatch`` on ``api.auth.deps``.
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
    return app, TestClient(app)
