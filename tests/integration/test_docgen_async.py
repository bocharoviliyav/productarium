#!/usr/bin/env python3
"""Tests for the async 202 + poll artifact documentation generation endpoint.

Covers:
- POST .../generate returns 202 + job_id immediately (does NOT block on the
  heavy pipeline), then the job runs in a worker thread with its own event loop
  and DB session and persists ``generated_docs`` onto the artifact.
- GET .../generate/status?job_id=... reports queued -> running -> succeeded.
- 404s for a missing artifact (POST) and an unknown job_id (status).

The heavy ``generate_artifact_documentation`` pipeline is monkeypatched with an
instant async fake so the worker thread completes in milliseconds. The worker
thread reads the module-global ``api.api.SessionLocal`` at runtime, so the test
rebinds it to the isolated StaticPool SQLite engine (one shared connection,
``check_same_thread=False``) — the same trick the other test modules use so the
worker's own session sees the seeded data and persists into the test DB.
"""

from __future__ import annotations

import importlib
import time

import pytest
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
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    yield


@pytest.fixture(autouse=True)
def _clear_app_overrides():
    """The main ``api.api.app`` is a module-level singleton; clear any
    dependency overrides leaked from a previous test after each test runs."""
    yield
    try:
        import api.api as api_mod
        api_mod.app.dependency_overrides.clear()
    except Exception:
        pass


def _setup_db():
    """Rebind api.db to an isolated StaticPool in-memory SQLite engine."""
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


def _build_app(db_mod, monkeypatch):
    """Return the real api.api app + client with get_db overridden and the
    worker-thread SessionLocal rebound to the test engine."""
    import api.api as api_mod
    # The worker thread reads this module global at runtime (db = SessionLocal()).
    monkeypatch.setattr(api_mod, "SessionLocal", db_mod.SessionLocal, raising=True)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    api_mod.app.dependency_overrides[api_mod.get_db] = _get_test_db
    return api_mod.app, TestClient(api_mod.app)


def _seed(db_mod):
    from api.models import ArtifactORM, ProductORM
    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id="prod_1", name="Acme"))
        db.flush()
        db.add(ArtifactORM(
            id="art_1", product_id="prod_1", name="svc", type="links",
            content='[{"url":"https://x","title":"X"}]', source="manual",
        ))
        db.commit()


class TestAsyncDocgen:
    def test_generate_returns_202_then_succeeded(self, monkeypatch):
        db_mod = _setup_db()
        _seed(db_mod)
        _app, client = _build_app(db_mod, monkeypatch)

        # Instant async fake for the heavy pipeline; it persists docs onto the
        # artifact ORM (loaded in the worker's own session) and returns.
        import api.artifact_docgen as adg

        async def _fake(artifact, product, **kwargs):
            artifact.generated_docs = "# Generated\n\nfake"
            artifact.pages = {
                "page_links": {
                    "id": "page_links", "title": "Links",
                    "content": "# Generated", "filePaths": [],
                    "importance": "medium", "relatedPages": [],
                }
            }
            return artifact.generated_docs

        monkeypatch.setattr(adg, "generate_artifact_documentation", _fake)

        resp = client.post(
            "/api/products/prod_1/artifacts/art_1/generate",
            json={"language": "en"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        job_id = body["job_id"]
        assert job_id

        # Poll until the worker reaches a terminal state (instant fake).
        deadline = time.time() + 10
        last = None
        while time.time() < deadline:
            s = client.get(
                "/api/products/prod_1/artifacts/art_1/generate/status",
                params={"job_id": job_id},
            )
            assert s.status_code == 200
            last = s.json()
            if last["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert last is not None, "status never reached a terminal state"
        assert last["status"] == "succeeded", last
        assert last["docs_chars"] == len("# Generated\n\nfake")

        # The generated docs were committed to the artifact in the shared DB.
        from api.models import ArtifactORM
        with db_mod.SessionLocal() as db:
            art = db.get(ArtifactORM, "art_1")
            assert art is not None
            assert art.generated_docs == "# Generated\n\nfake"
            assert art.pages is not None and "page_links" in art.pages

    def test_generate_404_missing_artifact(self, monkeypatch):
        db_mod = _setup_db()
        from api.models import ProductORM
        with db_mod.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme"))
            db.commit()
        _app, client = _build_app(db_mod, monkeypatch)
        resp = client.post(
            "/api/products/prod_1/artifacts/ghost/generate",
            json={"language": "en"},
        )
        assert resp.status_code == 404

    def test_status_404_unknown_job(self, monkeypatch):
        db_mod = _setup_db()
        _seed(db_mod)
        _app, client = _build_app(db_mod, monkeypatch)
        resp = client.get(
            "/api/products/prod_1/artifacts/art_1/generate/status",
            params={"job_id": "does-not-exist"},
        )
        assert resp.status_code == 404

    def test_status_404_job_of_other_artifact(self, monkeypatch):
        """A real job_id but queried against a different artifact must 404
        (prevents cross-artifact status reads)."""
        db_mod = _setup_db()
        _seed(db_mod)
        _app, client = _build_app(db_mod, monkeypatch)

        import api.artifact_docgen as adg

        async def _fake(artifact, product, **kwargs):
            artifact.generated_docs = "ok"
            return artifact.generated_docs

        monkeypatch.setattr(adg, "generate_artifact_documentation", _fake)
        resp = client.post(
            "/api/products/prod_1/artifacts/art_1/generate",
            json={"language": "en"},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Same product, WRONG artifact -> 404.
        bad = client.get(
            "/api/products/prod_1/artifacts/art_other/generate/status",
            params={"job_id": job_id},
        )
        assert bad.status_code == 404
