#!/usr/bin/env python3
"""Tests for the async 202 + poll artifact documentation generation endpoint.

Covers:
- POST .../generate returns 202 + job_id immediately (does NOT block on the
  heavy pipeline), then the job runs in a worker thread with its own event loop
  and DB session and persists ``generated_docs`` onto the artifact.
- GET .../generate/status?job_id=... reports queued -> running -> succeeded.
- 404s for a missing artifact (POST) and an unknown job_id (status).

The heavy ``generate_codebase_docs`` pipeline is monkeypatched with an
instant async fake so the worker thread completes in milliseconds. The worker
thread reads ``api.docgen.jobs.SessionLocal`` at runtime (its import site), so
the test rebinds it to the isolated StaticPool SQLite engine (one shared
connection, ``check_same_thread=False``) — the same trick the other test
modules use so the worker's own session sees the seeded data and persists into
the test DB.
"""

from __future__ import annotations

import importlib
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# The autouse ``_isolated_env`` fixture from ``tests/conftest.py`` provides the
# isolated SQLite DB + stable SETTINGS_SECRET_KEY + cognee/Ollama stubs for every
# test in this module. No per-module duplicate is needed here.


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
    import api.docgen.jobs as dj
    # The worker thread reads SessionLocal from api.docgen.jobs (its import
    # site) at runtime (db = SessionLocal()).
    monkeypatch.setattr(dj, "SessionLocal", db_mod.SessionLocal, raising=True)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    api_mod.app.dependency_overrides[api_mod.get_db] = _get_test_db
    return api_mod.app, TestClient(api_mod.app)


def _seed(db_mod):
    from api.models import CodebaseORM, ProductORM
    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id="prod_1", name="Acme"))
        db.flush()
        db.add(CodebaseORM(
            id="art_1", product_id="prod_1", name="svc",
            repo_url="https://github.com/x/y", repo_type="github",
            source="manual",
        ))
        db.commit()


class TestAsyncDocgen:
    def test_generate_returns_202_then_succeeded(self, monkeypatch):
        db_mod = _setup_db()
        _seed(db_mod)
        _app, client = _build_app(db_mod, monkeypatch)

        # Instant async fake for the heavy pipeline; it persists docs onto the
        # codebase ORM (loaded in the worker's own session) and returns.
        import api.docgen.codebase as adg

        async def _fake(artifact, product, **kwargs):
            artifact.generated_docs = "# Generated\n\nfake"
            artifact.pages = {
                "page_overview": {
                    "id": "page_overview", "title": "Overview",
                    "content": "# Generated", "filePaths": [],
                    "importance": "medium", "relatedPages": [],
                }
            }
            return artifact.generated_docs

        monkeypatch.setattr(adg, "generate_codebase_docs", _fake)

        resp = client.post(
            "/api/products/prod_1/codebases/art_1/generate",
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
                "/api/products/prod_1/codebases/art_1/generate/status",
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

        # The generated docs were committed to the codebase in the shared DB.
        from api.models import CodebaseORM
        with db_mod.SessionLocal() as db:
            art = db.get(CodebaseORM, "art_1")
            assert art is not None
            assert art.generated_docs == "# Generated\n\nfake"
            assert art.pages is not None and "page_overview" in art.pages

    def test_generate_404_missing_artifact(self, monkeypatch):
        db_mod = _setup_db()
        from api.models import ProductORM
        with db_mod.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme"))
            db.commit()
        _app, client = _build_app(db_mod, monkeypatch)
        resp = client.post(
            "/api/products/prod_1/codebases/ghost/generate",
            json={"language": "en"},
        )
        assert resp.status_code == 404

    def test_status_404_unknown_job(self, monkeypatch):
        db_mod = _setup_db()
        _seed(db_mod)
        _app, client = _build_app(db_mod, monkeypatch)
        resp = client.get(
            "/api/products/prod_1/codebases/art_1/generate/status",
            params={"job_id": "does-not-exist"},
        )
        assert resp.status_code == 404

    def test_status_404_job_of_other_artifact(self, monkeypatch):
        """A real job_id but queried against a different codebase must 404
        (prevents cross-codebase status reads)."""
        db_mod = _setup_db()
        _seed(db_mod)
        _app, client = _build_app(db_mod, monkeypatch)

        import api.docgen.codebase as adg

        async def _fake(artifact, product, **kwargs):
            artifact.generated_docs = "ok"
            return artifact.generated_docs

        monkeypatch.setattr(adg, "generate_codebase_docs", _fake)
        resp = client.post(
            "/api/products/prod_1/codebases/art_1/generate",
            json={"language": "en"},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Same product, WRONG codebase -> 404.
        bad = client.get(
            "/api/products/prod_1/codebases/art_other/generate/status",
            params={"job_id": job_id},
        )
        assert bad.status_code == 404

    def test_job_succeeds_without_waiting_on_indexing(self, monkeypatch):
        """Display is decoupled from the cognee knowledge graph: a job whose
        pipeline succeeds must reach status="succeeded" with committed docs,
        even while cognee indexing is still running (a long cognify). The
        indexing coroutine is handed off to the main event loop and never
        gates the docgen job."""
        import asyncio
        import api.api as api_mod
        import api.docgen as adg_pkg
        import api.docgen.codebase as adg
        import api.cognee as cm
        import api.docgen.jobs as dj
        from api.models import CodebaseORM

        _indexed = {"v": False}

        async def _long_index(content_or_path, dataset_name=None):
            # Simulates a 20-30 min cognify; runs on the MAIN loop, NOT the
            # worker loop, so it never gates the docgen job and is cleaned up
            # when the TestClient context exits.
            _indexed["v"] = True
            await asyncio.sleep(300)

        monkeypatch.setattr(cm, "add_and_index_document", _long_index)

        async def _fake(artifact, product, **kwargs):
            artifact.generated_docs = "# Generated\n\ncontent"
            artifact.pages = {
                "page_overview": {
                    "id": "page_overview", "title": "Overview",
                    "content": "# Generated", "filePaths": [],
                    "importance": "medium", "relatedPages": [],
                }
            }
            # Call _index_in_background like the real pipeline does. With the
            # main loop captured (via `with TestClient`), it hands off to the
            # main loop; the worker drain finds no pending tasks and returns
            # immediately — the job does NOT wait for the 300s cognify.
            adg_pkg._index_in_background("repo_dir", "prod_prod_1")
            return artifact.generated_docs

        monkeypatch.setattr(adg, "generate_codebase_docs", _fake)

        db_mod = _setup_db()
        _seed(db_mod)
        # `with TestClient` triggers the lifespan startup event, which calls
        # set_main_event_loop — so _index_in_background hands off to the main
        # loop instead of scheduling on the worker loop (production behavior).
        client = TestClient(api_mod.app)
        # Override get_db so the request-scoped session uses the test DB.
        def _get_test_db():
            s = db_mod.SessionLocal()
            try:
                yield s
            finally:
                s.close()
        api_mod.app.dependency_overrides[api_mod.get_db] = _get_test_db
        # The worker thread reads SessionLocal from api.docgen.jobs (its
        # import site), not api.api.
        monkeypatch.setattr(dj, "SessionLocal", db_mod.SessionLocal, raising=True)

        with client:
            resp = client.post(
                "/api/products/prod_1/codebases/art_1/generate",
                json={"language": "en"},
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            # Poll until terminal: should be "succeeded" quickly (the worker
            # drain finds no pending tasks — indexing was handed off).
            deadline = time.time() + 15
            last = None
            while time.time() < deadline:
                s = client.get(
                    "/api/products/prod_1/codebases/art_1/generate/status",
                    params={"job_id": job_id},
                )
                assert s.status_code == 200
                last = s.json()
                if last["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.1)
            assert last is not None, "status never reached a terminal state"
            assert last["status"] == "succeeded", last
            assert last["indexing_status"] == "succeeded", last
            assert _indexed["v"], "background indexing was not scheduled"

        # Docs were committed despite indexing still running in the background.
        with db_mod.SessionLocal() as db:
            art = db.get(CodebaseORM, "art_1")
            assert art is not None
            assert (art.generated_docs or "").startswith("# Generated")

    def test_all_placeholder_sections_marks_job_failed(self, monkeypatch):
        """When generation produces no usable content for ANY section, the job
        must be marked failed (with no committed docs) instead of committing
        the "Содержимое раздела временно недоступно" placeholder as success."""
        db_mod = _setup_db()
        _seed(db_mod)
        _app, client = _build_app(db_mod, monkeypatch)

        import api.docgen.codebase as adg

        async def _fake(artifact, product, **kwargs):
            # Simulate a total generation failure: every section is the
            # placeholder string. generate_codebase_docs raises before persist.
            raise ValueError(
                "Не удалось сгенерировать ни один раздел документации (LLM/RLM "
                "недоступны или превысили таймаут). Проверьте подключение к модели "
                "и перезапустите генерацию."
            )

        monkeypatch.setattr(adg, "generate_codebase_docs", _fake)
        resp = client.post(
            "/api/products/prod_1/codebases/art_1/generate",
            json={"language": "en"},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        deadline = time.time() + 10
        last = None
        while time.time() < deadline:
            s = client.get(
                "/api/products/prod_1/codebases/art_1/generate/status",
                params={"job_id": job_id},
            )
            assert s.status_code == 200
            last = s.json()
            if last["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert last is not None, "status never reached a terminal state"
        assert last["status"] == "failed", last
        assert "Не удалось сгенерировать" in (last.get("error") or "")

        # No placeholder-only docs were committed as a success.
        from api.models import CodebaseORM
        with db_mod.SessionLocal() as db:
            art = db.get(CodebaseORM, "art_1")
            assert art is not None
            assert not (art.generated_docs or "").strip()
