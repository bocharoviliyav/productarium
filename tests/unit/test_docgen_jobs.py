"""Unit tests for api.docgen.jobs (async 202+poll job registry).

Covers: create_job, get_job, submit_job, _docgen_prune_old_jobs,
_run_docgen_job_async (codebase dispatch, spec dispatch, error path,
unsupported entity_type, product not found), _resolve_indexing_drain_seconds,
_run_docgen_job (worker thread entry point).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.jobs as jobs_mod


# ============================================================================
# create_job / get_job
# ============================================================================
class TestCreateGetJob:
    def test_create_job_returns_id(self):
        job_id = jobs_mod.create_job("prod_1", "codebase", "cb_1")
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_created_job_has_queued_status(self):
        job_id = jobs_mod.create_job("prod_1", "codebase", "cb_1")
        job = jobs_mod.get_job(job_id)
        assert job is not None
        assert job["status"] == "queued"
        assert job["product_id"] == "prod_1"
        assert job["entity_type"] == "codebase"
        assert job["entity_id"] == "cb_1"
        assert job["created_at"] is not None
        assert job["started_at"] is None
        assert job["finished_at"] is None
        assert job["error"] is None
        assert job["docs_chars"] is None

    def test_get_unknown_job_returns_none(self):
        assert jobs_mod.get_job("nonexistent_job_id") is None

    def test_create_multiple_jobs_unique_ids(self):
        id1 = jobs_mod.create_job("prod_1", "codebase", "cb_1")
        id2 = jobs_mod.create_job("prod_1", "codebase", "cb_2")
        assert id1 != id2


# ============================================================================
# _docgen_prune_old_jobs
# ============================================================================
class TestPruneOldJobs:
    def test_prunes_finished_old_jobs(self):
        # Create a job and mark it as finished in the past
        job_id = jobs_mod.create_job("prod_1", "codebase", "cb_1")
        jobs_mod._docgen_jobs[job_id]["finished_at"] = time.time() - 7200  # 2h ago
        jobs_mod._docgen_jobs[job_id]["status"] = "succeeded"

        jobs_mod._docgen_prune_old_jobs(max_age_seconds=3600)
        assert jobs_mod.get_job(job_id) is None

    def test_keeps_recent_finished_jobs(self):
        job_id = jobs_mod.create_job("prod_1", "codebase", "cb_1")
        jobs_mod._docgen_jobs[job_id]["finished_at"] = time.time() - 100  # 100s ago
        jobs_mod._docgen_jobs[job_id]["status"] = "succeeded"

        jobs_mod._docgen_prune_old_jobs(max_age_seconds=3600)
        assert jobs_mod.get_job(job_id) is not None

    def test_keeps_unfinished_jobs(self):
        job_id = jobs_mod.create_job("prod_1", "codebase", "cb_1")
        # finished_at is None (queued/running)
        jobs_mod._docgen_prune_old_jobs(max_age_seconds=1)
        assert jobs_mod.get_job(job_id) is not None


# ============================================================================
# _resolve_indexing_drain_seconds
# ============================================================================
class TestResolveIndexingDrainSeconds:
    def test_returns_positive_float(self):
        result = jobs_mod._resolve_indexing_drain_seconds()
        assert isinstance(result, float)
        assert result > 0


# ============================================================================
# _run_docgen_job_async — codebase dispatch
# ============================================================================
class TestRunDocgenJobAsyncCodebase:
    def test_codebase_success(self, isolated_db, monkeypatch):
        """Codebase entity_type dispatches to generate_codebase_docs."""
        from api.models import ProductORM, CodebaseORM

        # Create product + codebase in the isolated DB
        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_job_cb", name="TestProduct")
            db.add(p)
            db.commit()
            cb = CodebaseORM(
                id="cb_job_1", product_id="prod_job_cb", name="testrepo",
                repo_url="https://github.com/o/testrepo",
            )
            db.add(cb)
            db.commit()
        finally:
            db.close()

        # Patch jobs.SessionLocal to use the isolated DB
        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        # Mock generate_codebase_docs
        async def fake_generate(artifact, product, model=None, language="ru"):
            return "Generated docs content"
        import api.docgen.codebase as codebase_mod
        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        job_id = jobs_mod.create_job("prod_job_cb", "codebase", "cb_job_1")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_cb", "codebase", "cb_job_1", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["docs_chars"] == len("Generated docs content")
        assert job["finished_at"] is not None
        assert job["error"] is None

    def test_codebase_not_found_fails(self, isolated_db, monkeypatch):
        from api.models import ProductORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_job_nf", name="TestProduct")
            db.add(p)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        job_id = jobs_mod.create_job("prod_job_nf", "codebase", "nonexistent_cb")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_nf", "codebase", "nonexistent_cb", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "failed"
        assert "Codebase not found" in job["error"]
        assert job["finished_at"] is not None


# ============================================================================
# _run_docgen_job_async — spec dispatch
# ============================================================================
class TestRunDocgenJobAsyncSpec:
    def test_spec_openapi_success(self, isolated_db, monkeypatch):
        from api.models import ProductORM, SpecORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_job_spec", name="TestProduct")
            db.add(p)
            db.commit()
            spec = SpecORM(
                id="spec_job_1", product_id="prod_job_spec", name="api",
                kind="openapi", content='{"openapi":"3.0.0"}',
            )
            db.add(spec)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        async def fake_generate(spec, product, model=None, language="ru"):
            return "Spec docs"
        import api.docgen.spec as spec_mod
        monkeypatch.setattr(spec_mod, "generate_openapi_docs", fake_generate)

        job_id = jobs_mod.create_job("prod_job_spec", "spec", "spec_job_1")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_spec", "spec", "spec_job_1", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["docs_chars"] == len("Spec docs")

    def test_spec_asyncapi_dispatch(self, isolated_db, monkeypatch):
        from api.models import ProductORM, SpecORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_job_async", name="TestProduct")
            db.add(p)
            db.commit()
            spec = SpecORM(
                id="spec_async_1", product_id="prod_job_async", name="asyncapi",
                kind="asyncapi", content='{"asyncapi":"2.6.0"}',
            )
            db.add(spec)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        # jobs.py reads spec_kind attr (not kind); add a property that
        # exposes kind as spec_kind so the asyncapi branch is exercised
        monkeypatch.setattr(SpecORM, "spec_kind", property(lambda self: self.kind), raising=False)

        async def fake_async_generate(spec, product, model=None, language="ru"):
            return "AsyncAPI docs"
        import api.docgen.spec as spec_mod
        monkeypatch.setattr(spec_mod, "generate_asyncapi_docs", fake_async_generate)

        # Also patch openapi to ensure it's NOT called
        openapi_called = {"v": False}
        async def fake_openapi_generate(spec, product, model=None, language="ru"):
            openapi_called["v"] = True
            return "OpenAPI docs"
        monkeypatch.setattr(spec_mod, "generate_openapi_docs", fake_openapi_generate)

        job_id = jobs_mod.create_job("prod_job_async", "spec", "spec_async_1")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_async", "spec", "spec_async_1", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["docs_chars"] == len("AsyncAPI docs")
        assert openapi_called["v"] is False

    def test_spec_not_found_fails(self, isolated_db, monkeypatch):
        from api.models import ProductORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_job_spec_nf", name="TestProduct")
            db.add(p)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        job_id = jobs_mod.create_job("prod_job_spec_nf", "spec", "nonexistent_spec")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_spec_nf", "spec", "nonexistent_spec", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "failed"
        assert "Spec not found" in job["error"]


# ============================================================================
# _run_docgen_job_async — error paths
# ============================================================================
class TestRunDocgenJobAsyncErrors:
    def test_product_not_found_fails(self, isolated_db, monkeypatch):
        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        job_id = jobs_mod.create_job("nonexistent_prod", "codebase", "cb_1")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "nonexistent_prod", "codebase", "cb_1", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "failed"
        assert "Product not found" in job["error"]

    def test_unsupported_entity_type_fails(self, isolated_db, monkeypatch):
        from api.models import ProductORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_job_unsup", name="TestProduct")
            db.add(p)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        job_id = jobs_mod.create_job("prod_job_unsup", "links", "link_1")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_unsup", "links", "link_1", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "failed"
        assert "Unsupported" in job["error"]
        assert "links" in job["error"]

    def test_generator_raises_marks_failed(self, isolated_db, monkeypatch):
        from api.models import ProductORM, CodebaseORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_job_err", name="TestProduct")
            db.add(p)
            db.commit()
            cb = CodebaseORM(
                id="cb_job_err", product_id="prod_job_err", name="repo",
                repo_url="https://github.com/o/repo",
            )
            db.add(cb)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        async def boom_generate(artifact, product, model=None, language="ru"):
            raise RuntimeError("Generator exploded")
        import api.docgen.codebase as codebase_mod
        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", boom_generate)

        job_id = jobs_mod.create_job("prod_job_err", "codebase", "cb_job_err")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_err", "codebase", "cb_job_err", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "failed"
        assert "Generator exploded" in job["error"]
        assert job["finished_at"] is not None


# ============================================================================
# _run_docgen_job (worker thread entry point)
# ============================================================================
class TestRunDocgenJob:
    def test_runs_in_new_loop_and_closes(self, isolated_db, monkeypatch):
        from api.models import ProductORM, CodebaseORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_thread", name="TestProduct")
            db.add(p)
            db.commit()
            cb = CodebaseORM(
                id="cb_thread", product_id="prod_thread", name="repo",
                repo_url="https://github.com/o/repo",
            )
            db.add(cb)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        async def fake_generate(artifact, product, model=None, language="ru"):
            return "Thread docs"
        import api.docgen.codebase as codebase_mod
        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        job_id = jobs_mod.create_job("prod_thread", "codebase", "cb_thread")
        # This runs synchronously in a new event loop on the current thread
        jobs_mod._run_docgen_job(
            job_id, "prod_thread", "codebase", "cb_thread", None, "ru"
        )

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["docs_chars"] == len("Thread docs")

    def test_drain_timeout_is_non_fatal(self, isolated_db, monkeypatch):
        from api.models import ProductORM, CodebaseORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_drain", name="TestProduct")
            db.add(p)
            db.commit()
            cb = CodebaseORM(
                id="cb_drain", product_id="prod_drain", name="repo",
                repo_url="https://github.com/o/repo",
            )
            db.add(cb)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        async def fake_generate(artifact, product, model=None, language="ru"):
            return "Drain docs"
        import api.docgen.codebase as codebase_mod
        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        # Force a very short drain timeout
        monkeypatch.setattr(jobs_mod, "_resolve_indexing_drain_seconds", lambda: 0.001)

        job_id = jobs_mod.create_job("prod_drain", "codebase", "cb_drain")
        # Should not raise even if drain times out
        jobs_mod._run_docgen_job(
            job_id, "prod_drain", "codebase", "cb_drain", None, "ru"
        )

        job = jobs_mod.get_job(job_id)
        # Job should still succeed (drain timeout is non-fatal)
        assert job["status"] == "succeeded"


# ============================================================================
# submit_job
# ============================================================================
class TestSubmitJob:
    def test_submit_and_complete(self, isolated_db, monkeypatch):
        from api.models import ProductORM, CodebaseORM

        db = isolated_db.SessionLocal()
        try:
            p = ProductORM(id="prod_submit", name="TestProduct")
            db.add(p)
            db.commit()
            cb = CodebaseORM(
                id="cb_submit", product_id="prod_submit", name="repo",
                repo_url="https://github.com/o/repo",
            )
            db.add(cb)
            db.commit()
        finally:
            db.close()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        async def fake_generate(artifact, product, model=None, language="ru"):
            return "Submit docs"
        import api.docgen.codebase as codebase_mod
        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        job_id = jobs_mod.create_job("prod_submit", "codebase", "cb_submit")
        jobs_mod.submit_job(
            job_id, "prod_submit", "codebase", "cb_submit", None, "ru"
        )

        # Wait for the worker thread to finish (poll with timeout)
        deadline = time.time() + 10
        while time.time() < deadline:
            job = jobs_mod.get_job(job_id)
            if job and job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["docs_chars"] == len("Submit docs")


# ============================================================================
# Cleanup fixture: clear job registry between tests
# ============================================================================
@pytest.fixture(autouse=True)
def _clear_jobs():
    """Clear the module-level job registry before each test."""
    jobs_mod._docgen_jobs.clear()
    yield
    jobs_mod._docgen_jobs.clear()
