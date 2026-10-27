"""Unit tests for SPEC-010 incremental plumbing.

Covers:
- ``_run_docgen_job_async`` incremental branch: unchanged/unverifiable →
  success WITHOUT a version; changed → force_pages = impacted page ids
  (empty impact → full regeneration), baseline + generate versions appended.
- ``_run_detect_job_async``: success persists ``source_state.checked_at``
  and the result summary; ValueError → failed with the message surfaced.
- ``_index_pages_in_background``: page-granular re-indexing with the
  legacy-blob bootstrap (first run indexes ALL pages) and ``only`` slicing
  once page sources exist.
- ``PgVectorMemoryBackend.list_page_sources`` / ``prune_stale_sources``.
- Detect / detect-status / incremental endpoints (202 + poll contract).
- ``source_state`` never leaks into client payloads.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import api.docgen.jobs as jobs_mod
from api.memory.pgvector_backend import PgVectorMemoryBackend
from tests.conftest import build_test_client

PROD = "prod_inc"
CB = "cb_inc"
BASELINE = "a" * 40


# Rate-limit buckets are process-global (P1-17): earlier test files POSTing
# generate/detect as the same fixed admin would otherwise exhaust the hourly
# bucket and this file's endpoint tests would 429 depending on run order.
@pytest.fixture(autouse=True)
def _clear_jobs():
    from api.utils.rate_limit import reset_rate_limits

    jobs_mod._docgen_jobs.clear()
    reset_rate_limits()
    yield
    jobs_mod._docgen_jobs.clear()
    reset_rate_limits()


def _seed_codebase(db_mod) -> None:
    """Product + codebase with stored (versioned) docs and a source_state."""
    from api.models import CodebaseORM, ProductORM

    db = db_mod.SessionLocal()
    try:
        db.add(ProductORM(id=PROD, name="IncProduct"))
        db.add(CodebaseORM(
            id=CB, product_id=PROD, name="repo",
            repo_url="https://github.com/o/repo",
            generated_docs="# docs",
            pages={
                "page_a": {"title": "A", "content": "AAA"},
                "page_b": {"title": "B", "content": "BBB"},
            },
            source_state={"kind": "codebase", "baseline": BASELINE},
        ))
        db.commit()
    finally:
        db.close()


def _run_incremental(db_mod, monkeypatch, detect_result, gen_calls):
    """Run ``_run_docgen_job_async(..., incremental=True)`` over fakes."""
    import api.docgen.codebase as codebase_mod
    import api.docgen.incremental as inc

    monkeypatch.setattr(jobs_mod, "SessionLocal", db_mod.SessionLocal)

    async def fake_detect(entity, p_orm, entity_type, model=None):
        if detect_result.get("changed") is not True:
            # Mimic the real detect: persist checked_at (the job commits it).
            state = dict(getattr(entity, "source_state", None) or {})
            state["checked_at"] = datetime.utcnow().isoformat()
            entity.source_state = state
        return dict(detect_result)

    monkeypatch.setattr(inc, "detect_changes", fake_detect)

    async def fake_generate(entity, p_orm, model=None, language="ru",
                            progress=None, should_cancel=None, force_units=None):
        gen_calls.append({"force_units": force_units})
        entity.generated_docs = "# updated"
        return "# updated"

    monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

    job_id = jobs_mod.create_job(PROD, "codebase", CB)
    asyncio.run(jobs_mod._run_docgen_job_async(
        job_id, PROD, "codebase", CB, None, "ru", incremental=True,
    ))
    return job_id


class TestRunDocgenJobAsyncIncremental:
    def test_unchanged_success_without_version(self, isolated_db, monkeypatch):
        from api.repositories import doc_version_repo

        _seed_codebase(isolated_db)
        job_id = _run_incremental(
            isolated_db, monkeypatch,
            {"changed": False, "reason": "tip unchanged", "impacted_pages": []},
            gen_calls=[],
        )
        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert "Изменений не обнаружено." in job["indexing_message"]
        assert job["docs_chars"] == 0

        db = isolated_db.SessionLocal()
        try:
            from api.models import CodebaseORM

            versions = doc_version_repo.list_versions(db, "codebase", CB)
            cb = db.query(CodebaseORM).filter(CodebaseORM.id == CB).one()
            assert versions == []
            assert cb.current_version is None
            # checked_at persisted by the detect phase commit
            assert (cb.source_state or {}).get("checked_at")
            assert cb.generated_docs == "# docs"  # untouched
        finally:
            db.close()

    def test_unverifiable_success_without_version(self, isolated_db, monkeypatch):
        _seed_codebase(isolated_db)
        gen_calls = []
        job_id = _run_incremental(
            isolated_db, monkeypatch,
            {"changed": None, "reason": "нет baseline", "impacted_pages": []},
            gen_calls,
        )
        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert "не удалось проверить" in job["indexing_message"]
        assert "нет baseline" in job["indexing_message"]
        assert job["docs_chars"] == 0
        assert gen_calls == []

    def test_changed_regenerates_only_impacted(self, isolated_db, monkeypatch):
        from api.repositories import doc_version_repo

        _seed_codebase(isolated_db)
        gen_calls = []
        job_id = _run_incremental(
            isolated_db, monkeypatch,
            {"changed": True, "reason": "1 file", "impacted_pages": [
                {"id": "page_a", "title": "A"},
            ]},
            gen_calls,
        )
        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["docs_chars"] == len("# updated")
        assert job["result"]["impacted_pages"][0]["id"] == "page_a"
        assert gen_calls == [{"force_units": ["page_a"]}]

        db = isolated_db.SessionLocal()
        try:
            from api.models import CodebaseORM

            versions = doc_version_repo.list_versions(db, "codebase", CB)
            assert [v.version for v in versions] == [2, 1]
            assert [v.source for v in versions] == ["generate", "baseline"]
            cb = db.query(CodebaseORM).filter(CodebaseORM.id == CB).one()
            assert cb.current_version == 2
            assert cb.generated_docs == "# updated"
        finally:
            db.close()

    def test_changed_empty_impact_forces_full_regeneration(
        self, isolated_db, monkeypatch,
    ):
        _seed_codebase(isolated_db)
        gen_calls = []
        _run_incremental(
            isolated_db, monkeypatch,
            {"changed": True, "reason": "всё изменилось", "impacted_pages": []},
            gen_calls,
        )
        assert gen_calls == [{"force_units": None}]


class TestRunDetectJobAsync:
    def _run(self, db_mod, monkeypatch, fake_detect):
        import api.docgen.incremental as inc

        monkeypatch.setattr(jobs_mod, "SessionLocal", db_mod.SessionLocal)
        monkeypatch.setattr(inc, "detect_changes", fake_detect)
        job_id = jobs_mod.create_job(PROD, "detect:codebase", CB)
        asyncio.run(jobs_mod._run_detect_job_async(
            job_id, PROD, "codebase", CB, None,
        ))
        return job_id

    def test_success_persists_checked_at_and_result(self, isolated_db, monkeypatch):
        _seed_codebase(isolated_db)

        async def fake_detect(entity, p_orm, entity_type, model=None):
            state = dict(entity.source_state or {})
            state["checked_at"] = datetime.utcnow().isoformat()
            entity.source_state = state
            return {"changed": True, "reason": "files changed", "impacted_pages": []}

        job_id = self._run(isolated_db, monkeypatch, fake_detect)
        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["entity_type"] == "detect:codebase"
        assert job["result"]["changed"] is True
        assert job["error"] is None

        db = isolated_db.SessionLocal()
        try:
            from api.models import CodebaseORM

            cb = db.query(CodebaseORM).filter(CodebaseORM.id == CB).one()
            assert (cb.source_state or {}).get("checked_at")
            assert cb.source_state["baseline"] == BASELINE
        finally:
            db.close()

    def test_value_error_fails_job_with_message(self, isolated_db, monkeypatch):
        _seed_codebase(isolated_db)

        async def fake_detect(entity, p_orm, entity_type, model=None):
            raise ValueError("MCP down")

        job_id = self._run(isolated_db, monkeypatch, fake_detect)
        job = jobs_mod.get_job(job_id)
        assert job["status"] == "failed"
        assert "MCP down" in job["error"]
        assert "MCP down" in job["indexing_message"]


class TestEntityBusyAndCancel:
    """Q-2: detect jobs (key detect:{type}) must be visible to the busy
    semantics and cancellable via request_cancel_for_entity."""

    def test_entity_busy_counts_detect_jobs(self):
        assert jobs_mod.entity_busy("codebase", CB) is False
        job_id, is_new = jobs_mod.create_or_get_job(PROD, "detect:codebase", CB)
        assert is_new
        assert jobs_mod.entity_busy("codebase", CB) is True
        job = jobs_mod.get_job(job_id)
        job["status"] = "succeeded"
        job["finished_at"] = 1.0
        assert jobs_mod.entity_busy("codebase", CB) is False

    def test_cancel_matches_detect_job(self):
        job_id, is_new = jobs_mod.create_or_get_job(PROD, "detect:codebase", CB)
        assert is_new
        jobs_mod.get_job(job_id)["status"] = "running"
        assert jobs_mod.request_cancel_for_entity(PROD, "codebase", CB) == job_id
        assert jobs_mod.get_job(job_id)["cancel_requested"] is True

    def test_cancel_prefers_generation_job_over_detect(self):
        gen_id, _ = jobs_mod.create_or_get_job(PROD, "codebase", CB)
        det_id, _ = jobs_mod.create_or_get_job(PROD, "detect:codebase", CB)
        jobs_mod.get_job(gen_id)["status"] = "running"
        jobs_mod.get_job(det_id)["status"] = "running"
        assert jobs_mod.request_cancel_for_entity(PROD, "codebase", CB) == gen_id
        assert jobs_mod.get_job(det_id)["cancel_requested"] is False

    def test_cancel_unknown_entity_is_none(self):
        assert jobs_mod.request_cancel_for_entity(PROD, "codebase", CB) is None


class TestIndexPagesInBackground:
    @staticmethod
    def _scenario(monkeypatch, stored):
        import api.docgen._common as common
        import api.memory as memory_mod

        indexed, pruned = [], []

        async def fake_index(content, product_id, source_type="codebase",
                             source_id=None, **kwargs):
            indexed.append(source_id)
            stored.setdefault("pages", set()).add(source_id)
            return 1

        async def fake_list(product_id, source_type, source_id):
            return sorted(stored.get("pages", set()))

        async def fake_prune(product_id, source_type, source_id, keep):
            pruned.append((source_id, list(keep)))
            return 0

        monkeypatch.setattr(memory_mod, "index_document", fake_index)
        monkeypatch.setattr(memory_mod, "list_page_sources", fake_list)
        monkeypatch.setattr(memory_mod, "prune_stale_sources", fake_prune)
        # No captured main loop: force the local create_task fallback.
        monkeypatch.setattr(common, "get_main_event_loop", lambda: None)

        pages = {
            "a": {"content": "AAA"},
            "b": {"content": "BBB"},
            "c": {"content": "CCC"},
        }

        async def run():
            common._index_pages_in_background(
                pages, "prod_1", source_type="codebase",
                source_id="cb_1", only=["a"],
            )
            await asyncio.sleep(0)

        asyncio.run(run())
        return indexed, pruned

    def test_first_incremental_run_bootstraps_all_pages(self, monkeypatch):
        indexed, pruned = self._scenario(monkeypatch, stored={})
        # Legacy blob-indexed artifact (no page sources yet): bootstrap ALL
        # pages so the prune cannot drop unchanged pages from recall.
        assert indexed == ["cb_1::page::a", "cb_1::page::b", "cb_1::page::c"]
        source_id, keep = pruned[0]
        assert source_id == "cb_1"
        assert sorted(keep) == [
            "cb_1::page::a", "cb_1::page::b", "cb_1::page::c",
        ]

    def test_subsequent_run_reindexes_only_forced_pages(self, monkeypatch):
        stored = {"pages": {"cb_1::page::a", "cb_1::page::b", "cb_1::page::c"}}
        indexed, _ = self._scenario(monkeypatch, stored)
        assert indexed == ["cb_1::page::a"]

    def test_noop_without_source_id(self, monkeypatch):
        import api.docgen._common as common

        monkeypatch.setattr(common, "get_main_event_loop", lambda: None)

        async def run():
            common._index_pages_in_background(
                {"a": {"content": "AAA"}}, "prod_1",
                source_type="codebase", source_id=None, only=["a"],
            )
            await asyncio.sleep(0)

        asyncio.run(run())  # no scheduling, no crash


class TestPgVectorPageSources:
    def _seed_chunks(self, db_mod):
        from api.models import KnowledgeChunkORM, ProductORM

        rows = [
            ("c1", "cb_1"),            # legacy blob source
            ("c2", "cb_1::page::a"),
            ("c3", "cb_1::page::gone"),
            ("c4", "cb_1::page::b"),
            ("c5", "other_cb"),        # foreign artifact: never touched
        ]
        db = db_mod.SessionLocal()
        try:
            db.add(ProductORM(id="p1", name="P"))
            for cid, src in rows:
                db.add(KnowledgeChunkORM(
                    id=cid, product_id="p1", source_type="codebase",
                    source_id=src, chunk_index=0, content="chunk " + cid,
                ))
            db.commit()
        finally:
            db.close()

    def test_list_and_prune_page_sources(self, isolated_db):
        self._seed_chunks(isolated_db)
        backend = PgVectorMemoryBackend()

        sources = asyncio.run(backend.list_page_sources("p1", "codebase", "cb_1"))
        assert sorted(sources) == ["cb_1::page::a", "cb_1::page::b", "cb_1::page::gone"]

        deleted = asyncio.run(backend.prune_stale_sources(
            "p1", "codebase", "cb_1",
            ["cb_1::page::a", "cb_1::page::b"],
        ))
        assert deleted == 2  # legacy blob + the stale `gone` page

        db = isolated_db.SessionLocal()
        try:
            from api.models import KnowledgeChunkORM

            remaining = {
                r[0] for r in db.query(KnowledgeChunkORM.source_id).all()
            }
            assert remaining == {
                "cb_1::page::a", "cb_1::page::b", "other_cb",
            }
        finally:
            db.close()


class TestDetectEndpoints:
    def _client(self, isolated_db, monkeypatch):
        import api.routers.docgen as docgen_mod
        from api.models import CodebaseORM, ProductORM, SpecORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P"))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo",
                repo_url="https://github.com/o/r",
                source_state={"kind": "codebase", "baseline": BASELINE},
            ))
            db.add(SpecORM(id="s1", product_id="prod_1", name="api", kind="openapi"))
            db.commit()
        finally:
            db.close()

        submitted = {"detect": [], "incremental": []}
        monkeypatch.setattr(
            docgen_mod, "submit_detect_job",
            lambda *a, **kw: (submitted["detect"].append(a) or ("job_x", True)),
        )
        monkeypatch.setattr(
            docgen_mod, "submit_incremental_job",
            lambda *a, **kw: (submitted["incremental"].append(a) or ("job_y", True)),
        )
        app, client = build_test_client(isolated_db, [docgen_mod])
        return client, submitted

    def test_detect_accepted(self, isolated_db, monkeypatch):
        client, submitted = self._client(isolated_db, monkeypatch)
        resp = client.post("/api/products/prod_1/codebases/cb_1/detect")
        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "job_x"
        assert body["reused"] is False
        assert body["entity_type"] == "codebase"
        assert submitted["detect"] == [("prod_1", "codebase", "cb_1")]

    def test_detect_rejects_hld(self, isolated_db, monkeypatch):
        client, submitted = self._client(isolated_db, monkeypatch)
        resp = client.post("/api/products/prod_1/hlds/hld_1/detect")
        assert resp.status_code == 400
        assert submitted["detect"] == []

    def test_detect_unknown_entity_404(self, isolated_db, monkeypatch):
        client, submitted = self._client(isolated_db, monkeypatch)
        resp = client.post("/api/products/prod_1/codebases/nope/detect")
        assert resp.status_code == 404
        assert submitted["detect"] == []

    def test_incremental_accepted(self, isolated_db, monkeypatch):
        client, _ = self._client(isolated_db, monkeypatch)
        resp = client.post("/api/products/prod_1/specs/s1/incremental", json={})
        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "job_y"
        assert body["entity_type"] == "spec"

    def test_detect_status_returns_result(self, isolated_db, monkeypatch):
        client, _ = self._client(isolated_db, monkeypatch)
        job_id = jobs_mod.create_job("prod_1", "detect:codebase", "cb_1")
        job = jobs_mod.get_job(job_id)
        job.update(
            status="succeeded", indexing_status="succeeded",
            indexing_message="Проверка изменений завершена.",
            result={"changed": False, "reason": "tip unchanged"},
        )
        resp = client.get(
            f"/api/products/prod_1/codebases/cb_1/detect/status?job_id={job_id}"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "succeeded"
        assert body["result"] == {"changed": False, "reason": "tip unchanged"}

    def test_detect_status_unknown_job_404(self, isolated_db, monkeypatch):
        client, _ = self._client(isolated_db, monkeypatch)
        resp = client.get(
            "/api/products/prod_1/codebases/cb_1/detect/status?job_id=zzz"
        )
        assert resp.status_code == 404


class TestSourceStateHidden:
    def test_product_payload_strips_source_state(self, isolated_db):
        import api.routers.products as products_mod
        from api.models import CodebaseORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P"))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo",
                repo_url="https://github.com/o/r",
                source_state={"kind": "codebase", "baseline": BASELINE},
            ))
            db.commit()
        finally:
            db.close()

        app, client = build_test_client(isolated_db, [products_mod])
        resp = client.get("/api/products/prod_1")
        assert resp.status_code == 200
        cb = resp.json()["codebases"][0]
        assert cb["id"] == "cb_1"
        assert "source_state" not in cb
        assert BASELINE not in resp.text
