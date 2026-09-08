"""Unit tests for per-entity docgen/API write locks (P1-18/19/22).

Covers:
- ``lock_for_entity`` refcounted registry: entry removed when the last
  holder/waiter leaves (single, nested/reentrant, and after EntityBusyError).
- Mutual exclusion: a second acquirer times out with ``EntityBusyError``
  while another thread holds the lock; same-thread reentrancy is allowed.
- Repo layer: ``update_codebase_content`` / ``_delete_child`` raise
  ``EntityBusyError`` while the entity lock is held, then succeed after
  release (no partial writes).
- Router layer: mutating endpoints return 409 while a docgen job holds the
  entity lock, data is preserved, and the same request succeeds after the
  job releases (deterministic generate-vs-PUT race).
- ``_run_docgen_job`` marks the job failed (busy) instead of racing when
  another job already holds the lock.
- P1-18: ``PUT /api/products/{id}`` — the PATH id wins over any body id, so
  a mismatched body can never spawn a second product.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.docgen import jobs as docgen_jobs
from api.docgen.jobs import EntityBusyError, lock_for_entity
from api.models import CodebaseORM, ProductORM, SpecORM, LinksORM
from api.repositories import product_repo as pr
from api.routers import products as products_router_module
from tests.conftest import build_test_client


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clean_lock_registry():
    """The lock registry must be empty around every test (P1-22 cleanup)."""
    assert not docgen_jobs._ENTITY_LOCKS, f"leaked locks before test: {docgen_jobs._ENTITY_LOCKS}"
    assert not docgen_jobs._ENTITY_REFCOUNTS
    yield
    try:
        assert not docgen_jobs._ENTITY_LOCKS, (
            f"leaked locks after test: {docgen_jobs._ENTITY_LOCKS}"
        )
        assert not docgen_jobs._ENTITY_REFCOUNTS
    finally:
        # Never cascade one leak into every later test.
        docgen_jobs._ENTITY_LOCKS.clear()
        docgen_jobs._ENTITY_REFCOUNTS.clear()


@pytest.fixture(autouse=True)
def _fast_lock_timeout(monkeypatch):
    """Keep 409 tests fast: API waits only 0.2s before failing busy."""
    monkeypatch.setattr(docgen_jobs, "_ENTITY_LOCK_TIMEOUT", 0.2)


def _product_payload(pid: str = "prod_1") -> dict:
    return {
        "id": pid,
        "name": "Widget",
        "description": "A widget",
        "summary": "sum",
        "owner_id": None,
        "codebases": [],
        "specs": [],
        "links": [],
    }


def _seed_product(db_mod, pid: str = "prod_1") -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(ProductORM(id=pid, name="Widget", description="desc", summary=None, owner_id=None))
        s.commit()
    finally:
        s.close()


def _seed_codebase(
    db_mod, pid: str = "prod_1", cid: str = "cb_1", docs: str | None = "OLD DOCS"
) -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(CodebaseORM(
            id=cid, product_id=pid, name="Repo A", source="manual", generated_docs=docs,
        ))
        s.commit()
    finally:
        s.close()


def _seed_spec(db_mod, pid: str = "prod_1", sid: str = "spec_1") -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(SpecORM(id=sid, product_id=pid, name="S", kind="openapi", source="manual"))
        s.commit()
    finally:
        s.close()


def _seed_links(db_mod, pid: str = "prod_1", lid: str = "links_1") -> None:
    s = db_mod.SessionLocal()
    try:
        s.add(LinksORM(id=lid, product_id=pid, name="L", source="manual"))
        s.commit()
    finally:
        s.close()


def _make_client(db_mod):
    return build_test_client(db_mod, [products_router_module], auth_none=True)


def _disable_reindex(monkeypatch):
    monkeypatch.setattr(products_router_module, "_reindex", lambda *a, **kw: None)


class _HeldLock:
    """Hold an entity lock from a background thread, exactly like a running
    docgen job (the worker thread). The locks are RLocks, so holding from the
    test's own thread would NOT block same-thread acquires."""

    def __init__(self, entity_type: str, entity_id: str):
        self._entity_type = entity_type
        self._entity_id = entity_id
        self._held = threading.Event()
        self._release = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        def _holder():
            with lock_for_entity(self._entity_type, self._entity_id):
                self._held.set()
                self._release.wait(timeout=30)

        self._thread = threading.Thread(target=_holder, daemon=True)
        self._thread.start()
        assert self._held.wait(timeout=5), "lock holder thread never started"
        return self

    def __exit__(self, *exc):
        self._release.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False


# --------------------------------------------------------------------------- #
# Registry hygiene (P1-22)
# --------------------------------------------------------------------------- #
class TestRegistryCleanup:
    def test_registry_empty_after_context_exit(self):
        with lock_for_entity("codebase", "cb_x"):
            assert docgen_jobs._ENTITY_LOCKS
            assert docgen_jobs._ENTITY_REFCOUNTS["codebase:cb_x"] == 1
        assert not docgen_jobs._ENTITY_LOCKS
        assert not docgen_jobs._ENTITY_REFCOUNTS

    def test_registry_empty_after_nested_reentrant_context(self):
        with lock_for_entity("spec", "spec_x"):
            with lock_for_entity("spec", "spec_x"):  # RLock: same thread re-entry
                assert docgen_jobs._ENTITY_REFCOUNTS["spec:spec_x"] == 2
            assert docgen_jobs._ENTITY_REFCOUNTS["spec:spec_x"] == 1
        assert not docgen_jobs._ENTITY_LOCKS
        assert not docgen_jobs._ENTITY_REFCOUNTS

    def test_registry_empty_after_busy_error(self):
        with _HeldLock("links", "links_x"):
            with pytest.raises(EntityBusyError):
                with lock_for_entity("links", "links_x", timeout=0.05):
                    pass  # pragma: no cover - never reached
        assert not docgen_jobs._ENTITY_LOCKS
        assert not docgen_jobs._ENTITY_REFCOUNTS

    def test_distinct_entities_do_not_conflict(self):
        with lock_for_entity("codebase", "cb_a"):
            with lock_for_entity("codebase", "cb_b"):  # different id -> no block
                assert set(docgen_jobs._ENTITY_LOCKS) == {"codebase:cb_a", "codebase:cb_b"}
        assert not docgen_jobs._ENTITY_LOCKS


# --------------------------------------------------------------------------- #
# Mutual exclusion (P1-19)
# --------------------------------------------------------------------------- #
class TestMutualExclusion:
    def test_second_acquire_times_out_with_entity_busy(self):
        with _HeldLock("codebase", "cb_race"):
            with pytest.raises(EntityBusyError, match="busy"):
                with lock_for_entity("codebase", "cb_race", timeout=0.05):
                    pass  # pragma: no cover

    def test_lock_available_again_after_release(self):
        with _HeldLock("codebase", "cb_seq"):
            pass
        # No timeout: would hang forever if the released entry were stuck.
        with lock_for_entity("codebase", "cb_seq", timeout=1.0):
            pass

    def test_cross_thread_exclusion(self):
        held = threading.Event()
        release = threading.Event()

        def _holder():
            with lock_for_entity("spec", "spec_race"):
                held.set()
                release.wait(timeout=5)

        t = threading.Thread(target=_holder)
        t.start()
        try:
            assert held.wait(timeout=2)
            with pytest.raises(EntityBusyError):
                with lock_for_entity("spec", "spec_race", timeout=0.05):
                    pass  # pragma: no cover
        finally:
            release.set()
            t.join(timeout=5)
        assert not docgen_jobs._ENTITY_LOCKS


# --------------------------------------------------------------------------- #
# Repo layer: writes fail fast while a job holds the lock
# --------------------------------------------------------------------------- #
class TestRepoWritesLocked:
    def test_update_codebase_content_busy_then_ok(self, isolated_db, session):
        _seed_product(isolated_db)
        _seed_codebase(isolated_db, docs="OLD")
        with _HeldLock("codebase", "cb_1"):
            with pytest.raises(EntityBusyError):
                pr.update_codebase_content(
                    isolated_db.SessionLocal(), "prod_1", "cb_1", generated_docs="NEW"
                )
            # Data untouched while locked.
            cb = session.query(CodebaseORM).filter(CodebaseORM.id == "cb_1").one()
            assert cb.generated_docs == "OLD"
        # After release the same write succeeds.
        pr.update_codebase_content(
            isolated_db.SessionLocal(), "prod_1", "cb_1", generated_docs="NEW"
        )
        session.expire_all()
        cb = session.query(CodebaseORM).filter(CodebaseORM.id == "cb_1").one()
        assert cb.generated_docs == "NEW"

    def test_delete_child_busy(self, isolated_db):
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        with _HeldLock("codebase", "cb_1"):
            with pytest.raises(EntityBusyError):
                pr.delete_codebase(isolated_db.SessionLocal(), "prod_1", "cb_1")


# --------------------------------------------------------------------------- #
# Router layer: 409 while a docgen job holds the entity lock
# --------------------------------------------------------------------------- #
class TestRouter409:
    def test_put_codebase_docs_409_then_success(self, isolated_db, monkeypatch):
        """Deterministic generate-vs-PUT race: job holds lock -> PUT 409 with
        data preserved; after the job finishes, the same PUT succeeds."""
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_codebase(isolated_db, docs="OLD")
        app, client = _make_client(isolated_db)

        with _HeldLock("codebase", "cb_1"):  # simulates the running docgen job
            r = client.put(
                "/api/products/prod_1/codebases/cb_1",
                json={"generated_docs": "NEW"},
            )
            assert r.status_code == 409
            assert "busy" in r.json()["detail"]

            # The failed write left the stored docs untouched.
            s = isolated_db.SessionLocal()
            try:
                cb = s.query(CodebaseORM).filter(CodebaseORM.id == "cb_1").one()
                assert cb.generated_docs == "OLD"
            finally:
                s.close()

        # Job done (lock released): the same edit now succeeds.
        r = client.put(
            "/api/products/prod_1/codebases/cb_1",
            json={"generated_docs": "NEW"},
        )
        assert r.status_code == 200
        s = isolated_db.SessionLocal()
        try:
            cb = s.query(CodebaseORM).filter(CodebaseORM.id == "cb_1").one()
            assert cb.generated_docs == "NEW"
        finally:
            s.close()

    def test_delete_codebase_409_while_locked(self, isolated_db):
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        app, client = _make_client(isolated_db)
        with _HeldLock("codebase", "cb_1"):
            r = client.delete("/api/products/prod_1/codebases/cb_1")
            assert r.status_code == 409

    def test_add_codebase_with_same_id_409_while_locked(self, isolated_db):
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        app, client = _make_client(isolated_db)
        with _HeldLock("codebase", "cb_1"):
            r = client.post(
                "/api/products/prod_1/codebases",
                json={
                    "id": "cb_1",
                    "name": "Repo A",
                    "repo_url": "https://github.com/x/y",
                    "repo_type": "github",
                    "token": None,
                    "generated_docs": None,
                    "pages": None,
                    "verified": False,
                    "verified_by": None,
                    "verified_at": None,
                    "source": "manual",
                },
            )
            assert r.status_code == 409

    def test_put_spec_and_links_409_while_locked(self, isolated_db):
        _seed_product(isolated_db)
        _seed_spec(isolated_db)
        _seed_links(isolated_db)
        app, client = _make_client(isolated_db)
        with _HeldLock("spec", "spec_1"):
            r = client.put(
                "/api/products/prod_1/specs/spec_1", json={"content": "openapi: 3.1"}
            )
            assert r.status_code == 409
        with _HeldLock("links", "links_1"):
            r = client.put(
                "/api/products/prod_1/links/links_1", json={"content": "[]"}
            )
            assert r.status_code == 409

    def test_put_product_409_while_child_locked(self, isolated_db):
        """Whole-product PUT takes all child locks, so a running job on one
        child blocks the replace with 409 instead of clobbering it."""
        _seed_product(isolated_db)
        _seed_codebase(isolated_db)
        app, client = _make_client(isolated_db)
        with _HeldLock("codebase", "cb_1"):
            r = client.put("/api/products/prod_1", json=_product_payload("prod_1"))
            assert r.status_code == 409


# --------------------------------------------------------------------------- #
# Job side: a second job fails fast instead of racing
# --------------------------------------------------------------------------- #
class TestDocgenJobBusy:
    def test_run_docgen_job_marks_failed_when_entity_busy(self, isolated_db):
        job_id = docgen_jobs.create_job("prod_1", "codebase", "cb_1")
        with _HeldLock("codebase", "cb_1"):
            # Runs synchronously in this thread like the worker would.
            docgen_jobs._run_docgen_job(
                job_id, "prod_1", "codebase", "cb_1", model=None, language="ru"
            )
        job = docgen_jobs.get_job(job_id)
        assert job is not None
        assert job["status"] == "failed"
        assert "busy" in (job["error"] or "")
        assert job["finished_at"] is not None


# --------------------------------------------------------------------------- #
# P1-18: PUT path-id priority
# --------------------------------------------------------------------------- #
class TestPutPathIdPriority:
    def test_path_id_wins_over_mismatched_body_id(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)

        body = _product_payload("prod_other")
        body["name"] = "Renamed"
        r = client.put("/api/products/prod_1", json=body)
        assert r.status_code == 200
        assert r.json()["id"] == "prod_1"
        assert r.json()["name"] == "Renamed"

        # Exactly one product exists; nothing was spawned for the body id.
        r = client.get("/api/products")
        assert r.status_code == 200
        assert len(r.json()) == 1
        assert [p["id"] for p in r.json()] == ["prod_1"]

        assert client.get("/api/products/prod_other").status_code == 404
        assert client.get("/api/products/prod_1").json()["name"] == "Renamed"

    def test_matching_body_id_unchanged(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        body = _product_payload("prod_1")
        body["name"] = "Renamed too"
        r = client.put("/api/products/prod_1", json=body)
        assert r.status_code == 200
        assert r.json()["id"] == "prod_1"
        assert len(client.get("/api/products").json()) == 1
