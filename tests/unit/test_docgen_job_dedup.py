"""Unit tests for docgen job deduplication + per-entity serialization (2.1).

Порт H3/H4 из форка (``api/wiki_generation.py``): ключ джоба
(product_id, entity_type, entity_id), атомарный check-then-act под
``_JOBS_LOCK`` (повторный POST подхватывает идущий job и НЕ диспетчеризует
второй прогон) и per-entity lock, сериализующий генерацию одной сущности.

Hermetic: изолированная SQLite (``isolated_db``), submit_job стабируется,
LLM/воркеры не запускаются (кроме теста сериализации, где воркер-функция
вызывается напрямую с мгновенным фейком generate).
"""

from __future__ import annotations

import os
import sys
import threading
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pytest

import api.docgen.jobs as jobs_mod
from api.models import CodebaseORM, DatabaseORM, ProductORM
from api.routers import databases as databases_router_module
from api.routers import docgen as docgen_router_module
from tests.conftest import build_test_client


@pytest.fixture(autouse=True)
def _clear_registry():
    """Модульный реестр джобов — глобальное состояние: чистим между тестами."""
    jobs_mod._docgen_jobs.clear()
    jobs_mod._ENTITY_LOCKS.clear()
    yield
    jobs_mod._docgen_jobs.clear()
    jobs_mod._ENTITY_LOCKS.clear()


# ============================================================================
# create_or_get_job — реестр
# ============================================================================
class TestCreateOrGetJob:
    def test_first_call_creates_new(self):
        job_id, is_new = jobs_mod.create_or_get_job("p1", "codebase", "c1")
        assert is_new is True
        job = jobs_mod.get_job(job_id)
        assert job["status"] == "queued"
        assert job["key"] == ("p1", "codebase", "c1")

    def test_second_call_while_queued_reuses(self):
        id1, new1 = jobs_mod.create_or_get_job("p1", "codebase", "c1")
        id2, new2 = jobs_mod.create_or_get_job("p1", "codebase", "c1")
        assert new1 is True and new2 is False
        assert id1 == id2

    def test_second_call_while_running_reuses(self):
        id1, _ = jobs_mod.create_or_get_job("p1", "spec", "s1")
        jobs_mod._docgen_jobs[id1]["status"] = "running"
        id2, new2 = jobs_mod.create_or_get_job("p1", "spec", "s1")
        assert new2 is False
        assert id1 == id2

    def test_terminal_succeeded_job_not_reused(self):
        id1, _ = jobs_mod.create_or_get_job("p1", "codebase", "c1")
        jobs_mod._docgen_jobs[id1]["status"] = "succeeded"
        jobs_mod._docgen_jobs[id1]["finished_at"] = time.time()
        id2, new2 = jobs_mod.create_or_get_job("p1", "codebase", "c1")
        assert new2 is True
        assert id1 != id2

    def test_terminal_failed_job_not_reused(self):
        """Проваленный прогон не блокирует повторную генерацию."""
        id1, _ = jobs_mod.create_or_get_job("p1", "database", "d1")
        jobs_mod._docgen_jobs[id1]["status"] = "failed"
        jobs_mod._docgen_jobs[id1]["finished_at"] = time.time()
        id2, new2 = jobs_mod.create_or_get_job("p1", "database", "d1")
        assert new2 is True
        assert id1 != id2

    def test_different_entity_gets_own_job(self):
        id1, _ = jobs_mod.create_or_get_job("p1", "codebase", "c1")
        id2, _ = jobs_mod.create_or_get_job("p1", "codebase", "c2")
        id3, _ = jobs_mod.create_or_get_job("p2", "codebase", "c1")
        id4, _ = jobs_mod.create_or_get_job("p1", "spec", "c1")
        assert len({id1, id2, id3, id4}) == 4

    def test_active_job_not_pruned_by_age(self):
        """Prune смотрит только на finished_at (терминальные джобы): активный
        (queued/running) джоб не выпадает из реестра по возрасту — повторный
        POST всё ещё re-use, каким бы долгим ни был прогон."""
        id1, _ = jobs_mod.create_or_get_job("p1", "codebase", "c1")
        jobs_mod._docgen_jobs[id1]["created_at"] = time.time() - 7200
        id2, new2 = jobs_mod.create_or_get_job("p1", "codebase", "c1")
        assert new2 is False
        assert id1 == id2

    def test_create_job_still_always_new(self):
        """Прямой create_job (тесты/внутренние вызовы) обходит дедуп."""
        id1 = jobs_mod.create_job("p1", "codebase", "c1")
        id2 = jobs_mod.create_job("p1", "codebase", "c1")
        assert id1 != id2


# ============================================================================
# lock_for_entity
# ============================================================================
class TestLockForEntity:
    def test_same_key_same_lock_object(self):
        l1 = jobs_mod.lock_for_entity("p1", "codebase", "c1")
        l2 = jobs_mod.lock_for_entity("p1", "codebase", "c1")
        assert l1 is l2

    def test_different_keys_distinct_locks(self):
        l1 = jobs_mod.lock_for_entity("p1", "codebase", "c1")
        l2 = jobs_mod.lock_for_entity("p1", "codebase", "c2")
        l3 = jobs_mod.lock_for_entity("p1", "spec", "c1")
        assert l1 is not l2 and l1 is not l3 and l2 is not l3


# ============================================================================
# Роутеры: повторный POST — тот же job_id, сабмит только один раз
# ============================================================================
def _stub_submit(monkeypatch, router_module):
    """Стаб submit_job на модуле роутера: без воркер-потока."""
    from unittest.mock import MagicMock
    stub = MagicMock()
    monkeypatch.setattr(router_module, "submit_job", stub)
    return stub


def _seed_codebase(db_mod, pid="prod_dedup", cid="cb_1"):
    with db_mod.SessionLocal() as s:
        s.add(ProductORM(id=pid, name="P"))
        s.flush()
        s.add(CodebaseORM(
            id=cid, product_id=pid, name="repo",
            repo_url="https://github.com/o/r",
        ))
        s.commit()


class TestDocgenRouterDedup:
    def test_double_post_same_job_and_single_submit(self, isolated_db, monkeypatch):
        _seed_codebase(isolated_db)
        stub = _stub_submit(monkeypatch, docgen_router_module)
        app, client = build_test_client(isolated_db, [docgen_router_module])

        r1 = client.post("/api/products/prod_dedup/codebases/cb_1/generate", json={})
        assert r1.status_code == 202
        b1 = r1.json()
        assert b1["reused"] is False
        assert b1["status"] == "queued"

        r2 = client.post("/api/products/prod_dedup/codebases/cb_1/generate", json={})
        assert r2.status_code == 202
        b2 = r2.json()
        assert b2["job_id"] == b1["job_id"]
        assert b2["reused"] is True

        assert stub.call_count == 1  # второй прогон НЕ диспетчеризуется

    def test_reuse_reflects_running_status(self, isolated_db, monkeypatch):
        """Повторный POST возвращает фактический статус идущего джоба."""
        _seed_codebase(isolated_db)
        _stub_submit(monkeypatch, docgen_router_module)
        app, client = build_test_client(isolated_db, [docgen_router_module])

        b1 = client.post(
            "/api/products/prod_dedup/codebases/cb_1/generate", json={}
        ).json()
        jobs_mod._docgen_jobs[b1["job_id"]]["status"] = "running"

        b2 = client.post(
            "/api/products/prod_dedup/codebases/cb_1/generate", json={}
        ).json()
        assert b2["job_id"] == b1["job_id"]
        assert b2["status"] == "running"

    def test_different_entities_both_submitted(self, isolated_db, monkeypatch):
        _seed_codebase(isolated_db, cid="cb_a")
        with isolated_db.SessionLocal() as s:
            s.add(CodebaseORM(
                id="cb_b", product_id="prod_dedup", name="repo2",
                repo_url="https://github.com/o/r2",
            ))
            s.commit()
        stub = _stub_submit(monkeypatch, docgen_router_module)
        app, client = build_test_client(isolated_db, [docgen_router_module])

        b1 = client.post("/api/products/prod_dedup/codebases/cb_a/generate", json={}).json()
        b2 = client.post("/api/products/prod_dedup/codebases/cb_b/generate", json={}).json()
        assert b1["job_id"] != b2["job_id"]
        assert stub.call_count == 2

    def test_after_terminal_new_job_submitted(self, isolated_db, monkeypatch):
        _seed_codebase(isolated_db)
        stub = _stub_submit(monkeypatch, docgen_router_module)
        app, client = build_test_client(isolated_db, [docgen_router_module])

        b1 = client.post(
            "/api/products/prod_dedup/codebases/cb_1/generate", json={}
        ).json()
        job = jobs_mod._docgen_jobs[b1["job_id"]]
        job["status"] = "succeeded"
        job["finished_at"] = time.time()

        b2 = client.post(
            "/api/products/prod_dedup/codebases/cb_1/generate", json={}
        ).json()
        assert b2["job_id"] != b1["job_id"]
        assert b2["reused"] is False
        assert stub.call_count == 2


class TestDatabaseRouterDedup:
    def test_double_post_same_job_and_single_submit(self, isolated_db, monkeypatch):
        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_db", name="P"))
            s.flush()
            s.add(DatabaseORM(
                id="db_1", product_id="prod_db", name="Main DB",
                dsn_masked="postgresql://app:***@h/db", source="manual",
            ))
            s.commit()
        stub = _stub_submit(monkeypatch, databases_router_module)
        app, client = build_test_client(isolated_db, [databases_router_module])

        b1 = client.post("/api/products/prod_db/databases/db_1/generate", json={}).json()
        b2 = client.post("/api/products/prod_db/databases/db_1/generate", json={}).json()
        assert b2["job_id"] == b1["job_id"]
        assert b2["reused"] is True
        assert stub.call_count == 1


# ============================================================================
# Воркер: per-entity lock сериализует два прогона одной сущности
# ============================================================================
class TestWorkerEntityLockSerializes:
    def test_two_same_entity_runs_never_overlap(self, isolated_db, monkeypatch):
        """Два джоба одной сущности (созданные в обход дедупа — сценарий
        defense-in-depth) сериализуются локом: генерации не пересекаются."""
        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_ser", name="P"))
            s.flush()
            s.add(CodebaseORM(
                id="cb_ser", product_id="prod_ser", name="repo",
                repo_url="https://github.com/o/r",
            ))
            s.commit()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        overlap = {"now": 0, "max": 0}
        guard = threading.Lock()

        import api.docgen.codebase as codebase_mod

        async def fake_generate(artifact, product, model=None, language="ru", progress=None):
            with guard:
                overlap["now"] += 1
                overlap["max"] = max(overlap["max"], overlap["now"])
            time.sleep(0.15)  # окно, в котором прогоны пересеклись бы без лока
            with guard:
                overlap["now"] -= 1
            return "docs"

        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        j1 = jobs_mod.create_job("prod_ser", "codebase", "cb_ser")
        j2 = jobs_mod.create_job("prod_ser", "codebase", "cb_ser")

        errors = {}

        def _run(jid):
            try:
                jobs_mod._run_docgen_job(
                    jid, "prod_ser", "codebase", "cb_ser", None, "ru"
                )
            except Exception as e:  # pragma: no cover - defensive
                errors[jid] = repr(e)

        t1 = threading.Thread(target=_run, args=(j1,))
        t2 = threading.Thread(target=_run, args=(j2,))
        t1.start()
        t2.start()
        t1.join(15)
        t2.join(15)
        assert not t1.is_alive() and not t2.is_alive(), "worker run hung"
        assert not errors, errors

        assert overlap["max"] == 1, f"generations overlided: {overlap}"
        assert jobs_mod.get_job(j1)["status"] == "succeeded"
        assert jobs_mod.get_job(j2)["status"] == "succeeded"

    def test_lock_released_after_run(self, isolated_db, monkeypatch):
        """Лок освобождается по завершении прогона (не остаётся залоченным)."""
        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_rel", name="P"))
            s.flush()
            s.add(CodebaseORM(
                id="cb_rel", product_id="prod_rel", name="repo",
                repo_url="https://github.com/o/r",
            ))
            s.commit()

        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        import api.docgen.codebase as codebase_mod

        async def fake_generate(artifact, product, model=None, language="ru", progress=None):
            return "ok"

        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        j1 = jobs_mod.create_job("prod_rel", "codebase", "cb_rel")
        jobs_mod._run_docgen_job(j1, "prod_rel", "codebase", "cb_rel", None, "ru")
        assert jobs_mod.get_job(j1)["status"] == "succeeded"

        lock = jobs_mod.lock_for_entity("prod_rel", "codebase", "cb_rel")
        acquired = lock.acquire(timeout=0.5)
        assert acquired, "entity lock was not released after the run"
        lock.release()
