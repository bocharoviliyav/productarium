"""Unit tests for api.memory.preflight (2.2 — embedder preflight).

Порт coverage-модели форка (test_embedder_diagnostics.py), адаптированный под
единый OpenAI-compatible стек: классификация отказов по тексту ошибки, проба
через get_embedder().embed_documents, таймаут, сверка размерности с существующими
чанками продукта, пропуск на не-pgvector инсталляции + интеграция с джоб-воркером
(EMBEDDER_ERROR: префикс в job["error"], генерация не стартует).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pytest

import api.memory.preflight as pf
from api.memory.preflight import (
    EMBEDDER_ERROR_PREFIX,
    EmbedderUnavailable,
    _classify,
    preflight_embedder,
)


@pytest.fixture(autouse=True)
def _pg_ready(monkeypatch):
    """По умолчанию считаем инсталляцию pgvector-capable (иначе всё пропускается)."""
    monkeypatch.setattr(pf, "_pgvector_ready", lambda: True)


# ============================================================================
# _classify — классификация по тексту ошибки
# ============================================================================
class TestClassify:
    def test_auth_markers(self):
        for detail in (
            "401 Unauthorized",
            "Error code: 403 - Forbidden",
            "Incorrect API key provided",
            "invalid credentials",
        ):
            assert _classify(detail) == "auth", detail

    def test_model_missing_markers(self):
        for detail in (
            "Error code: 404 - The model `x` does not exist",
            "model not found",
        ):
            assert _classify(detail) == "model_missing", detail

    def test_everything_else_is_unreachable(self):
        for detail in (
            "Connection refused",
            "Read timeout",
            "SSL error",
            "",
        ):
            assert _classify(detail) == "unreachable", detail


# ============================================================================
# Проба: успех / классифицированные отказы / таймаут / пустой вектор
# ============================================================================
def _fake_embedder_module(monkeypatch, vectors=None, error=None, delay=0.0):
    """Подменяет api.tools.embedder.get_embedder на детерминированный фейк."""
    import api.tools.embedder as embedder_mod

    class _FakeEmbedder:
        def embed_documents(self, items):
            if delay:
                time.sleep(delay)
            if error is not None:
                raise error
            return list(vectors or [])

    monkeypatch.setattr(embedder_mod, "get_embedder", lambda *a, **kw: _FakeEmbedder())


class TestProbe:
    def test_success_returns_dimension(self, monkeypatch):
        _fake_embedder_module(monkeypatch, vectors=[[0.1] * 768])
        monkeypatch.setattr(pf, "_check_dimension", lambda dim, pid: None)
        assert asyncio.run(preflight_embedder("p1")) == 768

    def test_auth_failure(self, monkeypatch):
        _fake_embedder_module(
            monkeypatch, error=RuntimeError("Error code: 401 - Unauthorized")
        )
        with pytest.raises(EmbedderUnavailable) as e:
            asyncio.run(preflight_embedder("p1"))
        assert e.value.reason == "auth"
        assert EMBEDDER_ERROR_PREFIX not in str(e.value)  # префикс ставит воркер

    def test_model_missing_failure(self, monkeypatch):
        _fake_embedder_module(monkeypatch, error=RuntimeError("model not found"))
        with pytest.raises(EmbedderUnavailable) as e:
            asyncio.run(preflight_embedder("p1"))
        assert e.value.reason == "model_missing"

    def test_unreachable_failure(self, monkeypatch):
        _fake_embedder_module(monkeypatch, error=RuntimeError("Connection refused"))
        with pytest.raises(EmbedderUnavailable) as e:
            asyncio.run(preflight_embedder("p1"))
        assert e.value.reason == "unreachable"
        assert "Connection refused" in str(e.value)

    def test_misconfigured_when_no_embedder_config(self, monkeypatch):
        _fake_embedder_module(
            monkeypatch, error=ValueError("No embedder configuration found.")
        )
        with pytest.raises(EmbedderUnavailable) as e:
            asyncio.run(preflight_embedder("p1"))
        assert e.value.reason == "misconfigured"

    def test_timeout_is_unreachable(self, monkeypatch):
        _fake_embedder_module(monkeypatch, vectors=[[0.0]], delay=5.0)
        with pytest.raises(EmbedderUnavailable) as e:
            asyncio.run(pf._probe(timeout=0.05))
        assert e.value.reason == "unreachable"
        assert "0 с" in str(e.value) or "эмбеддер" in str(e.value).lower()

    def test_empty_vector_is_unreachable(self, monkeypatch):
        _fake_embedder_module(monkeypatch, vectors=[[]])
        monkeypatch.setattr(pf, "_check_dimension", lambda dim, pid: None)
        with pytest.raises(EmbedderUnavailable) as e:
            asyncio.run(preflight_embedder("p1"))
        assert e.value.reason == "unreachable"
        assert "пустой вектор" in str(e.value)

    def test_failure_logs_structured_line(self, monkeypatch, caplog):
        import logging

        _fake_embedder_module(monkeypatch, error=RuntimeError("Error code: 401"))
        with caplog.at_level(logging.ERROR, logger="api.memory.preflight"):
            with pytest.raises(EmbedderUnavailable):
                asyncio.run(preflight_embedder("p1"))
        line = " ".join(r.getMessage() for r in caplog.records)
        assert "reason=auth" in line

    def test_probe_timeout_from_registry(self, monkeypatch):
        """Таймаут пробы берётся из реестра (provider_test), не из воздуха."""
        import api.config.timeout as timeout_mod

        monkeypatch.setattr(timeout_mod, "resolve_provider_test_timeout", lambda: 3.0)
        assert pf._resolve_probe_timeout() == 3.0

    def test_probe_timeout_fallback(self, monkeypatch):
        import api.config.timeout as timeout_mod

        def _boom():
            raise RuntimeError("registry down")

        monkeypatch.setattr(timeout_mod, "resolve_provider_test_timeout", _boom)
        assert pf._resolve_probe_timeout() == 15.0


# ============================================================================
# Сверка размерности
# ============================================================================
class TestDimensionCheck:
    def test_mismatch_raises(self, monkeypatch):
        _fake_embedder_module(monkeypatch, vectors=[[0.0] * 512])
        monkeypatch.setattr(pf, "_existing_chunk_dimension", lambda pid: 768)
        with pytest.raises(EmbedderUnavailable) as e:
            asyncio.run(preflight_embedder("p1"))
        assert e.value.reason == "dimension_mismatch"
        assert "512" in str(e.value) and "768" in str(e.value)

    def test_match_passes(self, monkeypatch):
        _fake_embedder_module(monkeypatch, vectors=[[0.0] * 768])
        monkeypatch.setattr(pf, "_existing_chunk_dimension", lambda pid: 768)
        assert asyncio.run(preflight_embedder("p1")) == 768

    def test_no_existing_chunks_skips_check(self, monkeypatch):
        _fake_embedder_module(monkeypatch, vectors=[[0.0] * 384])
        monkeypatch.setattr(pf, "_existing_chunk_dimension", lambda pid: None)
        assert asyncio.run(preflight_embedder("p1")) == 384

    def test_dim_query_failure_is_non_fatal(self, monkeypatch):
        """Сбой запроса размерности — не причина валить джоб: сам запрос глотает
        ошибки и возвращает None, сверка пропускается."""
        _fake_embedder_module(monkeypatch, vectors=[[0.0] * 8])
        monkeypatch.setattr(pf, "_existing_chunk_dimension", lambda pid: None)
        assert asyncio.run(preflight_embedder("p1")) == 8

    def test_dim_query_swallows_db_errors(self, monkeypatch):
        """_existing_chunk_dimension не бросает даже при мёртвой БД."""
        import api.db as db_mod

        class _BrokenSession:
            def __enter__(self):
                raise RuntimeError("db down")

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(db_mod, "SessionLocal", lambda: _BrokenSession())
        assert pf._existing_chunk_dimension("p1") is None


# ============================================================================
# Пропуск на не-pgvector инсталляции
# ============================================================================
class TestSkipWhenNotPgvector:
    def test_skipped_without_embedder_call(self, monkeypatch):
        monkeypatch.setattr(pf, "_pgvector_ready", lambda: False)

        def _boom(*a, **kw):
            raise AssertionError("пропуск не должен звать эмбеддер")

        import api.tools.embedder as embedder_mod

        monkeypatch.setattr(embedder_mod, "get_embedder", _boom)
        assert asyncio.run(preflight_embedder("p1")) is None


# ============================================================================
# Интеграция с джоб-воркером
# ============================================================================
class TestWorkerIntegration:
    def _seed(self, db_mod):
        from api.models import CodebaseORM, ProductORM

        with db_mod.SessionLocal() as s:
            s.add(ProductORM(id="prod_pf", name="P"))
            s.flush()
            s.add(CodebaseORM(
                id="cb_pf", product_id="prod_pf", name="repo",
                repo_url="https://github.com/o/r",
            ))
            s.commit()

    def test_preflight_failure_fails_job_with_prefix(self, isolated_db, monkeypatch):
        """Отказ предполёта: джоб падает с EMBEDDER_ERROR: в error, генерация
        не стартует вовсе."""
        import api.docgen.jobs as jobs_mod
        import api.memory.preflight as preflight_mod

        self._seed(isolated_db)
        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        async def _refuse(pid):
            raise EmbedderUnavailable(
                "Эмбеддер недоступен: connection refused.",
                reason="unreachable",
                detail="connection refused",
            )

        monkeypatch.setattr(preflight_mod, "preflight_embedder", _refuse)

        called = {"v": False}

        async def fake_generate(artifact, product, model=None, language="ru", progress=None):
            called["v"] = True
            return "docs"

        import api.docgen.codebase as codebase_mod

        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        j1 = jobs_mod.create_job("prod_pf", "codebase", "cb_pf")
        jobs_mod._run_docgen_job(j1, "prod_pf", "codebase", "cb_pf", None, "ru")

        job = jobs_mod.get_job(j1)
        assert job["status"] == "failed"
        assert job["error"].startswith(EMBEDDER_ERROR_PREFIX)
        assert "connection refused" in job["error"]
        assert called["v"] is False  # генерация не запускалась

    def test_preflight_skip_keeps_job_alive(self, isolated_db, monkeypatch):
        """Не-pgvector инсталляция: предполёт молча пропускается, джоб идёт
        дальше как раньше (герметичные тесты не должны ломаться)."""
        import api.docgen.jobs as jobs_mod
        import api.memory.preflight as preflight_mod

        self._seed(isolated_db)
        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        async def _skip(pid):
            return None

        monkeypatch.setattr(preflight_mod, "preflight_embedder", _skip)

        async def fake_generate(artifact, product, model=None, language="ru", progress=None):
            return "docs"

        import api.docgen.codebase as codebase_mod

        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        j1 = jobs_mod.create_job("prod_pf", "codebase", "cb_pf")
        jobs_mod._run_docgen_job(j1, "prod_pf", "codebase", "cb_pf", None, "ru")

        job = jobs_mod.get_job(j1)
        assert job["status"] == "succeeded"
        assert job["docs_chars"] == len("docs")

    def test_preflight_internal_error_is_non_fatal(self, isolated_db, monkeypatch):
        """Внутренний сбой самого предполёта (не диагноз) не должен валить джоб."""
        import api.docgen.jobs as jobs_mod
        import api.memory.preflight as preflight_mod

        self._seed(isolated_db)
        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        async def _buggy(pid):
            raise RuntimeError("preflight plumbing bug")

        monkeypatch.setattr(preflight_mod, "preflight_embedder", _buggy)

        async def fake_generate(artifact, product, model=None, language="ru", progress=None):
            return "docs"

        import api.docgen.codebase as codebase_mod

        monkeypatch.setattr(codebase_mod, "generate_codebase_docs", fake_generate)

        j1 = jobs_mod.create_job("prod_pf", "codebase", "cb_pf")
        jobs_mod._run_docgen_job(j1, "prod_pf", "codebase", "cb_pf", None, "ru")

        job = jobs_mod.get_job(j1)
        assert job["status"] == "succeeded"
