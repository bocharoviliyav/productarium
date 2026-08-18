"""Unit tests for ``api.memory.cognee_backend``.

Covers the thin adapter over ``api.cognee``:
- ``index``: delegates to ``add_and_index_document(content, dataset_name=...)``
  with the ``prod_{product_id}`` dataset convention; returns 1 on success / 0
  on failure; 0 on empty content / product_id.
- ``query``: delegates to ``query_cognee(query, dataset_name=..., top_k=...)``;
  returns "" on failure / empty input.
- ``clear_product``: delegates to ``_empty_cognee_dataset(dataset)``.
- ``reindex_product``: delegates to ``reindex_product_knowledge_graph``.
- ``init``: delegates to ``init_cognee`` (non-fatal).
- ``status``: reports ``_COGNEE_AVAILABLE``.

Each method imports ``api.cognee`` lazily (``from api.cognee import X``), so
tests patch the attributes on the real ``api.cognee`` module via ``monkeypatch``.
No real cognee dependency is required.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
class TestIndex:
    def test_empty_content_returns_zero(self, isolated_db):
        from api.memory.cognee_backend import CogneeMemoryBackend

        be = CogneeMemoryBackend()
        assert asyncio.run(be.index("", "prod_1")) == 0
        assert asyncio.run(be.index("   ", "prod_1")) == 0
        assert asyncio.run(be.index("content", "")) == 0

    def test_delegates_to_add_and_index_document(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        captured: dict = {}

        async def _fake_add(content, dataset_name=None):
            captured["content"] = content
            captured["dataset_name"] = dataset_name
            return None

        monkeypatch.setattr(cognee_mod, "add_and_index_document", _fake_add)

        be = CogneeMemoryBackend()
        result = asyncio.run(be.index("some docs", "prod_1", source_type="codebase", source_id="cb_1"))
        assert result == 1
        assert captured["content"] == "some docs"
        assert captured["dataset_name"] == "prod_prod_1"

    def test_failure_returns_zero(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        async def _boom(content, dataset_name=None):
            raise RuntimeError("cognee down")

        monkeypatch.setattr(cognee_mod, "add_and_index_document", _boom)

        be = CogneeMemoryBackend()
        assert asyncio.run(be.index("some docs", "prod_1")) == 0


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #
class TestQuery:
    def test_empty_input_returns_empty(self, isolated_db):
        from api.memory.cognee_backend import CogneeMemoryBackend

        be = CogneeMemoryBackend()
        assert asyncio.run(be.query("", "prod_1")) == ""
        assert asyncio.run(be.query("q", "")) == ""

    def test_delegates_to_query_cognee(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        captured: dict = {}

        async def _fake_query(query, dataset_name=None, top_k=20):
            captured["query"] = query
            captured["dataset_name"] = dataset_name
            captured["top_k"] = top_k
            return "recalled context"

        monkeypatch.setattr(cognee_mod, "query_cognee", _fake_query)

        be = CogneeMemoryBackend()
        result = asyncio.run(be.query("how does auth work", "prod_1", top_k=15))
        assert result == "recalled context"
        assert captured["query"] == "how does auth work"
        assert captured["dataset_name"] == "prod_prod_1"
        assert captured["top_k"] == 15

    def test_failure_returns_empty(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        async def _boom(query, dataset_name=None, top_k=20):
            raise RuntimeError("cognee down")

        monkeypatch.setattr(cognee_mod, "query_cognee", _boom)

        be = CogneeMemoryBackend()
        assert asyncio.run(be.query("q", "prod_1")) == ""


# --------------------------------------------------------------------------- #
# clear_product
# --------------------------------------------------------------------------- #
class TestClearProduct:
    def test_empty_product_id_returns_false(self, isolated_db):
        from api.memory.cognee_backend import CogneeMemoryBackend

        be = CogneeMemoryBackend()
        assert asyncio.run(be.clear_product("")) is False

    def test_delegates_to_empty_cognee_dataset(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        captured: dict = {}

        async def _fake_empty(dataset):
            captured["dataset"] = dataset
            return True

        monkeypatch.setattr(cognee_mod, "_empty_cognee_dataset", _fake_empty)

        be = CogneeMemoryBackend()
        assert asyncio.run(be.clear_product("prod_1")) is True
        assert captured["dataset"] == "prod_prod_1"

    def test_failure_returns_false(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        async def _boom(dataset):
            raise RuntimeError("cognee down")

        monkeypatch.setattr(cognee_mod, "_empty_cognee_dataset", _boom)

        be = CogneeMemoryBackend()
        assert asyncio.run(be.clear_product("prod_1")) is False


# --------------------------------------------------------------------------- #
# reindex_product
# --------------------------------------------------------------------------- #
class TestReindexProduct:
    def test_delegates_to_reindex_product_knowledge_graph(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        captured: dict = {}

        async def _fake_reindex(pid):
            captured["pid"] = pid
            return {"success": True, "message": "ok", "reindexed_count": 3}

        monkeypatch.setattr(cognee_mod, "reindex_product_knowledge_graph", _fake_reindex)

        be = CogneeMemoryBackend()
        result = asyncio.run(be.reindex_product("prod_1"))
        assert result["success"] is True
        assert result["reindexed_count"] == 3
        assert captured["pid"] == "prod_1"

    def test_none_pid_passed_through(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        captured: dict = {}

        async def _fake_reindex(pid):
            captured["pid"] = pid
            return {"success": True, "reindexed_count": 0}

        monkeypatch.setattr(cognee_mod, "reindex_product_knowledge_graph", _fake_reindex)

        be = CogneeMemoryBackend()
        asyncio.run(be.reindex_product(None))
        assert captured["pid"] is None

    def test_failure_returns_error_dict(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        async def _boom(pid):
            raise RuntimeError("cognee down")

        monkeypatch.setattr(cognee_mod, "reindex_product_knowledge_graph", _boom)

        be = CogneeMemoryBackend()
        result = asyncio.run(be.reindex_product("prod_1"))
        assert result["success"] is False
        assert result["reindexed_count"] == 0


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #
class TestInit:
    def test_delegates_to_init_cognee(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        called: list = []

        async def _fake_init():
            called.append(True)

        monkeypatch.setattr(cognee_mod, "init_cognee", _fake_init)

        be = CogneeMemoryBackend()
        asyncio.run(be.init())
        assert called == [True]

    def test_init_failure_is_non_fatal(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        async def _boom():
            raise RuntimeError("cognee init failed")

        monkeypatch.setattr(cognee_mod, "init_cognee", _boom)

        be = CogneeMemoryBackend()
        # Must not raise.
        asyncio.run(be.init())


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
class TestStatus:
    def test_reports_available_true(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        monkeypatch.setattr(cognee_mod, "_COGNEE_AVAILABLE", True)

        be = CogneeMemoryBackend()
        status = be.status()
        assert status["backend"] == "cognee"
        assert status["available"] is True

    def test_reports_available_false(self, monkeypatch, isolated_db):
        import api.cognee as cognee_mod
        from api.memory.cognee_backend import CogneeMemoryBackend

        monkeypatch.setattr(cognee_mod, "_COGNEE_AVAILABLE", False)

        be = CogneeMemoryBackend()
        status = be.status()
        assert status["backend"] == "cognee"
        assert status["available"] is False
