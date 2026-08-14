"""Unit tests for ``api.expert.knowledge``.

Covers:
- ``_retrieve_product_knowledge``: cognee recall mocked + fallback artifact docs
  + confluence fallback + all-empty.
- ``_fallback_artifact_docs``: codebase generated_docs, codebase pages (dict +
  str), spec content, missing product, empty.
- ``_product_name_by_id``: found + missing + DB-down.
- ``_format_history``: empty + turns (user/assistant/other) + skipped invalid.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.expert.knowledge import (
    _fallback_artifact_docs,
    _format_history,
    _product_name_by_id,
    _retrieve_product_knowledge,
)


def _install_fake_cognee(monkeypatch, query_cognee_fn):
    """Inject a fake api.cognee module with a mock query_cognee.

    Under --cov, importing the real api.cognee triggers the api.config ->
    adalflow -> numpy chain which corrupts numpy's C extension. This fake
    avoids it.
    """
    fake_cognee = types.ModuleType("api.cognee")
    fake_cognee.query_cognee = query_cognee_fn
    monkeypatch.setitem(sys.modules, "api.cognee", fake_cognee)


# --------------------------------------------------------------------------- #
# _format_history
# --------------------------------------------------------------------------- #
class TestFormatHistory:
    def test_empty_returns_empty(self):
        assert _format_history([]) == ""

    def test_none_returns_empty(self):
        assert _format_history(None) == ""

    def test_user_turn(self):
        result = _format_history([{"role": "user", "content": "hello"}])
        assert "<user>hello</user>" in result

    def test_assistant_turn(self):
        result = _format_history([{"role": "assistant", "content": "hi there"}])
        assert "<assistant>hi there</assistant>" in result

    def test_other_role(self):
        result = _format_history([{"role": "system", "content": "sys msg"}])
        assert "<system>sys msg</system>" in result

    def test_multiple_turns(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        result = _format_history(msgs)
        assert "<user>q1</user>" in result
        assert "<assistant>a1</assistant>" in result
        assert "<user>q2</user>" in result

    def test_skips_missing_role(self):
        result = _format_history([{"content": "no role"}])
        assert result == ""

    def test_skips_missing_content(self):
        result = _format_history([{"role": "user"}])
        assert result == ""

    def test_skips_empty_content(self):
        result = _format_history([{"role": "user", "content": ""}])
        assert result == ""

    def test_skips_empty_role(self):
        result = _format_history([{"role": "", "content": "x"}])
        assert result == ""

    def test_role_case_insensitive(self):
        result = _format_history([{"role": "USER", "content": "x"}])
        assert "<user>x</user>" in result

    def test_content_stripped(self):
        result = _format_history([{"role": "user", "content": "  spaced  "}])
        assert "<user>spaced</user>" in result


# --------------------------------------------------------------------------- #
# _product_name_by_id
# --------------------------------------------------------------------------- #
class TestProductNameById:
    def test_found_returns_name(self, isolated_db):
        from api.models import ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="MyService", description=""))
            db.commit()
        finally:
            db.close()

        assert _product_name_by_id("prod_1") == "MyService"

    def test_missing_returns_id(self, isolated_db):
        assert _product_name_by_id("prod_nonexistent") == "prod_nonexistent"

    def test_db_down_returns_id(self, monkeypatch):
        # Force SessionLocal to raise.
        import api.db as db_mod

        class _BoomSession:
            def __enter__(self):
                raise RuntimeError("db down")

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "SessionLocal", lambda: _BoomSession())
        assert _product_name_by_id("prod_x") == "prod_x"


# --------------------------------------------------------------------------- #
# _fallback_artifact_docs
# --------------------------------------------------------------------------- #
class TestFallbackArtifactDocs:
    def test_missing_product_returns_empty(self, isolated_db):
        assert _fallback_artifact_docs("prod_missing") == ""

    def test_codebase_generated_docs(self, isolated_db):
        from api.models import CodebaseORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo-a",
                generated_docs="Generated wiki content",
            ))
            db.commit()
        finally:
            db.close()

        result = _fallback_artifact_docs("prod_1")
        assert "## Codebase: repo-a" in result
        assert "Generated wiki content" in result

    def test_codebase_pages_dict(self, isolated_db):
        from api.models import CodebaseORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo-a",
                generated_docs="",
                pages={"page1": {"content": "Page 1 content"}},
            ))
            db.commit()
        finally:
            db.close()

        result = _fallback_artifact_docs("prod_1")
        assert "## Codebase: repo-a / page page1" in result
        assert "Page 1 content" in result

    def test_codebase_pages_string(self, isolated_db):
        from api.models import CodebaseORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo-a",
                generated_docs="",
                pages={"page1": "raw string content"},
            ))
            db.commit()
        finally:
            db.close()

        result = _fallback_artifact_docs("prod_1")
        assert "raw string content" in result

    def test_spec_content(self, isolated_db):
        from api.models import ProductORM, SpecORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(SpecORM(
                id="spec_1", product_id="prod_1", name="openapi.yaml",
                kind="openapi", content="openapi: 3.0.0",
            ))
            db.commit()
        finally:
            db.close()

        result = _fallback_artifact_docs("prod_1")
        assert "## Spec: openapi.yaml" in result
        assert "openapi: 3.0.0" in result

    def test_empty_codebase_and_spec(self, isolated_db):
        from api.models import CodebaseORM, ProductORM, SpecORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo-a",
                generated_docs="", pages={},
            ))
            db.add(SpecORM(
                id="spec_1", product_id="prod_1", name="spec.yaml",
                kind="openapi", content="",
            ))
            db.commit()
        finally:
            db.close()

        assert _fallback_artifact_docs("prod_1") == ""

    def test_codebase_uses_id_when_name_missing(self, isolated_db):
        from api.models import CodebaseORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_99", product_id="prod_1", name="",
                generated_docs="docs content",
            ))
            db.commit()
        finally:
            db.close()

        result = _fallback_artifact_docs("prod_1")
        # When name is empty, the id is used as fallback.
        assert "cb_99" in result
        assert "docs content" in result

    def test_db_down_returns_empty(self, monkeypatch):
        import api.db as db_mod

        class _BoomSession:
            def __enter__(self):
                raise RuntimeError("db down")

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "SessionLocal", lambda: _BoomSession())
        assert _fallback_artifact_docs("prod_x") == ""


# --------------------------------------------------------------------------- #
# _retrieve_product_knowledge
# --------------------------------------------------------------------------- #
class TestRetrieveProductKnowledge:
    def test_cognee_recall_returns_context(self, monkeypatch):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            assert dataset_name == "prod_prod_1"
            return "cognee recalled knowledge"

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        result = asyncio.run(_retrieve_product_knowledge("prod_1", "query"))
        assert result == "cognee recalled knowledge"

    def test_cognee_recall_empty_falls_back_to_artifacts(self, monkeypatch, isolated_db):
        from api.models import CodebaseORM, ProductORM

        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo",
                generated_docs="fallback docs",
            ))
            db.commit()
        finally:
            db.close()

        result = asyncio.run(_retrieve_product_knowledge("prod_1", "query"))
        assert "fallback docs" in result

    def test_cognee_recall_exception_falls_back_to_artifacts(self, monkeypatch, isolated_db):
        from api.models import CodebaseORM, ProductORM

        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            raise RuntimeError("cognee crashed")

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo",
                generated_docs="fallback docs",
            ))
            db.commit()
        finally:
            db.close()

        result = asyncio.run(_retrieve_product_knowledge("prod_1", "query"))
        assert "fallback docs" in result

    def test_cognee_empty_and_no_artifacts_returns_empty(self, monkeypatch, isolated_db):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        # No product -> no artifacts -> empty.
        result = asyncio.run(_retrieve_product_knowledge("prod_missing", "query"))
        assert result == ""

    def test_confluence_fallback_when_cognee_and_artifacts_empty(self, monkeypatch, isolated_db):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        # Mock the Confluence connector path.
        fake_connector = SimpleNamespace()
        fake_connector.is_configured = lambda: True
        fake_connector.list_spaces = lambda: [{"key": "DEV"}]
        fake_connector.pull = lambda sp_id, opts=None: {"markdown": "confluence content"}

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: fake_connector if name == "confluence" else None)

        result = asyncio.run(_retrieve_product_knowledge("prod_missing", "query"))
        assert result == "confluence content"

    def test_confluence_fallback_not_configured(self, monkeypatch, isolated_db):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        fake_connector = SimpleNamespace()
        fake_connector.is_configured = lambda: False

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: fake_connector if name == "confluence" else None)

        result = asyncio.run(_retrieve_product_knowledge("prod_missing", "query"))
        assert result == ""

    def test_confluence_fallback_no_spaces(self, monkeypatch, isolated_db):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        fake_connector = SimpleNamespace()
        fake_connector.is_configured = lambda: True
        fake_connector.list_spaces = lambda: []

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: fake_connector if name == "confluence" else None)

        result = asyncio.run(_retrieve_product_knowledge("prod_missing", "query"))
        assert result == ""

    def test_confluence_fallback_no_markdown(self, monkeypatch, isolated_db):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        fake_connector = SimpleNamespace()
        fake_connector.is_configured = lambda: True
        fake_connector.list_spaces = lambda: [{"key": "DEV"}]
        fake_connector.pull = lambda sp_id, opts=None: {"markdown": ""}

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: fake_connector if name == "confluence" else None)

        result = asyncio.run(_retrieve_product_knowledge("prod_missing", "query"))
        assert result == ""

    def test_confluence_fallback_exception_returns_empty(self, monkeypatch, isolated_db):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        def _boom(name):
            raise RuntimeError("registry crashed")

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", _boom)

        result = asyncio.run(_retrieve_product_knowledge("prod_missing", "query"))
        assert result == ""

    def test_confluence_fallback_connector_none(self, monkeypatch, isolated_db):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: None)

        result = asyncio.run(_retrieve_product_knowledge("prod_missing", "query"))
        assert result == ""

    def test_confluence_fallback_space_id_from_id_key(self, monkeypatch, isolated_db):
        async def _fake_query_cognee(query, dataset_name=None, top_k=20):
            return ""

        _install_fake_cognee(monkeypatch, _fake_query_cognee)

        captured = {}

        class _FakeConnector:
            def is_configured(self):
                return True

            def list_spaces(self):
                return [{"id": "SPACE123"}]

            def pull(self, sp_id, opts=None):
                captured["sp_id"] = sp_id
                return {"markdown": "content from id-keyed space"}

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: _FakeConnector())

        result = asyncio.run(_retrieve_product_knowledge("prod_missing", "query"))
        assert result == "content from id-keyed space"
        assert captured["sp_id"] == "SPACE123"
