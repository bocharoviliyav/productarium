"""Unit tests for ``api.cognee.query``.

Covers:
- ``_recall_response_to_text``: extracts text from cognee V1 RecallResponse
  variants (``.text``, ``.answer``, ``.content``), plain-string passthrough,
  empty/blank fallback to ``""``.
- ``query_cognee``: returns ``""`` when cognee unavailable; happy path with
  fake_cognee.recall returning results; empty results -> ``""``; recall
  exception -> ``""`` (never raises); SearchType import path.
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.cognee.query import _recall_response_to_text, query_cognee


# --------------------------------------------------------------------------- #
# _recall_response_to_text
# --------------------------------------------------------------------------- #
class TestRecallResponseToText:
    def test_plain_string_passthrough(self):
        assert _recall_response_to_text("hello world") == "hello world"

    def test_graph_entry_text_attr(self):
        obj = SimpleNamespace(text="graph text", answer=None, content=None)
        assert _recall_response_to_text(obj) == "graph text"

    def test_qa_entry_answer_attr(self):
        obj = SimpleNamespace(text=None, answer="qa answer", content=None)
        assert _recall_response_to_text(obj) == "qa answer"

    def test_context_entry_content_attr(self):
        obj = SimpleNamespace(text=None, answer=None, content="context body")
        assert _recall_response_to_text(obj) == "context body"

    def test_text_takes_precedence_over_answer(self):
        obj = SimpleNamespace(text="from text", answer="from answer", content=None)
        assert _recall_response_to_text(obj) == "from text"

    def test_blank_string_text_falls_through_to_answer(self):
        obj = SimpleNamespace(text="   ", answer="from answer", content=None)
        assert _recall_response_to_text(obj) == "from answer"

    def test_empty_string_text_falls_through(self):
        obj = SimpleNamespace(text="", answer="from answer", content=None)
        assert _recall_response_to_text(obj) == "from answer"

    def test_all_blank_attrs_falls_back_to_str(self):
        obj = SimpleNamespace(text="", answer="", content="")
        # str(obj) is non-empty, so it's returned
        result = _recall_response_to_text(obj)
        assert result == str(obj)

    def test_object_with_no_attrs_falls_back_to_str(self):
        obj = SimpleNamespace()
        result = _recall_response_to_text(obj)
        assert result == str(obj)

    def test_object_with_non_string_attrs(self):
        obj = SimpleNamespace(text=123, answer=None, content=None)
        # 123 is not a str, so text is skipped -> falls to str(obj)
        result = _recall_response_to_text(obj)
        assert result == str(obj)


# --------------------------------------------------------------------------- #
# query_cognee: cognee unavailable
# --------------------------------------------------------------------------- #
class TestQueryCogneeUnavailable:
    def test_returns_empty_when_cognee_unavailable(self, monkeypatch):
        """When _COGNEE_AVAILABLE is False, query_cognee returns ''."""
        import api.cognee.query as qmod

        monkeypatch.setattr(qmod, "_COGNEE_AVAILABLE", False)
        result = asyncio.run(query_cognee("query", "prod_123"))
        assert result == ""


# --------------------------------------------------------------------------- #
# query_cognee: happy path with fake_cognee
# --------------------------------------------------------------------------- #
class TestQueryCogneeHappyPath:
    def test_returns_joined_text_from_results(self, monkeypatch, fake_cognee):
        """With fake_cognee injected, query returns joined recall text."""
        import api.cognee.query as qmod
        import api.cognee._runtime as rtmod

        # Force availability + rebind the cognee reference used by query.py.
        monkeypatch.setattr(qmod, "_COGNEE_AVAILABLE", True)

        # fake_cognee has a recall-like method? No — it has search/add/cognify.
        # query_cognee calls cognee.recall(...). We need to add recall to the fake.
        recall_results = [
            SimpleNamespace(text="result one", answer=None, content=None),
            SimpleNamespace(text=None, answer="result two", content=None),
        ]

        async def _recall(query_text=None, query_type=None, datasets=None, top_k=20):
            return recall_results

        fake_cognee.recall = _recall

        # Patch the cognee reference in query.py to point at the fake.
        monkeypatch.setattr(qmod, "cognee", fake_cognee)

        # Patch apply_cognee_runtime_config so it doesn't try settings DB.
        monkeypatch.setattr(qmod, "apply_cognee_runtime_config", lambda: None)

        # Provide a fake SearchType for the `from cognee import SearchType` import.
        import types

        fake_search_type = types.SimpleNamespace(GRAPH_COMPLETION="graph_completion")
        # The import `from cognee import SearchType` reads fake_cognee.SearchType.
        fake_cognee.SearchType = fake_search_type

        result = asyncio.run(query_cognee("my query", "prod_123", top_k=20))
        assert "result one" in result
        assert "result two" in result
        assert "\n\n" in result

    def test_empty_results_returns_empty_string(self, monkeypatch, fake_cognee):
        import api.cognee.query as qmod

        monkeypatch.setattr(qmod, "_COGNEE_AVAILABLE", True)

        async def _recall(query_text=None, query_type=None, datasets=None, top_k=20):
            return []

        fake_cognee.recall = _recall
        monkeypatch.setattr(qmod, "cognee", fake_cognee)
        monkeypatch.setattr(qmod, "apply_cognee_runtime_config", lambda: None)

        import types

        fake_cognee.SearchType = types.SimpleNamespace(GRAPH_COMPLETION="graph_completion")

        result = asyncio.run(query_cognee("query", "prod_456"))
        assert result == ""

    def test_none_results_returns_empty_string(self, monkeypatch, fake_cognee):
        import api.cognee.query as qmod

        monkeypatch.setattr(qmod, "_COGNEE_AVAILABLE", True)

        async def _recall(query_text=None, query_type=None, datasets=None, top_k=20):
            return None

        fake_cognee.recall = _recall
        monkeypatch.setattr(qmod, "cognee", fake_cognee)
        monkeypatch.setattr(qmod, "apply_cognee_runtime_config", lambda: None)

        import types

        fake_cognee.SearchType = types.SimpleNamespace(GRAPH_COMPLETION="graph_completion")

        result = asyncio.run(query_cognee("query", "prod_789"))
        assert result == ""


# --------------------------------------------------------------------------- #
# query_cognee: error / timeout -> ""
# --------------------------------------------------------------------------- #
class TestQueryCogneeErrors:
    def test_recall_exception_returns_empty(self, monkeypatch, fake_cognee):
        """A recall() exception is caught and '' returned (never raises)."""
        import api.cognee.query as qmod

        monkeypatch.setattr(qmod, "_COGNEE_AVAILABLE", True)

        async def _recall(query_text=None, query_type=None, datasets=None, top_k=20):
            raise RuntimeError("cognee blew up")

        fake_cognee.recall = _recall
        monkeypatch.setattr(qmod, "cognee", fake_cognee)
        monkeypatch.setattr(qmod, "apply_cognee_runtime_config", lambda: None)

        import types

        fake_cognee.SearchType = types.SimpleNamespace(GRAPH_COMPLETION="graph_completion")

        result = asyncio.run(query_cognee("query", "prod_err"))
        assert result == ""

    def test_recall_timeout_returns_empty(self, monkeypatch, fake_cognee):
        """An asyncio.TimeoutError from recall -> '' (never raises)."""
        import api.cognee.query as qmod

        monkeypatch.setattr(qmod, "_COGNEE_AVAILABLE", True)

        async def _recall(query_text=None, query_type=None, datasets=None, top_k=20):
            raise asyncio.TimeoutError()

        fake_cognee.recall = _recall
        monkeypatch.setattr(qmod, "cognee", fake_cognee)
        monkeypatch.setattr(qmod, "apply_cognee_runtime_config", lambda: None)

        import types

        fake_cognee.SearchType = types.SimpleNamespace(GRAPH_COMPLETION="graph_completion")

        result = asyncio.run(query_cognee("query", "prod_timeout"))
        assert result == ""

    def test_recall_results_with_blank_pieces_filtered(self, monkeypatch, fake_cognee):
        """Blank text pieces are filtered out before joining."""
        import api.cognee.query as qmod

        monkeypatch.setattr(qmod, "_COGNEE_AVAILABLE", True)

        recall_results = [
            SimpleNamespace(text="real text", answer=None, content=None),
            SimpleNamespace(text="", answer="", content=""),  # all blank -> str(obj)
            SimpleNamespace(text="more text", answer=None, content=None),
        ]

        async def _recall(query_text=None, query_type=None, datasets=None, top_k=20):
            return recall_results

        fake_cognee.recall = _recall
        monkeypatch.setattr(qmod, "cognee", fake_cognee)
        monkeypatch.setattr(qmod, "apply_cognee_runtime_config", lambda: None)

        import types

        fake_cognee.SearchType = types.SimpleNamespace(GRAPH_COMPLETION="graph_completion")

        result = asyncio.run(query_cognee("query", "prod_filter"))
        # The all-blank obj falls back to str(obj) which is non-empty, so it's
        # included. The two real texts are present.
        assert "real text" in result
        assert "more text" in result

    def test_searchtype_import_failure_returns_empty(self, monkeypatch, fake_cognee):
        """If `from cognee import SearchType` fails, query returns ''."""
        import api.cognee.query as qmod

        monkeypatch.setattr(qmod, "_COGNEE_AVAILABLE", True)

        # Remove SearchType from fake_cognee so the import fails.
        if hasattr(fake_cognee, "SearchType"):
            del fake_cognee.SearchType

        # Also ensure the real import path doesn't resolve.
        # The `from cognee import SearchType` will raise AttributeError.
        monkeypatch.setattr(qmod, "cognee", fake_cognee)
        monkeypatch.setattr(qmod, "apply_cognee_runtime_config", lambda: None)

        result = asyncio.run(query_cognee("query", "prod_no_searchtype"))
        assert result == ""
