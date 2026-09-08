"""Unit tests for ``api.expert.generate``.

Covers (post fast-rlm removal — the expert answer is a plain langchain
``ChatOpenAI`` call via ``_ExpertLLM``):
- ``_generate_answer``: LLM success (text cleaned via ``_clean_llm_text``),
  LLM returns "" , LLM unavailable (None) -> "", generate exception -> "".
- ``_stream_answer``: LLM stream passthrough of typed events, LLM
  unavailable (None) -> no events.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from api.expert.generate import (
    _generate_answer,
    _stream_answer,
)
from api.expert.types import EVENT_CONTENT, ExpertStreamEvent


# ---------------------------------------------------------------------------
# Patching strategy
# ---------------------------------------------------------------------------
# ``_generate_answer`` / ``_stream_answer`` read ``_safe_build_llm`` as a
# top-level module global, so patching the function's own ``__globals__``
# dict is the exact dict the function body reads. This stays robust even
# under coverage's import hooks (which can create a second module object).
_gen_globals = _generate_answer.__globals__
_stream_globals = _stream_answer.__globals__


class _FakeLLM:
    """Fake _ExpertLLM: async generate + async stream of typed events."""

    def __init__(self, generate_text="LLM answer", events=None, exc=None):
        self._generate_text = generate_text
        self._events = events or []
        self._exc = exc

    async def generate(self, prompt):
        if self._exc:
            raise self._exc
        return self._generate_text

    async def stream(self, prompt):
        if self._exc:
            raise self._exc
        for ev in self._events:
            yield ev


# ---------------------------------------------------------------------------
# _generate_answer
# ---------------------------------------------------------------------------
class TestGenerateAnswer:
    def test_llm_returns_text(self, monkeypatch):
        monkeypatch.setitem(
            _gen_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: _FakeLLM("LLM answer"),
        )

        result = asyncio.run(_generate_answer("prompt", "model", None, None))
        assert result == "LLM answer"

    def test_llm_text_cleaned(self, monkeypatch):
        # The raw generation is wrapped in a markdown fence + surrounding
        # whitespace; _clean_llm_text must strip it before returning.
        monkeypatch.setitem(
            _gen_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: _FakeLLM("```\nLLM answer\n```"),
        )

        result = asyncio.run(_generate_answer("prompt", "model", None, None))
        assert result == "LLM answer"

    def test_llm_empty_returns_empty(self, monkeypatch):
        monkeypatch.setitem(
            _gen_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: _FakeLLM(""),
        )

        result = asyncio.run(_generate_answer("prompt", "model", None, None))
        assert result == ""

    def test_llm_none_returns_empty(self, monkeypatch):
        monkeypatch.setitem(
            _gen_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: None,
        )

        result = asyncio.run(_generate_answer("prompt", "model", None, None))
        assert result == ""

    def test_llm_generate_exception_returns_empty(self, monkeypatch):
        # The defensive except in _generate_answer catches a mid-call failure
        # (an LLM that raised despite being built) and returns "".
        monkeypatch.setitem(
            _gen_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: _FakeLLM(
                exc=RuntimeError("llm crashed")
            ),
        )

        result = asyncio.run(_generate_answer("prompt", "model", None, None))
        assert result == ""

    def test_forwarding_optional_context_kwargs(self, monkeypatch):
        # The context kwargs (product_id/name/query/history) are accepted and
        # forwarded nowhere for now — they exist for the LangGraph agent that
        # lands with Wave B. They must not break the call.
        captured = {}

        class _CapturingLLM(_FakeLLM):
            async def generate(self, prompt):
                captured["prompt"] = prompt
                return "ok"

        monkeypatch.setitem(
            _gen_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: _CapturingLLM(),
        )

        result = asyncio.run(
            _generate_answer(
                "prompt", "model", None, None,
                product_id="prod_1", product_name="P", query="q", history="h",
            )
        )
        assert result == "ok"
        assert captured["prompt"] == "prompt"


# ---------------------------------------------------------------------------
# _stream_answer
# ---------------------------------------------------------------------------
class TestStreamAnswer:
    def test_llm_stream_passthrough(self, monkeypatch):
        events = [
            ExpertStreamEvent(EVENT_CONTENT, "chunk1"),
            ExpertStreamEvent(EVENT_CONTENT, "chunk2"),
        ]
        monkeypatch.setitem(
            _stream_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: _FakeLLM(events=events),
        )

        got = []

        async def _collect():
            async for ev in _stream_answer("prompt", "model", None, None):
                got.append(ev)

        asyncio.run(_collect())
        assert [e.content for e in got] == ["chunk1", "chunk2"]
        assert all(e.type == EVENT_CONTENT for e in got)

    def test_llm_none_yields_nothing(self, monkeypatch):
        monkeypatch.setitem(
            _stream_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: None,
        )

        got = []

        async def _collect():
            async for ev in _stream_answer("prompt", "model", None, None):
                got.append(ev)

        asyncio.run(_collect())
        assert got == []

    def test_empty_event_stream_yields_nothing(self, monkeypatch):
        monkeypatch.setitem(
            _stream_globals, "_safe_build_llm",
            lambda m, base_url=None, api_key=None: _FakeLLM(events=[]),
        )

        got = []

        async def _collect():
            async for ev in _stream_answer("prompt", "model", None, None):
                got.append(ev)

        asyncio.run(_collect())
        assert got == []
