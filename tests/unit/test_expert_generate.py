"""Unit tests for ``api.expert.generate``.

Covers:
- ``_resolve_use_rlm``: True/False/None with admin modes auto/rlm/llm; auto
  threshold (RLM_MIN_CHARS).
- ``_rlm_generate``: monkeypatch run_rlm_task success dict / failure / timeout
  -> ''.
- ``_generate_answer``: RLM path + LLM fallback + both empty.
- ``_stream_answer``: RLM chunked + LLM stream passthrough + LLM None.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

# Under --cov with the default (prepend) import mode + tests/unit/__init__.py,
# coverage's import hooks can raise KeyError when importing new test modules
# after certain api.expert.* modules have been traced.  To avoid this, inject
# a fake api.config (prevents the adalflow→numpy corruption chain).
#
# The real api.config imports adalflow→numpy, and numpy's C extension corrupts
# under coverage's import hooks.  After corruption, any NEW module import
# through coverage's hooks raises KeyError, which breaks collection of
# test_expert_knowledge.py and test_expert_prompt.py.
#
# api.prompts._default_language() does ``from api.config import configs`` and
# reads configs.get("lang_config", {}).get("default", "ru").  The fake provides
# just that (plus get_model_config for api.expert.llm), so the real api.config
# is never loaded.
_fake_config = types.ModuleType("api.config")
_fake_config.configs = {"lang_config": {"default": "ru"}}
_fake_config.get_model_config = lambda model=None: {"model_client": type("_MC", (), {}), "model_kwargs": {"model": model or "test"}}
sys.modules.setdefault("api.config", _fake_config)

from api.expert.generate import (
    _generate_answer,
    _resolve_use_rlm,
    _rlm_generate,
    _stream_answer,
)
from api.expert.prompt import RLM_MIN_CHARS
from api.expert.types import EVENT_CONTENT, ExpertStreamEvent


def _install_fake_settings(monkeypatch, get_rlm_mode_fn):
    """Inject a fake api.config.settings module with a mock get_rlm_mode.

    Under --cov, importing the real api.config.settings triggers the
    api.config -> adalflow -> numpy chain which corrupts numpy's C extension.
    This fake avoids it.
    """
    fake_settings = types.ModuleType("api.config.settings")
    fake_settings.get_rlm_mode = get_rlm_mode_fn
    monkeypatch.setitem(sys.modules, "api.config.settings", fake_settings)


# --------------------------------------------------------------------------- #
# _resolve_use_rlm
# --------------------------------------------------------------------------- #
class TestResolveUseRlm:
    def test_explicit_true_always_rlm(self):
        assert _resolve_use_rlm(True, "expert", 100) is True

    def test_explicit_false_always_llm(self):
        assert _resolve_use_rlm(False, "expert", 100_000) is False

    def test_none_mode_llm_returns_false(self, monkeypatch):
        _install_fake_settings(monkeypatch, lambda task: "llm")
        assert _resolve_use_rlm(None, "expert", 100_000) is False

    def test_none_mode_rlm_returns_true(self, monkeypatch):
        _install_fake_settings(monkeypatch, lambda task: "rlm")
        assert _resolve_use_rlm(None, "expert", 100) is True

    def test_none_mode_auto_small_prompt_returns_false(self, monkeypatch):
        _install_fake_settings(monkeypatch, lambda task: "auto")
        assert _resolve_use_rlm(None, "expert", RLM_MIN_CHARS - 1) is False

    def test_none_mode_auto_large_prompt_returns_true(self, monkeypatch):
        _install_fake_settings(monkeypatch, lambda task: "auto")
        assert _resolve_use_rlm(None, "expert", RLM_MIN_CHARS) is True

    def test_none_mode_auto_exact_threshold(self, monkeypatch):
        _install_fake_settings(monkeypatch, lambda task: "auto")
        assert _resolve_use_rlm(None, "expert", RLM_MIN_CHARS + 1) is True

    def test_none_get_rlm_mode_exception_falls_back_to_auto(self, monkeypatch):
        def _boom(task):
            raise RuntimeError("db down")

        _install_fake_settings(monkeypatch, _boom)
        # Exception -> mode defaults to "auto" -> threshold check.
        assert _resolve_use_rlm(None, "expert", RLM_MIN_CHARS + 1) is True
        assert _resolve_use_rlm(None, "expert", RLM_MIN_CHARS - 1) is False

    def test_none_unknown_mode_falls_through_to_auto_threshold(self, monkeypatch):
        _install_fake_settings(monkeypatch, lambda task: "unknown_mode")
        # Unknown mode -> not "llm" and not "rlm" -> threshold check.
        assert _resolve_use_rlm(None, "expert", RLM_MIN_CHARS + 1) is True
        assert _resolve_use_rlm(None, "expert", RLM_MIN_CHARS - 1) is False


# --------------------------------------------------------------------------- #
# _rlm_generate
# --------------------------------------------------------------------------- #
class TestRlmGenerate:
    def test_success_returns_cleaned_text(self, monkeypatch):
        async def _fake_run_rlm_task(query, model):
            return {"success": True, "results": "  RLM answer  "}

        # Patch on the use-site module (generate.py does a lazy import inside
        # the function, so we patch api.rlm.runner.run_rlm_task).
        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)

        result = asyncio.run(_rlm_generate("prompt", "model"))
        assert result == "RLM answer"

    def test_success_with_markdown_fence_stripped(self, monkeypatch):
        async def _fake_run_rlm_task(query, model):
            return {"success": True, "results": "```\nRLM answer\n```"}

        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)

        result = asyncio.run(_rlm_generate("prompt", "model"))
        assert result == "RLM answer"

    def test_failure_returns_empty(self, monkeypatch):
        async def _fake_run_rlm_task(query, model):
            return {"success": False, "results": "error occurred"}

        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)

        result = asyncio.run(_rlm_generate("prompt", "model"))
        assert result == ""

    def test_no_results_returns_empty(self, monkeypatch):
        async def _fake_run_rlm_task(query, model):
            return {"success": True, "results": ""}

        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)

        result = asyncio.run(_rlm_generate("prompt", "model"))
        assert result == ""

    def test_no_success_key_returns_empty(self, monkeypatch):
        async def _fake_run_rlm_task(query, model):
            return {"results": "some text"}

        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)

        result = asyncio.run(_rlm_generate("prompt", "model"))
        assert result == ""

    def test_exception_returns_empty(self, monkeypatch):
        async def _fake_run_rlm_task(query, model):
            raise RuntimeError("rlm crashed")

        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)

        result = asyncio.run(_rlm_generate("prompt", "model"))
        assert result == ""

    def test_timeout_returns_empty(self, monkeypatch):
        async def _fake_run_rlm_task(query, model):
            raise asyncio.TimeoutError()

        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)

        result = asyncio.run(_rlm_generate("prompt", "model"))
        assert result == ""

    def test_large_prompt_truncated_for_rlm(self, monkeypatch):
        captured = {}

        async def _fake_run_rlm_task(query, model):
            captured["query"] = query
            return {"success": True, "results": "answer"}

        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)

        # Build a prompt large enough to trigger the char-cap inside _rlm_generate.
        # The cap is safe_prompt_limit * 4 where safe_prompt_limit depends on
        # the model context window. With a tiny context window the cap fires.
        import api.utils as utils_mod
        monkeypatch.setattr(utils_mod, "get_model_context_window", lambda **kw: 4096)

        big_prompt = "X" * 500_000
        result = asyncio.run(_rlm_generate(big_prompt, "model"))
        assert result == "answer"
        # The query passed to run_rlm_task should have been truncated.
        assert "контекст обрезан" in captured["query"]
        assert len(captured["query"]) < len(big_prompt)

    def test_normal_prompt_not_truncated(self, monkeypatch):
        captured = {}

        async def _fake_run_rlm_task(query, model):
            captured["query"] = query
            return {"success": True, "results": "answer"}

        import api.rlm.runner as rlm_mod
        monkeypatch.setattr(rlm_mod, "run_rlm_task", _fake_run_rlm_task)
        import api.utils as utils_mod
        monkeypatch.setattr(utils_mod, "get_model_context_window", lambda **kw: 131072)

        prompt = "normal sized prompt"
        result = asyncio.run(_rlm_generate(prompt, "model"))
        assert result == "answer"
        assert captured["query"] == prompt


# --------------------------------------------------------------------------- #
# _generate_answer
# --------------------------------------------------------------------------- #
# When --cov=api.expert.knowledge is active, coverage's import hooks create a
# second copy of api.expert.generate in sys.modules, so ``import api.expert
# .generate as gen_mod`` inside a test returns a different module object than
# the one _generate_answer / _stream_answer look up names from.  Patching
# gen_mod._rlm_generate then has no effect.  Instead, patch the function's
# own __globals__ dict — it is always the exact dict the function body reads.
_gen_globals = _generate_answer.__globals__
_stream_globals = _stream_answer.__globals__


class TestGenerateAnswer:
    def test_rlm_path_returns_text(self, monkeypatch):
        async def _fake_rlm(prompt, model):
            return "RLM answer"

        monkeypatch.setitem(_gen_globals, "_rlm_generate", _fake_rlm)
        # Ensure _safe_build_llm is NOT called (would fail if it were).
        monkeypatch.setitem(_gen_globals, "_safe_build_llm", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not build LLM")))

        result = asyncio.run(_generate_answer("prompt", "model", None, None, use_rlm=True))
        assert result == "RLM answer"

    def test_rlm_empty_falls_back_to_llm(self, monkeypatch):
        async def _fake_rlm(prompt, model):
            return ""

        monkeypatch.setitem(_gen_globals, "_rlm_generate", _fake_rlm)

        # Mock the LLM path.
        class _FakeLLM:
            async def generate(self, prompt):
                return "LLM answer"

        monkeypatch.setitem(_gen_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: _FakeLLM())

        result = asyncio.run(_generate_answer("prompt", "model", None, None, use_rlm=True))
        assert result == "LLM answer"

    def test_rlm_path_and_llm_both_empty_returns_empty(self, monkeypatch):
        async def _fake_rlm(prompt, model):
            return ""

        monkeypatch.setitem(_gen_globals, "_rlm_generate", _fake_rlm)

        class _FakeLLM:
            async def generate(self, prompt):
                return ""

        monkeypatch.setitem(_gen_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: _FakeLLM())

        result = asyncio.run(_generate_answer("prompt", "model", None, None, use_rlm=True))
        assert result == ""

    def test_no_rlm_llm_returns_text(self, monkeypatch):
        class _FakeLLM:
            async def generate(self, prompt):
                return "LLM answer"

        monkeypatch.setitem(_gen_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: _FakeLLM())

        result = asyncio.run(_generate_answer("prompt", "model", None, None, use_rlm=False))
        assert result == "LLM answer"

    def test_no_rlm_llm_none_returns_empty(self, monkeypatch):
        monkeypatch.setitem(_gen_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: None)

        result = asyncio.run(_generate_answer("prompt", "model", None, None, use_rlm=False))
        assert result == ""

    def test_llm_generate_exception_returns_empty(self, monkeypatch):
        class _FakeLLM:
            async def generate(self, prompt):
                raise RuntimeError("llm crashed")

        monkeypatch.setitem(_gen_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: _FakeLLM())

        result = asyncio.run(_generate_answer("prompt", "model", None, None, use_rlm=False))
        assert result == ""


# --------------------------------------------------------------------------- #
# _stream_answer
# --------------------------------------------------------------------------- #
class TestStreamAnswer:
    def test_rlm_path_yields_chunked_content(self, monkeypatch):
        async def _fake_rlm(prompt, model):
            return "RLM chunked answer text"

        monkeypatch.setitem(_stream_globals, "_rlm_generate", _fake_rlm)
        monkeypatch.setitem(_stream_globals, "_safe_build_llm", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not build LLM")))

        events = []
        async def _collect():
            async for ev in _stream_answer("prompt", "model", None, None, use_rlm=True):
                events.append(ev)
        asyncio.run(_collect())

        assert len(events) > 0
        assert all(e.type == EVENT_CONTENT for e in events)
        text = "".join(e.content for e in events)
        assert "RLM chunked answer text" in text

    def test_rlm_empty_falls_back_to_llm_stream(self, monkeypatch):
        async def _fake_rlm(prompt, model):
            return ""

        monkeypatch.setitem(_stream_globals, "_rlm_generate", _fake_rlm)

        class _FakeLLM:
            async def stream(self, prompt):
                yield ExpertStreamEvent(EVENT_CONTENT, "LLM ")
                yield ExpertStreamEvent(EVENT_CONTENT, "stream")

        monkeypatch.setitem(_stream_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: _FakeLLM())

        events = []
        async def _collect():
            async for ev in _stream_answer("prompt", "model", None, None, use_rlm=True):
                events.append(ev)
        asyncio.run(_collect())

        text = "".join(e.content for e in events)
        assert "LLM stream" in text

    def test_no_rlm_llm_stream_passthrough(self, monkeypatch):
        class _FakeLLM:
            async def stream(self, prompt):
                yield ExpertStreamEvent(EVENT_CONTENT, "chunk1")
                yield ExpertStreamEvent(EVENT_CONTENT, "chunk2")

        monkeypatch.setitem(_stream_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: _FakeLLM())

        events = []
        async def _collect():
            async for ev in _stream_answer("prompt", "model", None, None, use_rlm=False):
                events.append(ev)
        asyncio.run(_collect())

        assert len(events) == 2
        assert events[0].content == "chunk1"
        assert events[1].content == "chunk2"

    def test_no_rlm_llm_none_yields_nothing(self, monkeypatch):
        monkeypatch.setitem(_stream_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: None)

        events = []
        async def _collect():
            async for ev in _stream_answer("prompt", "model", None, None, use_rlm=False):
                events.append(ev)
        asyncio.run(_collect())

        assert events == []

    def test_rlm_empty_and_llm_none_yields_nothing(self, monkeypatch):
        async def _fake_rlm(prompt, model):
            return ""

        monkeypatch.setitem(_stream_globals, "_rlm_generate", _fake_rlm)
        monkeypatch.setitem(_stream_globals, "_safe_build_llm", lambda m, base_url=None, api_key=None: None)

        events = []
        async def _collect():
            async for ev in _stream_answer("prompt", "model", None, None, use_rlm=True):
                events.append(ev)
        asyncio.run(_collect())

        assert events == []
