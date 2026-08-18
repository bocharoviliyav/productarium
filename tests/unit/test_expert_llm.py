"""Unit tests for ``api.expert.llm``.

Covers:
- ``_extract_chunk_fields`` for all chunk shapes: native message
  (object + dict), OpenAI /v1 choices/delta, OpenAI object .choices[0].delta,
  adalflow .response/.data/.text, empty chunk -> (None, None).
- ``_ThinkingStreamParser``: feed/flush (open+close across chunks, unclosed
  flush as reasoning, partial tag buffering, no-tag passthrough, empty feed).
- ``_strip_thinking_tags``: closed block, unclosed block, no tags.
- ``_ExpertLLM.generate``: monkeypatch adal Generator; error returns '',
  normal returns data.
- ``_ExpertLLM.stream``: monkeypatch model_client.acall to yield chunks;
  fallback to generate on failure; no chunks -> fallback.
- ``_safe_build_llm``: success + exception returns None.
- ``_resolve_expert_model``: admin config present + missing -> defaults.
- ``_get_field``: dict + object, multiple keys, missing.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.expert.llm import (
    _ExpertLLM,
    _ThinkingStreamParser,
    _extract_chunk_fields,
    _get_field,
    _resolve_expert_model,
    _safe_build_llm,
    _strip_thinking_tags,
)
from api.expert.types import EVENT_CONTENT, EVENT_REASONING, ExpertStreamEvent


def _install_fake_adalflow(monkeypatch, *, generator_result=None, generate_text=None):
    """Inject a fake adalflow module into sys.modules so _ExpertLLM can construct.

    Under --cov, numpy's C extension corrupts when loaded via the adalflow import
    chain. This fake provides ``Generator`` (for __init__) and ``core.types.ModelType"
    (for stream) without loading real adalflow/numpy.
    """
    fake = types.ModuleType("adalflow")

    class _FakeGenerator:
        def __init__(self, *a, **kw):
            self._result = generator_result
            self._text = generate_text

        def __call__(self, prompt_kwargs):
            if self._result is not None:
                return self._result
            return SimpleNamespace(error=None, data=self._text, response=None)

    fake.Generator = _FakeGenerator

    # adalflow.core.types.ModelType is imported in _ExpertLLM.stream
    core_mod = types.ModuleType("adalflow.core")
    types_mod = types.ModuleType("adalflow.core.types")
    types_mod.ModelType = type("ModelType", (), {"LLM": "llm"})
    core_mod.types = types_mod
    fake.core = core_mod

    monkeypatch.setitem(sys.modules, "adalflow", fake)
    monkeypatch.setitem(sys.modules, "adalflow.core", core_mod)
    monkeypatch.setitem(sys.modules, "adalflow.core.types", types_mod)


def _install_fake_api_config(monkeypatch, *, client_chunks=None, client_exc=None):
    """Inject a fake api.config module with get_model_config + fake model client.

    The fake model client supports both generate (via Generator) and stream
    (via acall yielding chunks).
    """
    fake_cfg = types.ModuleType("api.config")

    class _FakeClient:
        def __init__(self, **kw):
            self._chunks = client_chunks
            self._exc = client_exc

        def convert_inputs_to_api_kwargs(self, **kw):
            return {"messages": [{"role": "user", "content": kw.get("input", "")}]}

        async def acall(self, **kw):
            if self._exc:
                raise self._exc
            return _AsyncIter(self._chunks or [])

    def _get_model_config(model=None):
        return {
            "model_client": _FakeClient,
            "model_kwargs": {"model": model or "test-model", "temperature": 0.1, "top_p": 0.9},
        }

    fake_cfg.get_model_config = _get_model_config
    monkeypatch.setitem(sys.modules, "api.config", fake_cfg)


class _AsyncIter:
    """Minimal async iterator wrapping a sync list."""

    def __init__(self, items):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


# --------------------------------------------------------------------------- #
# _get_field
# --------------------------------------------------------------------------- #
class TestGetField:
    def test_dict_first_key(self):
        assert _get_field({"content": "x"}, "content") == "x"

    def test_dict_second_key_when_first_missing(self):
        assert _get_field({"thinking": "t"}, "content", "thinking") == "t"

    def test_dict_all_missing_returns_none(self):
        assert _get_field({"a": 1}, "content", "thinking") is None

    def test_object_attribute(self):
        obj = SimpleNamespace(content="hello")
        assert _get_field(obj, "content") == "hello"

    def test_object_fallback_keys(self):
        obj = SimpleNamespace(reasoning="r")
        assert _get_field(obj, "content", "thinking", "reasoning") == "r"

    def test_empty_value_skipped(self):
        assert _get_field({"content": "", "thinking": "t"}, "content", "thinking") == "t"

    def test_no_keys_returns_none(self):
        assert _get_field({"content": "x"}) is None


# --------------------------------------------------------------------------- #
# _extract_chunk_fields
# --------------------------------------------------------------------------- #
class TestExtractChunkFields:
    def test_empty_chunk_returns_none_none(self):
        assert _extract_chunk_fields(SimpleNamespace()) == (None, None)

    def test_native_object_message_content(self):
        msg = SimpleNamespace(content="hello", thinking=None)
        chunk = SimpleNamespace(message=msg)
        assert _extract_chunk_fields(chunk) == ("hello", None)

    def test_native_object_message_thinking(self):
        msg = SimpleNamespace(content=None, thinking="reasoning text")
        chunk = SimpleNamespace(message=msg)
        content, reasoning = _extract_chunk_fields(chunk)
        assert reasoning == "reasoning text"

    def test_native_dict_message_content(self):
        chunk = SimpleNamespace(message={"content": "hi", "thinking": None})
        assert _extract_chunk_fields(chunk) == ("hi", None)

    def test_native_dict_message_reasoning(self):
        chunk = SimpleNamespace(message={"content": "", "reasoning_content": "rc"})
        _, reasoning = _extract_chunk_fields(chunk)
        assert reasoning == "rc"

    def test_dict_chunk_with_message_key(self):
        chunk = {"message": {"content": "text", "thinking": None}}
        assert _extract_chunk_fields(chunk) == ("text", None)

    def test_dict_chunk_with_choices_delta(self):
        chunk = {"choices": [{"delta": {"content": "delta_text"}}]}
        assert _extract_chunk_fields(chunk) == ("delta_text", None)

    def test_dict_chunk_with_choices_delta_reasoning(self):
        chunk = {"choices": [{"delta": {"content": None, "reasoning": "r"}}]}
        _, reasoning = _extract_chunk_fields(chunk)
        assert reasoning == "r"

    def test_dict_chunk_with_choices_message_key(self):
        chunk = {"choices": [{"message": {"content": "msg_text"}}]}
        assert _extract_chunk_fields(chunk) == ("msg_text", None)

    def test_dict_chunk_with_top_level_content(self):
        chunk = {"content": "top_content"}
        assert _extract_chunk_fields(chunk) == ("top_content", None)

    def test_dict_chunk_empty_returns_none(self):
        assert _extract_chunk_fields({}) == (None, None)

    def test_dict_chunk_empty_choices(self):
        assert _extract_chunk_fields({"choices": []}) == (None, None)

    def test_openai_object_choices_delta(self):
        delta = SimpleNamespace(content="openai_text", reasoning_content=None)
        choice = SimpleNamespace(delta=delta)
        chunk = SimpleNamespace(choices=[choice])
        assert _extract_chunk_fields(chunk) == ("openai_text", None)

    def test_openai_object_choices_delta_reasoning(self):
        delta = SimpleNamespace(content=None, reasoning_content="ocr")
        choice = SimpleNamespace(delta=delta)
        chunk = SimpleNamespace(choices=[choice])
        _, reasoning = _extract_chunk_fields(chunk)
        assert reasoning == "ocr"

    def test_openai_object_choices_delta_empty(self):
        delta = SimpleNamespace(content=None, reasoning_content=None)
        choice = SimpleNamespace(delta=delta)
        chunk = SimpleNamespace(choices=[choice])
        assert _extract_chunk_fields(chunk) == (None, None)

    def test_openai_object_choices_index_error(self):
        chunk = SimpleNamespace(choices=[])
        assert _extract_chunk_fields(chunk) == (None, None)

    def test_adalflow_response_attr(self):
        chunk = SimpleNamespace(response="resp_text")
        assert _extract_chunk_fields(chunk) == ("resp_text", None)

    def test_adalflow_data_attr(self):
        chunk = SimpleNamespace(data="data_text")
        assert _extract_chunk_fields(chunk) == ("data_text", None)

    def test_adalflow_text_attr(self):
        chunk = SimpleNamespace(text="text_val")
        assert _extract_chunk_fields(chunk) == ("text_val", None)

    def test_adalflow_non_string_data_skipped(self):
        chunk = SimpleNamespace(data=123)
        assert _extract_chunk_fields(chunk) == (None, None)

    def test_native_empty_content_skipped(self):
        msg = SimpleNamespace(content="", thinking=None)
        chunk = SimpleNamespace(message=msg)
        assert _extract_chunk_fields(chunk) == (None, None)


# --------------------------------------------------------------------------- #
# _ThinkingStreamParser
# --------------------------------------------------------------------------- #
class TestThinkingStreamParser:
    def test_empty_feed_returns_nothing(self):
        p = _ThinkingStreamParser()
        assert p.feed("") == []

    def test_no_tags_passthrough_as_content(self):
        p = _ThinkingStreamParser()
        events = p.feed("just plain text")
        assert len(events) == 1
        assert events[0].type == EVENT_CONTENT
        assert events[0].content == "just plain text"

    def test_open_close_in_single_chunk(self):
        p = _ThinkingStreamParser()
        events = p.feed("before<think>reasoning</think>after")
        types = [e.type for e in events]
        contents = [e.content for e in events]
        assert EVENT_CONTENT in types
        assert EVENT_REASONING in types
        assert "before" in contents
        assert "reasoning" in contents
        assert "after" in contents

    def test_open_close_across_chunks(self):
        p = _ThinkingStreamParser()
        events1 = p.feed("text<think>rea")
        events2 = p.feed("soning</think>more")
        all_events = events1 + events2
        types = [e.type for e in all_events]
        assert EVENT_CONTENT in types
        assert EVENT_REASONING in types
        reasoning_text = "".join(e.content for e in all_events if e.type == EVENT_REASONING)
        assert "reasoning" in reasoning_text

    def test_unclosed_think_emitted_as_reasoning_during_feed(self):
        # When the buffer ends inside <think> and there's no partial close-tag
        # match, the entire remaining buffer is emitted as reasoning during
        # feed() (not flush). flush() then returns [].
        p = _ThinkingStreamParser()
        events = p.feed("text<think>unterminated reasoning")
        reasoning_events = [e for e in events if e.type == EVENT_REASONING]
        assert len(reasoning_events) == 1
        assert "unterminated reasoning" in reasoning_events[0].content
        assert p.flush() == []

    def test_flush_empty_buffer_returns_empty(self):
        p = _ThinkingStreamParser()
        assert p.flush() == []

    def test_flush_after_complete_returns_empty(self):
        p = _ThinkingStreamParser()
        p.feed("text<think>r</think>after")
        assert p.flush() == []

    def test_partial_open_tag_buffered(self):
        p = _ThinkingStreamParser()
        events = p.feed("hello <thi")
        assert any(e.type == EVENT_CONTENT and "hello" in e.content for e in events)
        events2 = p.feed("nk>reasoning</think>done")
        all_events = events + events2
        assert any(e.type == EVENT_REASONING for e in all_events)

    def test_partial_close_tag_buffered(self):
        p = _ThinkingStreamParser()
        p.feed("<think>reasoning</thi")
        events2 = p.feed("nk>after")
        all_events = p.feed("") + events2
        all_events += p.flush()
        assert any(e.type == EVENT_CONTENT and "after" in e.content for e in all_events)

    def test_whitespace_stripped_after_tags(self):
        p = _ThinkingStreamParser()
        events = p.feed("<think>r</think>\n\nvisible")
        contents = [e.content for e in events if e.type == EVENT_CONTENT]
        assert any("visible" == c for c in contents)

    def test_whitespace_stripped_after_open_tag(self):
        p = _ThinkingStreamParser()
        events = p.feed("<think>\n\nreasoning</think>after")
        reasoning = "".join(e.content for e in events if e.type == EVENT_REASONING)
        assert reasoning.startswith("reasoning")

    def test_multiple_think_blocks(self):
        p = _ThinkingStreamParser()
        events = p.feed("a<think>r1</think>b<think>r2</think>c")
        reasoning_parts = [e.content for e in events if e.type == EVENT_REASONING]
        assert "r1" in reasoning_parts
        assert "r2" in reasoning_parts
        content_parts = [e.content for e in events if e.type == EVENT_CONTENT]
        assert "a" in content_parts
        assert "b" in content_parts
        assert "c" in content_parts

    def test_partial_tag_match_entire_buffer_is_prefix(self):
        # The entire buffer is a prefix of the tag (e.g. buffer="<thi").
        # The parser emits a content event with an empty string (the text
        # before the partial tag prefix) and buffers the prefix.
        p = _ThinkingStreamParser()
        events = p.feed("<thi")
        assert len(events) == 1
        assert events[0].type == EVENT_CONTENT
        assert events[0].content == ""
        events2 = p.feed("nk>reasoning</think>after")
        all_events = events + events2
        assert any(e.type == EVENT_REASONING for e in all_events)


# --------------------------------------------------------------------------- #
# _strip_thinking_tags
# --------------------------------------------------------------------------- #
class TestStripThinkingTags:
    def test_no_tags(self):
        assert _strip_thinking_tags("plain text") == "plain text"

    def test_closed_block(self):
        assert _strip_thinking_tags("before<think>r</think>after") == "beforeafter"

    def test_closed_block_with_whitespace(self):
        assert _strip_thinking_tags("before<think>r</think>   after") == "beforeafter"

    def test_unclosed_block(self):
        assert _strip_thinking_tags("text<think>unterminated") == "text"

    def test_empty_string(self):
        assert _strip_thinking_tags("") == ""

    def test_multiline_block(self):
        text = "a<think>line1\nline2</think>b"
        assert _strip_thinking_tags(text) == "ab"


# --------------------------------------------------------------------------- #
# _ExpertLLM.generate
# --------------------------------------------------------------------------- #
class TestExpertLLMGenerate:
    def _make_llm(self, monkeypatch, generator_result):
        """Build an _ExpertLLM with a mocked adal Generator."""
        _install_fake_adalflow(monkeypatch, generator_result=generator_result)
        _install_fake_api_config(monkeypatch)
        return _ExpertLLM("test-model")

    def test_generate_returns_data(self, monkeypatch):
        result = SimpleNamespace(error=None, data="generated text", response=None, answer=None, raw_response=None, output=None)
        llm = self._make_llm(monkeypatch, result)
        text = asyncio.run(llm.generate("my prompt"))
        assert text == "generated text"

    def test_generate_returns_response_when_no_data(self, monkeypatch):
        result = SimpleNamespace(error=None, data=None, response="resp", answer=None, raw_response=None, output=None)
        llm = self._make_llm(monkeypatch, result)
        text = asyncio.run(llm.generate("prompt"))
        assert text == "resp"

    def test_generate_returns_answer_when_no_data_response(self, monkeypatch):
        result = SimpleNamespace(error=None, data=None, response=None, answer="ans", raw_response=None, output=None)
        llm = self._make_llm(monkeypatch, result)
        text = asyncio.run(llm.generate("prompt"))
        assert text == "ans"

    def test_generate_error_returns_empty(self, monkeypatch):
        result = SimpleNamespace(error="some error", data=None, response=None)
        llm = self._make_llm(monkeypatch, result)
        text = asyncio.run(llm.generate("prompt"))
        assert text == ""

    def test_generate_no_fields_returns_empty(self, monkeypatch):
        result = SimpleNamespace(error=None, data=None, response=None, answer=None, raw_response=None, output=None)
        llm = self._make_llm(monkeypatch, result)
        text = asyncio.run(llm.generate("prompt"))
        assert text == ""


# --------------------------------------------------------------------------- #
# _ExpertLLM.stream
# --------------------------------------------------------------------------- #
class TestExpertLLMStream:
    def _make_llm_with_client(self, monkeypatch, acall_chunks=None, acall_exc=None, generate_text=None):
        """Build an _ExpertLLM with a mocked model_client for streaming."""
        _install_fake_adalflow(monkeypatch, generator_result=None, generate_text=generate_text)
        _install_fake_api_config(monkeypatch, client_chunks=acall_chunks, client_exc=acall_exc)
        return _ExpertLLM("test-model")

    def test_stream_yields_content_from_chunks(self, monkeypatch):
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello ", reasoning_content=None))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world", reasoning_content=None))]),
        ]
        llm = self._make_llm_with_client(monkeypatch, acall_chunks=chunks)
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        assert len(events) >= 2
        assert all(e.type == EVENT_CONTENT for e in events)
        text = "".join(e.content for e in events)
        assert "hello" in text
        assert "world" in text

    def test_stream_yields_reasoning_from_chunks(self, monkeypatch):
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content="thinking..."))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="answer", reasoning_content=None))]),
        ]
        llm = self._make_llm_with_client(monkeypatch, acall_chunks=chunks)
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        types = [e.type for e in events]
        assert EVENT_REASONING in types
        assert EVENT_CONTENT in types

    def test_stream_inline_think_tags_parsed(self, monkeypatch):
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="text<think>reasoning</think>after", reasoning_content=None))]),
        ]
        llm = self._make_llm_with_client(monkeypatch, acall_chunks=chunks)
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        types = [e.type for e in events]
        assert EVENT_REASONING in types
        assert EVENT_CONTENT in types

    def test_stream_no_chunks_falls_back_to_generate(self, monkeypatch):
        chunks = []
        llm = self._make_llm_with_client(monkeypatch, acall_chunks=chunks, generate_text="fallback text")
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        assert len(events) > 0
        text = "".join(e.content for e in events)
        assert "fallback text" in text

    def test_stream_acall_exception_falls_back_to_generate(self, monkeypatch):
        llm = self._make_llm_with_client(
            monkeypatch,
            acall_chunks=None,
            acall_exc=RuntimeError("connection refused"),
            generate_text="fallback after error",
        )
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        text = "".join(e.content for e in events)
        assert "fallback after error" in text


# --------------------------------------------------------------------------- #
# _safe_build_llm
# --------------------------------------------------------------------------- #
class TestSafeBuildLLM:
    def test_success_returns_llm(self, monkeypatch):
        _install_fake_adalflow(monkeypatch, generator_result=None, generate_text="")
        _install_fake_api_config(monkeypatch)
        llm = _safe_build_llm("m")
        assert llm is not None
        assert isinstance(llm, _ExpertLLM)

    def test_exception_returns_none(self, monkeypatch):
        # Patch get_model_config on the real api.config module (if loaded) or
        # inject a fake api.config that raises.
        import sys
        if "api.config" in sys.modules:
            monkeypatch.setattr(sys.modules["api.config"], "get_model_config", lambda model=None: (_ for _ in ()).throw(RuntimeError("config error")))
        else:
            fake_cfg = types.ModuleType("api.config")
            fake_cfg.get_model_config = lambda model=None: (_ for _ in ()).throw(RuntimeError("config error"))
            monkeypatch.setitem(sys.modules, "api.config", fake_cfg)
        result = _safe_build_llm("m")
        assert result is None


# --------------------------------------------------------------------------- #
# _resolve_expert_model
# --------------------------------------------------------------------------- #
def _install_fake_abstraction(monkeypatch, get_task_config_fn):
    """Inject a fake api.config.abstraction module with a mock get_task_config.

    Under --cov, importing the real api.config.abstraction triggers the
    api.config -> adalflow -> numpy chain which corrupts. This fake avoids it.
    """
    fake_abs = types.ModuleType("api.config.abstraction")
    fake_abs.get_task_config = get_task_config_fn
    monkeypatch.setitem(sys.modules, "api.config.abstraction", fake_abs)


class TestResolveExpertModel:
    def test_admin_config_present(self, monkeypatch):
        def _fake_get_task_config(task):
            assert task == "expert"
            return {"model": "custom-model", "base_url": "http://gw:8080/v1", "api_key": "key123"}

        _install_fake_abstraction(monkeypatch, _fake_get_task_config)
        model, base_url, api_key = _resolve_expert_model(None)
        assert model == "custom-model"
        assert base_url == "http://gw:8080/v1"
        assert api_key == "key123"

    def test_admin_config_present_explicit_model_wins(self, monkeypatch):
        _install_fake_abstraction(
            monkeypatch,
            lambda task: {"model": "stored-model", "base_url": None, "api_key": None},
        )
        model, base_url, api_key = _resolve_expert_model("explicit-model")
        assert model == "explicit-model"

    def test_admin_config_missing_uses_defaults(self, monkeypatch):
        _install_fake_abstraction(monkeypatch, lambda task: None)
        model, base_url, api_key = _resolve_expert_model(None)
        assert model == "qwen/qwen3.6-27b"
        assert base_url is None
        assert api_key is None

    def test_admin_config_empty_dict_uses_defaults(self, monkeypatch):
        _install_fake_abstraction(monkeypatch, lambda task: {})
        model, base_url, api_key = _resolve_expert_model(None)
        assert model == "qwen/qwen3.6-27b"
        assert base_url is None
        assert api_key is None

    def test_get_task_config_exception_uses_defaults(self, monkeypatch):
        def _boom(task):
            raise RuntimeError("db down")

        _install_fake_abstraction(monkeypatch, _boom)
        model, base_url, api_key = _resolve_expert_model("my-model")
        assert model == "my-model"
        assert base_url is None
        assert api_key is None

    def test_get_task_config_exception_no_model_uses_default(self, monkeypatch):
        def _boom(task):
            raise RuntimeError("db down")

        _install_fake_abstraction(monkeypatch, _boom)
        model, base_url, api_key = _resolve_expert_model(None)
        assert model == "qwen/qwen3.6-27b"
