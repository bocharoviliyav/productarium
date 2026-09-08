"""Unit tests for ``api.expert.llm``.

Covers:
- ``_extract_chunk_fields`` for all chunk shapes: native message
  (object + dict), OpenAI /v1 choices/delta, OpenAI object .choices[0].delta,
  plain-text object shapes (.response/.data/.text), empty chunk -> (None, None).
- ``_ThinkingStreamParser``: feed/flush (open+close across chunks, unclosed
  flush as reasoning, partial tag buffering, no-tag passthrough, empty feed).
- ``_strip_thinking_tags``: closed block, unclosed block, no tags.
- ``_ExpertLLM.generate``: delegates to the wrapped ``api.llm.GenerateLLM``;
  errors surface as ``""`` (suppressed inside GenerateLLM contract) — here the
  wrapper is exercised via a monkeypatched GenerateLLM module attribute.
- ``_ExpertLLM.stream``: content + reasoning deltas via
  ``api.llm.stream_chat_fields``; inline ``<think>`` tags parsed; fallback to
  chunked generate when the stream yields nothing / raises.
- ``_safe_build_llm``: success + exception returns None.
- ``_resolve_expert_model``: admin config present + missing -> defaults.
- ``_get_field``: dict + object, multiple keys, missing.
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.llm as llm_pkg
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


def _install_fake_generate(monkeypatch, *, generate_text=""):
    """Patch api.llm.GenerateLLM with a fake returning canned text.

    The expert wrapper imports ``GenerateLLM`` lazily from ``api.llm`` inside
    ``_ExpertLLM.__init__``, so patching the package attribute is enough.
    """
    captured: dict = {}

    class _FakeGenerateLLM:
        def __init__(self, model=None, base_url=None, api_key=None):
            captured["model"] = model
            captured["base_url"] = base_url
            captured["api_key"] = api_key

        async def generate(self, prompt: str) -> str:
            captured["prompts"] = captured.get("prompts", []) + [prompt]
            return generate_text

    monkeypatch.setattr(llm_pkg, "GenerateLLM", _FakeGenerateLLM)
    return captured


def _install_fake_stream(monkeypatch, *, pairs=None, exc=None):
    """Patch api.llm.stream_chat_fields with a fake async generator."""
    pairs = pairs or []

    async def _fake_stream(prompt, model=None, base_url=None, api_key=None):
        if exc is not None:
            raise exc
        for content, reasoning in pairs:
            yield content, reasoning

    monkeypatch.setattr(llm_pkg, "stream_chat_fields", _fake_stream)
    return _fake_stream


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

    def test_plain_object_response_attr(self):
        chunk = SimpleNamespace(response="resp_text")
        assert _extract_chunk_fields(chunk) == ("resp_text", None)

    def test_plain_object_data_attr(self):
        chunk = SimpleNamespace(data="data_text")
        assert _extract_chunk_fields(chunk) == ("data_text", None)

    def test_plain_object_text_attr(self):
        chunk = SimpleNamespace(text="text_val")
        assert _extract_chunk_fields(chunk) == ("text_val", None)

    def test_plain_object_non_string_data_skipped(self):
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
    def test_generate_delegates_to_wrapped_llm(self, monkeypatch):
        captured = _install_fake_generate(monkeypatch, generate_text="generated text")
        llm = _ExpertLLM("test-model")
        text = asyncio.run(llm.generate("my prompt"))
        assert text == "generated text"
        assert captured["prompts"] == ["my prompt"]
        assert captured["model"] == "test-model"

    def test_generate_failure_returns_empty(self, monkeypatch):
        class _FailingLLM:
            def __init__(self, model=None, base_url=None, api_key=None):
                pass

            async def generate(self, prompt: str) -> str:
                # GenerateLLM's contract: "" on failure, never raises.
                return ""

        monkeypatch.setattr(llm_pkg, "GenerateLLM", _FailingLLM)
        llm = _ExpertLLM("test-model")
        assert asyncio.run(llm.generate("prompt")) == ""


# --------------------------------------------------------------------------- #
# _ExpertLLM.stream
# --------------------------------------------------------------------------- #
class TestExpertLLMStream:
    def _make_llm(self, monkeypatch):
        _install_fake_generate(monkeypatch, generate_text="")
        return _ExpertLLM("test-model")

    def test_stream_yields_content_from_chunks(self, monkeypatch):
        _install_fake_stream(
            monkeypatch,
            pairs=[("hello ", ""), ("world", "")],
        )
        llm = self._make_llm(monkeypatch)
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        assert len(events) == 2
        assert all(e.type == EVENT_CONTENT for e in events)
        text = "".join(e.content for e in events)
        assert "hello" in text
        assert "world" in text

    def test_stream_yields_reasoning_from_chunks(self, monkeypatch):
        _install_fake_stream(
            monkeypatch,
            pairs=[("", "thinking..."), ("answer", "")],
        )
        llm = self._make_llm(monkeypatch)
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        types = [e.type for e in events]
        assert EVENT_REASONING in types
        assert EVENT_CONTENT in types

    def test_stream_inline_think_tags_parsed(self, monkeypatch):
        _install_fake_stream(
            monkeypatch,
            pairs=[("text<think>reasoning</think>after", "")],
        )
        llm = self._make_llm(monkeypatch)
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        types = [e.type for e in events]
        assert EVENT_REASONING in types
        assert EVENT_CONTENT in types

    def test_stream_no_chunks_falls_back_to_generate(self, monkeypatch):
        _install_fake_stream(monkeypatch, pairs=[])
        _install_fake_generate(monkeypatch, generate_text="fallback text")
        llm = _ExpertLLM("test-model")
        events = []
        async def _collect():
            async for ev in llm.stream("prompt"):
                events.append(ev)
        asyncio.run(_collect())
        assert len(events) > 0
        text = "".join(e.content for e in events)
        assert "fallback text" in text

    def test_stream_exception_falls_back_to_generate(self, monkeypatch):
        _install_fake_stream(monkeypatch, exc=RuntimeError("connection refused"))
        _install_fake_generate(monkeypatch, generate_text="fallback after error")
        llm = _ExpertLLM("test-model")
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
        _install_fake_generate(monkeypatch, generate_text="")
        llm = _safe_build_llm("m")
        assert llm is not None
        assert isinstance(llm, _ExpertLLM)

    def test_exception_returns_none(self, monkeypatch):
        class _Boom:
            def __init__(self, *a, **kw):
                raise RuntimeError("config error")

        monkeypatch.setattr(llm_pkg, "GenerateLLM", _Boom)
        assert _safe_build_llm("m") is None


# --------------------------------------------------------------------------- #
# _resolve_expert_model
# --------------------------------------------------------------------------- #
def _install_fake_abstraction(monkeypatch, get_task_config_fn):
    """Patch the real api.config.abstraction.get_task_config."""
    import api.config.abstraction as ab

    monkeypatch.setattr(ab, "get_task_config", get_task_config_fn)


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
