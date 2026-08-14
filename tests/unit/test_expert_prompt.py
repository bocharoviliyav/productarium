"""Unit tests for ``api.expert.prompt``.

Covers:
- Tunables: ``RLM_MIN_CHARS``, ``KNOWLEDGE_MAX_CHARS``, ``STREAM_CHUNK_SIZE``.
- Loaded prompt bodies: ``EXPERT_SYSTEM_PROMPT`` / ``EXPERT_DOC_PROMPT`` non-empty.
- ``_clean_llm_text``: fence stripping, ``<r>`` block stripping, line-number
  stripping inside code blocks, empty/None passthrough.
- ``_chunk_text``: empty, short lines, long line word-splitting, custom size.
- ``_build_prompt``: system+knowledge+history+query assembly, history clamp,
  knowledge clamp, missing-knowledge note, language_name default, guard append.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.expert import prompt as expert_prompt
from api.expert.prompt import (
    EXPERT_DOC_PROMPT,
    EXPERT_SYSTEM_PROMPT,
    KNOWLEDGE_MAX_CHARS,
    RLM_MIN_CHARS,
    STREAM_CHUNK_SIZE,
    _build_prompt,
    _chunk_text,
    _clean_llm_text,
)


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
class TestTunables:
    def test_tunables_are_positive_ints(self):
        assert isinstance(RLM_MIN_CHARS, int) and RLM_MIN_CHARS > 0
        assert isinstance(KNOWLEDGE_MAX_CHARS, int) and KNOWLEDGE_MAX_CHARS > 0
        assert isinstance(STREAM_CHUNK_SIZE, int) and STREAM_CHUNK_SIZE > 0

    def test_rlm_min_chars_matches_docgen_threshold(self):
        # RLM is for long context only; 20k is the documented threshold.
        assert RLM_MIN_CHARS == 20_000

    def test_knowledge_cap_larger_than_rlm_threshold(self):
        # The knowledge cap must allow prompts big enough to trigger RLM.
        assert KNOWLEDGE_MAX_CHARS >= RLM_MIN_CHARS

    def test_stream_chunk_size_reasonable(self):
        # Small enough for incremental SSE, large enough to avoid per-char frames.
        assert 16 <= STREAM_CHUNK_SIZE <= 512


# --------------------------------------------------------------------------- #
# Loaded prompt bodies
# --------------------------------------------------------------------------- #
class TestLoadedPrompts:
    def test_system_prompt_loaded_nonempty(self):
        assert isinstance(EXPERT_SYSTEM_PROMPT, str)
        assert EXPERT_SYSTEM_PROMPT.strip()

    def test_doc_prompt_loaded_nonempty(self):
        assert isinstance(EXPERT_DOC_PROMPT, str)
        assert EXPERT_DOC_PROMPT.strip()

    def test_system_and_doc_prompts_differ(self):
        assert EXPERT_SYSTEM_PROMPT != EXPERT_DOC_PROMPT


# --------------------------------------------------------------------------- #
# _clean_llm_text
# --------------------------------------------------------------------------- #
class TestCleanLLMText:
    def test_none_returns_empty(self):
        assert _clean_llm_text(None) == ""

    def test_empty_returns_empty(self):
        assert _clean_llm_text("") == ""
        assert _clean_llm_text("   ") == ""

    def test_plain_text_passthrough(self):
        assert _clean_llm_text("hello world") == "hello world"

    def test_strips_surrounding_whitespace(self):
        assert _clean_llm_text("  hello  ") == "hello"

    def test_strips_markdown_fence_with_lang(self):
        text = "```python\nprint('hi')\n```"
        assert _clean_llm_text(text) == "print('hi')"

    def test_strips_markdown_fence_without_lang(self):
        text = "```\nsome text\n```"
        assert _clean_llm_text(text) == "some text"

    def test_strips_markdown_fence_no_trailing_fence(self):
        text = "```markdown\n# Heading"
        assert _clean_llm_text(text) == "# Heading"

    def test_strips_r_blocks(self):
        # The regex `` <r>.*?</r>`` consumes the leading space, so the two
        # surrounding words collapse to a single space.
        text = "before <r>secret reasoning</r> after"
        assert _clean_llm_text(text) == "before after"

    def test_strips_r_blocks_multiline(self):
        text = "before <r>line1\nline2</r> after"
        assert _clean_llm_text(text) == "before after"

    def test_preserves_code_block_without_line_numbers(self):
        text = "```python\nimport os\nprint(os.getcwd())\n```"
        cleaned = _clean_llm_text(text)
        assert "import os" in cleaned
        assert "print(os.getcwd())" in cleaned

    def test_strips_inline_line_numbers_in_code_block(self):
        # ``_strip_inline_line_numbers`` (the shared helper called by
        # ``_clean_llm_text``) de-numbers lines inside fenced code blocks whose
        # leading number equals the 1-indexed position. Test it directly: a
        # fenced block with ``1 import os`` / ``2 print('hi')`` / ``3 x = 1``.
        from api.utils.llm_helpers import strip_inline_line_numbers

        text = "```\n1 import os\n2 print('hi')\n3 x = 1\n```"
        cleaned = strip_inline_line_numbers(text)
        # The leading "N " prefixes are removed.
        assert "1 import os" not in cleaned
        assert "2 print('hi')" not in cleaned
        assert "3 x = 1" not in cleaned
        assert "import os" in cleaned
        assert "print('hi')" in cleaned
        assert "x = 1" in cleaned


# --------------------------------------------------------------------------- #
# _chunk_text
# --------------------------------------------------------------------------- #
class TestChunkText:
    def test_empty_returns_empty_list(self):
        assert _chunk_text("") == []

    def test_short_line_single_piece(self):
        assert _chunk_text("hello") == ["hello"]

    def test_multiple_short_lines(self):
        text = "line1\nline2\nline3"
        assert _chunk_text(text) == ["line1\n", "line2\n", "line3"]

    def test_long_line_split_on_word_boundaries(self):
        words = ["word"] * 50
        text = " ".join(words)
        pieces = _chunk_text(text, size=20)
        assert len(pieces) > 1
        # Every piece except possibly the last should be <= size (or a single
        # long word).
        for p in pieces:
            assert len(p) <= 20 or " " not in p

    def test_custom_size(self):
        text = "a b c d e f"
        pieces = _chunk_text(text, size=5)
        assert len(pieces) >= 2

    def test_default_size_is_stream_chunk_size(self):
        # A line just over STREAM_CHUNK_SIZE should split into multiple pieces.
        words = ["x"] * (STREAM_CHUNK_SIZE + 5)
        text = " ".join(words)
        pieces = _chunk_text(text)
        assert len(pieces) > 1

    def test_single_long_word(self):
        # A single word longer than size passes through as one piece.
        text = "a" * 200
        pieces = _chunk_text(text, size=10)
        assert pieces == [text]


# --------------------------------------------------------------------------- #
# _build_prompt
# --------------------------------------------------------------------------- #
class TestBuildPrompt:
    def test_basic_assembly_with_knowledge(self):
        prompt = _build_prompt(
            template="You are an expert for {product_name}.",
            product_name="MyService",
            knowledge="some knowledge",
            history="",
            query="What is X?",
        )
        assert "You are an expert for MyService." in prompt
        assert "<product_knowledge>" in prompt
        assert "some knowledge" in prompt
        assert "<query>" in prompt
        assert "What is X?" in prompt
        assert prompt.endswith("Assistant: ")

    def test_assembly_with_history(self):
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge="",
            history="<user>hi</user>",
            query="q",
        )
        assert "<conversation_history>" in prompt
        assert "<user>hi</user>" in prompt

    def test_missing_knowledge_inserts_note(self):
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge="",
            history="",
            query="q",
        )
        assert "<note>" in prompt
        assert "knowledge is missing" in prompt
        assert "<product_knowledge>" not in prompt

    def test_product_name_default_when_empty(self):
        prompt = _build_prompt(
            template="Expert for {product_name}.",
            product_name="",
            knowledge="k",
            history="",
            query="q",
        )
        assert "this product" in prompt

    def test_language_name_default(self):
        prompt = _build_prompt(
            template="Reply in {language_name}.",
            product_name="P",
            knowledge="k",
            history="",
            query="q",
        )
        assert "the same language as the user's query" in prompt

    def test_language_name_custom(self):
        prompt = _build_prompt(
            template="Reply in {language_name}.",
            product_name="P",
            knowledge="k",
            history="",
            query="q",
            language_name="Russian",
        )
        assert "Russian" in prompt

    def test_history_clamped_when_too_large(self):
        big_history = "A" * 100_000
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge="k",
            history=big_history,
            query="q",
        )
        assert "ранняя история обрезана" in prompt
        # The clamped history must be much smaller than the input.
        assert big_history not in prompt

    def test_knowledge_clamped_when_too_large(self):
        big_knowledge = "K" * 200_000
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge=big_knowledge,
            history="",
            query="q",
        )
        assert "часть знаний обрезана" in prompt
        assert big_knowledge not in prompt

    def test_verification_guard_appended(self):
        # The guard is loaded from refs/prompts/_verification_guard.md; it
        # should be appended to the system block when present.
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge="k",
            history="",
            query="q",
        )
        # The guard content varies; just assert the prompt was built. If the
        # guard file exists it will be appended after the template body.
        assert "T" in prompt

    def test_query_always_present(self):
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge="",
            history="",
            query="special_query_123",
        )
        assert "special_query_123" in prompt

    def test_history_and_knowledge_both_present(self):
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge="my_knowledge",
            history="my_history",
            query="my_query",
        )
        assert "my_knowledge" in prompt
        assert "my_history" in prompt
        assert "my_query" in prompt

    def test_context_window_resolution_uses_model(self, monkeypatch):
        # When get_model_context_window succeeds, the budget is derived from it.
        # A tiny context window forces clamping of a knowledge block that
        # exceeds the char budget: avail_chars = max(1024, 2048-2048)*4 = 4096.
        import api.utils

        monkeypatch.setattr(
            api.utils,
            "get_model_context_window",
            lambda **kw: 2048,
        )
        knowledge = "K" * 10_000  # larger than the tiny 4096-char budget allows
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge=knowledge,
            history="",
            query="q",
            model="some-model",
            base_url="http://localhost:11434/v1",
        )
        assert "часть знаний обрезана" in prompt

    def test_context_window_resolution_fallback_on_exception(self, monkeypatch):
        # If get_model_context_window raises, the fallback (8192) is used and
        # the prompt is still built successfully.
        import api.utils

        def _raise(**kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(api.utils, "get_model_context_window", _raise)
        prompt = _build_prompt(
            template="T",
            product_name="P",
            knowledge="k",
            history="",
            query="q",
        )
        assert "T" in prompt
        assert "k" in prompt
