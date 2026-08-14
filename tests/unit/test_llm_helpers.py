"""Unit tests for ``api.utils.llm_helpers`` (shared LLM text helpers).

Covers:
- ``cap(text, limit)`` (empty, under limit, over limit with truncation suffix).
- ``safe_replace(template, variables)`` (basic substitution, None values,
  unmatched placeholders left intact, empty template).
- ``strip_number_prefixes_from_block`` (position-matched stripping, numeric
  literals protected, out-of-order protected, bare-number lines, single-line
  no-op, empty block).
- ``strip_inline_line_numbers`` (fenced code blocks, mermaid blocks skipped,
  prose untouched, no fence passthrough, empty/None input).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.utils.llm_helpers import (
    LINE_NUM_ONLY_RE,
    LINE_NUM_PREFIX_RE,
    cap,
    safe_replace,
    strip_inline_line_numbers,
    strip_number_prefixes_from_block,
)


# ---------------------------------------------------------------------------
# cap
# ---------------------------------------------------------------------------

class TestCap:
    def test_empty_string(self):
        assert cap("", 100) == ""

    def test_none(self):
        assert cap(None, 100) == ""

    def test_under_limit_returns_as_is(self):
        text = "short text"
        assert cap(text, 100) == text

    def test_exactly_at_limit(self):
        text = "exactly 20 chars!!"
        assert cap(text, len(text)) == text

    def test_over_limit_truncated_with_suffix(self):
        text = "a" * 200
        result = cap(text, 50)
        assert result.startswith("a" * 50)
        assert "... (обрезано для контекста LLM)" in result

    def test_truncation_suffix_format(self):
        result = cap("x" * 100, 10)
        assert result == "xxxxxxxxxx\n... (обрезано для контекста LLM)\n"


# ---------------------------------------------------------------------------
# safe_replace
# ---------------------------------------------------------------------------

class TestSafeReplace:
    def test_basic_substitution(self):
        result = safe_replace("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_placeholders(self):
        result = safe_replace("{a} + {b} = {c}", {"a": "1", "b": "2", "c": "3"})
        assert result == "1 + 2 = 3"

    def test_none_value_becomes_empty(self):
        result = safe_replace("val={x}", {"x": None})
        assert result == "val="

    def test_non_string_value_converted(self):
        result = safe_replace("n={count}", {"count": 42})
        assert result == "n=42"

    def test_unmatched_placeholder_left_intact(self):
        result = safe_replace("{a} and {b}", {"a": "1"})
        assert result == "1 and {b}"

    def test_empty_template(self):
        assert safe_replace("", {"a": "1"}) == ""

    def test_none_template(self):
        assert safe_replace(None, {"a": "1"}) == ""

    def test_empty_variables(self):
        assert safe_replace("{a} text", {}) == "{a} text"

    def test_no_placeholders(self):
        assert safe_replace("plain text", {"a": "1"}) == "plain text"


# ---------------------------------------------------------------------------
# strip_number_prefixes_from_block
# ---------------------------------------------------------------------------

class TestStripNumberPrefixesFromBlock:
    def test_empty_block(self):
        assert strip_number_prefixes_from_block([]) == []

    def test_strips_sequential_line_numbers(self):
        block = [
            "1 import os",
            "2 import sys",
            "3 def main():",
        ]
        result = strip_number_prefixes_from_block(block)
        assert result == ["import os", "import sys", "def main():"]

    def test_strips_with_dot_separator(self):
        block = ["1. import os", "2. import sys"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["import os", "import sys"]

    def test_strips_with_colon_separator(self):
        block = ["1: import os", "2: import sys"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["import os", "import sys"]

    def test_strips_with_tab_separator(self):
        block = ["1\timport os", "2\timport sys"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["import os", "import sys"]

    def test_numeric_literals_protected(self):
        # 1000 != position 1, 2000 != position 2, 3000 != position 3
        block = ["1000", "2000", "3000"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["1000", "2000", "3000"]

    def test_out_of_order_protected(self):
        block = ["5 something", "3 something", "1 something"]
        result = strip_number_prefixes_from_block(block)
        # None match their position, so nothing stripped
        assert result == ["5 something", "3 something", "1 something"]

    def test_single_line_not_stripped(self):
        # Need at least 2 position-matched lines
        block = ["1 only one line"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["1 only one line"]

    def test_bare_number_lines_stripped(self):
        block = ["1 import os", "2", "3 def main():"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["import os", "", "def main():"]

    def test_gap_in_numbering(self):
        # Each matched line stripped independently
        block = ["1 line one", "", "3 line three"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["line one", "", "line three"]

    def test_partial_match_only_strips_matched(self):
        block = ["1 matched", "2 matched", "999 not matched"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["matched", "matched", "999 not matched"]

    def test_leading_spaces_in_number(self):
        block = ["  1 import os", "  2 import sys"]
        result = strip_number_prefixes_from_block(block)
        assert result == ["import os", "import sys"]


# ---------------------------------------------------------------------------
# strip_inline_line_numbers
# ---------------------------------------------------------------------------

class TestStripInlineLineNumbers:
    def test_none_input(self):
        assert strip_inline_line_numbers(None) == ""

    def test_empty_string(self):
        assert strip_inline_line_numbers("") == ""

    def test_no_code_blocks_untouched(self):
        text = "This is prose.\nNo code here.\nJust text."
        assert strip_inline_line_numbers(text) == text

    def test_strips_numbers_in_fenced_block(self):
        text = "Some intro.\n\n```python\n1 import os\n2 import sys\n3 def main():\n```\n\nDone."
        result = strip_inline_line_numbers(text)
        assert "import os" in result
        assert "1 import os" not in result
        assert "import sys" in result
        assert "def main():" in result

    def test_mermaid_block_not_stripped(self):
        text = "```mermaid\n1 graph TD\n2 A --> B\n```"
        result = strip_inline_line_numbers(text)
        # Mermaid blocks should keep their content
        assert "1 graph TD" in result
        assert "2 A --> B" in result

    def test_unfenced_code_untouched(self):
        text = "1 some text\n2 more text"
        result = strip_inline_line_numbers(text)
        assert result == text

    def test_single_numbered_line_in_block_not_stripped(self):
        text = "```\n1 only one line\n```"
        result = strip_inline_line_numbers(text)
        # Single matched line -> not enough to confirm numbered block
        assert "1 only one line" in result

    def test_multiple_code_blocks(self):
        text = (
            "```python\n1 a = 1\n2 b = 2\n```\n"
            "prose\n"
            "```js\n1 let x = 1\n2 let y = 2\n```"
        )
        result = strip_inline_line_numbers(text)
        assert "a = 1" in result
        assert "1 a = 1" not in result
        assert "let x = 1" in result
        assert "1 let x = 1" not in result

    def test_unclosed_fence(self):
        text = "```python\n1 a = 1\n2 b = 2\n"
        result = strip_inline_line_numbers(text)
        # Should still process the block
        assert "a = 1" in result
        assert "1 a = 1" not in result

    def test_plain_lang_fence(self):
        text = "```\n1 line one\n2 line two\n```"
        result = strip_inline_line_numbers(text)
        assert "line one" in result
        assert "1 line one" not in result

    def test_empty_block(self):
        text = "```\n```"
        result = strip_inline_line_numbers(text)
        assert result == text

    def test_code_block_with_no_numbers(self):
        text = "```python\nimport os\nimport sys\n```"
        result = strip_inline_line_numbers(text)
        assert result == text

    def test_numeric_literals_in_code_protected(self):
        text = "```\n1000\n2000\n3000\n```"
        result = strip_inline_line_numbers(text)
        # 1000 != 1, 2000 != 2, 3000 != 3 -> not stripped
        assert "1000" in result
        assert "2000" in result
        assert "3000" in result


# ---------------------------------------------------------------------------
# Regex patterns (smoke tests)
# ---------------------------------------------------------------------------

class TestRegexPatterns:
    def test_line_num_prefix_re_matches(self):
        assert LINE_NUM_PREFIX_RE.match("1 import os")
        assert LINE_NUM_PREFIX_RE.match("  12. def f():")
        assert LINE_NUM_PREFIX_RE.match("3:  x = 1")
        assert LINE_NUM_PREFIX_RE.match("10\t# comment")

    def test_line_num_prefix_re_no_match(self):
        assert not LINE_NUM_PREFIX_RE.match("import os")
        assert not LINE_NUM_PREFIX_RE.match("# comment")

    def test_line_num_only_re_matches(self):
        assert LINE_NUM_ONLY_RE.match("2")
        assert LINE_NUM_ONLY_RE.match("2  ")
        assert LINE_NUM_ONLY_RE.match("3.")
        assert LINE_NUM_ONLY_RE.match("  5  ")

    def test_line_num_only_re_no_match(self):
        assert not LINE_NUM_ONLY_RE.match("2 import os")
        assert not LINE_NUM_ONLY_RE.match("import os")
