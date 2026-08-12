"""Shared LLM text helpers (dedup across docgen / expert / summary).

Only genuinely-IDENTICAL helpers live here. The three LLM wrapper classes
(``_StandardLLM`` / ``_ExpertLLM`` / ``_SummaryLLM``) are NOT identical
(retry vs streaming vs simple) and stay in their domain packages. Likewise
``_clean_llm_text`` differs between modules (expert strips ``<r>`` blocks) so
each module keeps its own.

Moved here so the cross-module ``from api.artifact_docgen import
_strip_inline_line_numbers`` in expert_agent did not break when artifact_docgen
was split into the ``api/docgen/`` package (Step 4), and so the
prompt-substitution helper is defined once, not three times.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def safe_replace(template: str, variables: Dict[str, Any]) -> str:
    """Substitute ``{var}`` placeholders in ``template`` using exact replacement.

    Unmatched placeholders are left intact (so they remain visible rather than
    silently disappearing), matching the behaviour of
    ``WikiGenerator._format_prompt``.
    """
    if not template:
        return ""
    out = template
    for key, value in variables.items():
        out = out.replace("{" + key + "}", "" if value is None else str(value))
    return out


def cap(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (обрезано для контекста LLM)\n"


# Regex for a leading line-number prefix on a code line: optional spaces, then
# digits, then an optional separator (spaces, '.', ':' or a tab), then the rest.
# Matches "1 import os", "  12. def f():", "3:  x = 1", "10\t# comment".
LINE_NUM_PREFIX_RE = re.compile(r"^[ \t]*\d+[ \t]*[:.]?[ \t]+")
# A line that is ONLY a number (optionally with a separator/trailing spaces) —
# i.e. a line number for a blank code line, e.g. "2", "2  ", "3.". Stripped only
# inside a confirmed numbered block (see strip_number_prefixes_from_block) so
# standalone numeric-literal blocks are not mangled.
LINE_NUM_ONLY_RE = re.compile(r"^[ \t]*\d+[ \t]*[:.]?[ \t]*$")


def strip_number_prefixes_from_block(block: List[str]) -> List[str]:
    """Remove leading line-number prefixes from a single code block's lines.

    A line is de-numbered only when its leading number equals its 1-indexed
    position in the block (``val == idx + 1``) -- the signature of LLM-emitted
    line numbers that start at 1. This single rule naturally protects:
      * numeric literals (``1000``/``2000``/``3000`` never equal their position),
      * out-of-order numbers (``5``/``3``/``1`` never equal ``1``/``2``/``3``),
      * large numbers (an excerpt numbered ``5``/``6``/``7`` is left intact -- a
        conservative trade-off since it is indistinguishable from numeric data).
    At least 2 position-matched lines are required before any stripping happens,
    so an isolated numbered line is left alone. Gaps are handled gracefully: each
    matched line is stripped independently, so an unnumbered blank line in the
    middle of a numbered block does not prevent the rest from being cleaned.
    """
    if not block:
        return block
    # Collect lines whose leading number equals their 1-indexed position.
    content_matches: List[int] = []  # idx of content-bearing matched lines
    bare_matches: List[int] = []     # idx of bare-number matched lines
    for idx, ln in enumerate(block):
        num_match = re.match(r"^[ \t]*(\d+)", ln)
        if not num_match:
            continue
        val = int(num_match.group(1))
        if val != idx + 1:
            continue
        if LINE_NUM_ONLY_RE.match(ln):
            bare_matches.append(idx)
        elif LINE_NUM_PREFIX_RE.match(ln):
            content_matches.append(idx)
    # Need at least 2 position-matched numbered lines to confirm a numbered block.
    if len(content_matches) + len(bare_matches) < 2:
        return block
    out = list(block)
    for idx in content_matches:
        out[idx] = LINE_NUM_PREFIX_RE.sub("", out[idx], count=1)
    for idx in bare_matches:
        out[idx] = ""
    return out


def strip_inline_line_numbers(text: Optional[str]) -> str:
    """Strip leading ``N``/``N.``/``N:`` prefixes from lines INSIDE fenced code
    blocks only.

    The UI's ``SyntaxHighlighter`` already renders line numbers via
    ``showLineNumbers``; when an LLM ALSO emits ``1 import os`` prefixes the
    numbers are duplicated/ugly. This post-processor removes them as a safety
    net on top of the prompt rule (which asks the model not to emit them).

    Only fenced code blocks (``` ... ```) are touched: prose, Mermaid diagrams
    (which are their own fenced lang) and already-clean code are left intact.
    A line is only de-numbered when its leading number equals the line's
    1-indexed position in the block (the signature of LLM-emitted line numbers
    that start at 1) and at least one sibling line shares that property, so
    legitimate code that happens to begin with a number (e.g. a numeric literal
    on the first line) is not mangled. Mermaid blocks are skipped explicitly
    (their content is not code).
    """
    if not text:
        return text or ""
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        # A fenced block opener: ```lang  (lang may be empty or ````mermaid````).
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            is_mermaid = lang == "mermaid"
            out.append(line)
            i += 1
            # Collect the block body until the closing fence.
            block: List[str] = []
            while i < n and not lines[i].lstrip().startswith("```"):
                block.append(lines[i])
                i += 1
            if not is_mermaid and block:
                block = strip_number_prefixes_from_block(block)
            out.extend(block)
            # The closing fence (if present) — append as-is.
            if i < n:
                out.append(lines[i])
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)
