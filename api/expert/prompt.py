"""Expert prompt assembly + text cleaning + tunables + loaded prompt bodies.

Split out of the former ``api/expert_agent.py`` (Step 6). Owns:
- Tunables: ``RLM_MIN_CHARS``, ``KNOWLEDGE_MAX_CHARS``, ``STREAM_CHUNK_SIZE``,
  ``_DEFAULT_LANGUAGE_NAME``.
- Loaded prompt bodies: ``EXPERT_SYSTEM_PROMPT`` / ``EXPERT_DOC_PROMPT``
  (from ``refs/prompts/expert_agent_*.md`` via ``api.prompts.load_prompt_file``).
- ``_clean_llm_text`` (expert variant — also strips ``<r>...</r>`` blocks that
  some local models emit; differs from the docgen variant in
  ``api.docgen._common``, so each module keeps its own).
- ``_chunk_text``: split text into incremental pieces for chunked streaming.
- ``_build_prompt``: assemble the full expert prompt from a loaded template,
  passing full knowledge/history when they fit the context window and
  char-capping each to its budget share when they overflow.

``_safe_replace`` / ``cap`` / ``strip_inline_line_numbers`` are shared
(generic text helpers) and imported from ``api.utils.llm_helpers``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from api.utils import setup_logging
from api.utils.llm_helpers import (  # noqa: E402
    safe_replace as _safe_replace,
    cap as _cap,
    strip_inline_line_numbers as _strip_inline_line_numbers,
)
from api.prompts import load_prompt_file

setup_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
# RLM is for long context only (per the plan / api.docgen.codebase). Below this
# combined-prompt size we use the standard LLM directly.
RLM_MIN_CHARS = 20_000
# Cap the knowledge block injected into the prompt so very large products don't
# blow the LLM context window. RLM (when triggered) still receives the full
# prompt; this cap keeps the standard-LLM path manageable.
KNOWLEDGE_MAX_CHARS = 60_000
# Chunk size used when streaming a non-streaming source (RLM, or standard-LLM
# fallback) so the client still receives incremental SSE chunks.
STREAM_CHUNK_SIZE = 80

# Loaded once at import; the .md files are the source of truth.
EXPERT_SYSTEM_PROMPT: str = load_prompt_file("expert_agent_system.md", "")
EXPERT_DOC_PROMPT: str = load_prompt_file("expert_agent_doc.md", "")

# Default language instruction substituted into {language_name}. The expert
# agent follows the user's language rather than a fixed one.
_DEFAULT_LANGUAGE_NAME = "the same language as the user's query (keep code identifiers, file paths, and API names in English)"


# --------------------------------------------------------------------------- #
# Text cleaning (expert variant — strips <r>...</r> blocks too)
# --------------------------------------------------------------------------- #
def _clean_llm_text(text: Optional[str]) -> str:
    """Strip surrounding whitespace, a wrapping ```markdown fence, <r> blocks,
    and any inline line-number prefixes the LLM emitted inside code blocks."""
    if not text:
        return ""
    t = text.strip()
    # Remove a leading ```lang fence and a trailing ``` if present.
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3]
    # Some local models emit  <r>... </r> blocks; strip them.
    t = re.sub(r" <r>.*?</r>", "", t, flags=re.DOTALL).strip()
    # Strip inline line-number prefixes inside code blocks (the UI renders its
    # own line numbers; duplicated prefixes are ugly). Shared implementation.
    t = _strip_inline_line_numbers(t)
    return t.strip()


def _chunk_text(text: str, size: int = STREAM_CHUNK_SIZE) -> List[str]:
    """Split ``text`` into incremental pieces for chunked streaming delivery.

    Splits on newlines first (so line structure survives), then further splits
    over-long lines on word boundaries near ``size``.
    """
    if not text:
        return []
    pieces: List[str] = []
    for line in text.splitlines(keepends=True):
        if len(line) <= size:
            pieces.append(line)
            continue
        words = line.split(" ")
        buf = ""
        for w in words:
            candidate = w if not buf else buf + " " + w
            if len(candidate) > size and buf:
                pieces.append(buf)
                buf = w
            else:
                buf = candidate
        if buf:
            pieces.append(buf)
    return pieces


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def _build_prompt(
    template: str,
    product_name: str,
    knowledge: str,
    history: str,
    query: str,
    language_name: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Assemble the full expert prompt from a loaded template body.

    The template (from refs/prompts/*.md) carries role/guidelines and the
    ``{product_name}`` / ``{language_name}`` placeholders. Knowledge, history,
    and the query are appended as structured blocks (NOT substituted into the
    template) so the template body stays small and Mermaid/JSON-safe.

    Knowledge and history are passed in full when they fit the model's context
    window (RLM benefits from the full context); when they overflow, each is
    char-capped to its share of the budget so the standard-LLM path still gets
    a usable prompt instead of blowing the context.
    """
    system = _safe_replace(
        template,
        {
            "product_name": product_name or "this product",
            "language_name": language_name or _DEFAULT_LANGUAGE_NAME,
        },
    )
    try:
        from api.prompts import VERIFICATION_GUARD as _guard
    except Exception:  # pragma: no cover - import-safe
        _guard = ""
    if _guard:
        system = system + "\n\n" + _guard

    # Resolve the model's context window so the standard-LLM path stays in
    # budget. RLM is a long-context engine and ignores this cap (it receives
    # the full prompt); the cap only protects the standard-LLM fallback.
    try:
        from api.utils import get_model_context_window, _count_tokens
        ctx_win = get_model_context_window(base_url=base_url, model_name=model, api_key=api_key, task="expert")
    except Exception:
        ctx_win = 8192
        from api.utils import _count_tokens

    # Reserve tokens for system instructions, query, and LLM output completion.
    avail_tokens = max(1024, ctx_win - 2048)
    # Approximate chars-per-token for the char cap fallback (4 is the standard
    # heuristic used by _count_tokens' own fallback).
    avail_chars = avail_tokens * 4

    # 1. History gets at most 1/3 of the budget (keep the tail: most recent
    #    turns are most relevant).
    history_budget_chars = max(2048, avail_chars // 3)
    clamped_history = ""
    if history:
        clamped_history = history if len(history) <= history_budget_chars else (
            "... (ранняя история обрезана для контекста)\n" + history[-history_budget_chars:]
        )

    # 2. Knowledge gets the remaining budget (keep the head: knowledge is
    #    retrieved top-first by cognee, so the most relevant docs come first).
    knowledge_budget_chars = max(2048, avail_chars - len(clamped_history))
    clamped_knowledge = ""
    if knowledge:
        clamped_knowledge = knowledge if len(knowledge) <= knowledge_budget_chars else (
            knowledge[:knowledge_budget_chars] + "\n... (часть знаний обрезана для контекста)"
        )

    prompt = system + "\n\n"
    if clamped_history:
        prompt += f"<conversation_history>\n{clamped_history}\n</conversation_history>\n\n"
    if clamped_knowledge:
        prompt += f"<product_knowledge>\n{clamped_knowledge}\n</product_knowledge>\n\n"
    else:
        prompt += (
            "<note>No indexed product knowledge was available. Answer honestly: "
            "say the knowledge is missing and suggest indexing artifacts.</note>\n\n\n"
        )
    prompt += f"<query>\n{query}\n</query>\n\nAssistant: "
    return prompt
