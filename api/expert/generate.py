"""Expert answer generation (RLM long-context -> standard-LLM fallback).

Split out of the former ``api/expert_agent.py`` (Step 6). Owns:
- ``_resolve_use_rlm``: resolve the final RLM flag from an optional override +
  the admin ``rlm.<task>.mode`` setting.
- ``_rlm_generate``: run fast-rlm (capped by the per-section RLM expert
  timeout) and return cleaned text, or "" on any failure/empty result.
- ``_generate_answer``: non-streaming answer (RLM -> standard LLM -> "").
- ``_stream_answer``: streaming answer (RLM chunked -> standard LLM stream).

Imports ``_safe_build_llm`` from ``api.expert.llm`` and ``_clean_llm_text`` /
``_chunk_text`` / ``RLM_MIN_CHARS`` from ``api.expert.prompt`` as top-level
names so they are patchable use-site globals (tests monkeypatch these on the
module where the calling function looks them up, not on a facade).
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from api.utils import setup_logging
from api.expert.llm import _safe_build_llm
from api.expert.prompt import (
    RLM_MIN_CHARS,
    _chunk_text,
    _clean_llm_text,
)
from api.expert.types import (
    EVENT_CONTENT,
    ExpertStreamEvent,
)

setup_logging()
logger = logging.getLogger(__name__)


def _resolve_use_rlm(
    use_rlm: Optional[bool], task: str, prompt_len: int
) -> bool:
    """Resolve the final RLM flag from an optional override + the admin mode.

    Precedence:
    - ``use_rlm=True``  -> always RLM (caller forced it; LLM fallback still fires
      if RLM fails/times out).
    - ``use_rlm=False`` -> never RLM (caller forced the standard LLM).
    - ``use_rlm=None``  -> follow the admin ``rlm.<task>.mode`` setting:
        * ``llm``  -> False
        * ``rlm``  -> True
        * ``auto`` -> True only when the prompt is large (>= RLM_MIN_CHARS).
    This keeps "LLM is the guaranteed baseline; RLM is opt-in" regardless of
    how the mode is configured: ``get_rlm_mode`` already returns ``llm`` when
    fast-rlm is not installed, so ``None`` never accidentally enables RLM.
    """
    if use_rlm is True:
        return True
    if use_rlm is False:
        return False
    try:
        from api.settings_store import get_rlm_mode
        mode = get_rlm_mode(task)
    except Exception:  # pragma: no cover - settings store import-safe
        mode = "auto"
    if mode == "llm":
        return False
    if mode == "rlm":
        return True
    return prompt_len >= RLM_MIN_CHARS


async def _rlm_generate(prompt: str, model: Optional[str]) -> str:
    """Run fast-rlm and return cleaned text, or "" on any failure/empty result.

    Capped by the per-section RLM expert timeout (resolved at call time via the
    central timeout config: admin > env > default) so a hung fast-rlm
    (first-run bootstrap, dead Pyodide worker, a local model that never
    returns) cannot hold the expert SSE stream open indefinitely. On timeout we
    returns "" and the caller falls back to the standard LLM.
    """
    try:
        from api.timeout_config import resolve_rlm_expert_timeout
        rlm_expert_timeout = resolve_rlm_expert_timeout()
    except Exception:
        rlm_expert_timeout = 1800.0
    try:
        from api.utils import get_model_context_window, clamp_text_by_tokens
        ctx_win = get_model_context_window(model_name=model, task="expert")
        completion_res = max(1024, min(4096, ctx_win // 4))
        max_prompt_limit = max(1024, ctx_win - completion_res)
        safe_prompt_limit = max(2000, max_prompt_limit - 6000)
        safe_prompt = clamp_text_by_tokens(prompt, safe_prompt_limit)

        from api.rlm.runner import run_rlm_task  # lazy: fast_rlm is optional

        res = await asyncio.wait_for(
            run_rlm_task(safe_prompt, model), timeout=rlm_expert_timeout
        )
        if isinstance(res, dict) and res.get("success") and res.get("results"):
            return _clean_llm_text(str(res["results"]))
        logger.warning("RLM returned no usable result for expert task.")
    except asyncio.TimeoutError:
        logger.warning(
            "RLM expert generation timed out after %ss; falling back to standard LLM.",
            rlm_expert_timeout,
        )
    except Exception as e:  # pragma: no cover - depends on live fast-rlm
        logger.warning("RLM expert generation failed: %s", e)
    return ""


async def _generate_answer(
    prompt: str,
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    use_rlm: bool,
) -> str:
    """Non-streaming answer: RLM (long context) -> standard LLM fallback -> ""."""
    if use_rlm:
        text = await _rlm_generate(prompt, model)
        if text:
            return text
        logger.warning("Expert RLM path empty; falling back to standard LLM.")
    llm = _safe_build_llm(provider, model, base_url=base_url, api_key=api_key)
    if llm is None:
        return ""
    try:
        return _clean_llm_text(await llm.generate(prompt))
    except Exception as e:  # pragma: no cover - depends on live Ollama
        logger.warning("Expert standard LLM generate failed: %s", e)
        return ""


async def _stream_answer(
    prompt: str,
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    use_rlm: bool,
) -> AsyncIterator[ExpertStreamEvent]:
    """Streaming answer: RLM (chunked) -> standard LLM stream (with fallback).

    Yields ``ExpertStreamEvent`` objects so the router can emit typed SSE
    frames (status / reasoning / content). The RLM path wraps each text
    piece as a ``content`` event; the standard LLM path passes through the
    events from ``_ExpertLLM.stream()`` (which may include ``reasoning``
    events for thinking-capable models).
    """
    if use_rlm:
        text = await _rlm_generate(prompt, model)
        if text:
            for piece in _chunk_text(text):
                yield ExpertStreamEvent(EVENT_CONTENT, piece)
            return
        logger.warning("Expert RLM stream empty; falling back to standard LLM.")
    llm = _safe_build_llm(provider, model, base_url=base_url, api_key=api_key)
    if llm is None:
        return
    async for event in llm.stream(prompt):
        yield event
