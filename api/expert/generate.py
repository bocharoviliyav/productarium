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
        from api.config.settings import get_rlm_mode
        mode = get_rlm_mode(task)
    except Exception:  # pragma: no cover - settings store import-safe
        mode = "auto"
    if mode == "llm":
        return False
    if mode == "rlm":
        return True
    return prompt_len >= RLM_MIN_CHARS


async def _rlm_generate(
    prompt: str,
    model: Optional[str],
    product_id: Optional[str] = None,
    product_name: Optional[str] = None,
    query: Optional[str] = None,
    history: Optional[str] = None,
) -> str:
    """Run fast-rlm and return cleaned text, or "" on any failure/empty result.

    Two paths:
    - **Structured path** (when ``product_id`` is supplied): calls
      ``run_rlm_structured`` with a dict query, the expert retrieval tools, and
      a per-product session. Knowledge is NOT stringified into the prompt — the
      agent pulls it on demand via the tools (exhaustive recursive search).
      Falls back to the flat-string path if the structured call rejects the
      new kwargs (older fast-rlm) or fails.
    - **Flat-string path** (no ``product_id``, or structured fallback): the
      legacy ``run_rlm_task(prompt)`` with the knowledge already in the prompt.

    Both are capped by the per-section RLM expert timeout (resolved at call time
    via the central timeout config: admin > env > default) so a hung fast-rlm
    cannot hold the expert SSE stream open indefinitely. On timeout we return
    "" and the caller falls back to the standard LLM.
    """
    try:
        from api.config.timeout import resolve_rlm_expert_timeout
        rlm_expert_timeout = resolve_rlm_expert_timeout()
    except Exception:
        rlm_expert_timeout = 1800.0
    try:
        from api.utils import get_model_context_window
        ctx_win = get_model_context_window(model_name=model, task="expert")
        completion_res = max(1024, min(4096, ctx_win // 4))
        max_prompt_limit = max(1024, ctx_win - completion_res)
        safe_prompt_limit = max(2000, max_prompt_limit - 6000)
        # RLM is a long-context engine: pass the full prompt when it fits the
        # model's context window. When it overflows, char-cap to the token
        # budget (4 chars/token heuristic) so a single RLM call cannot blow
        # max_prompt_tokens. The expert prompt is already budget-capped in
        # _build_prompt, so this only fires for very large knowledge blocks.
        safe_prompt = prompt
        safe_prompt_chars = safe_prompt_limit * 4
        if len(prompt) > safe_prompt_chars:
            safe_prompt = prompt[:safe_prompt_chars] + "\n... (контекст обрезан для RLM)"

        from api.rlm.runner import run_rlm_structured, run_rlm_task, get_rlm_session_dir

        # Structured path: dict query + tools + per-product session. The agent
        # fetches knowledge itself via the tools, so the dict carries the task,
        # product name, query, and history (NOT the stringified knowledge).
        if product_id:
            try:
                from api.rlm.tools import build_expert_tools, resolve_env_variables

                dict_query = {
                    "task": prompt,
                    "product": product_name or product_id,
                    "query": query or "",
                    "history": history or "",
                }
                tools = build_expert_tools()
                env_vars = resolve_env_variables(product_id)
                session_dir = get_rlm_session_dir(f"expert_{product_id}")
                res = await asyncio.wait_for(
                    run_rlm_structured(
                        dict_query,
                        model_name=model,
                        tools=tools,
                        env_variables=env_vars,
                        session_dir=session_dir,
                        session_id=product_id,
                        task="expert",
                    ),
                    timeout=rlm_expert_timeout,
                )
                if isinstance(res, dict) and res.get("success") and res.get("results"):
                    return _clean_llm_text(str(res["results"]))
                # Structured call failed/rejected kwargs -> fall through to the
                # flat-string path below so the run still produces an answer.
                if isinstance(res, dict) and not res.get("success"):
                    logger.info(
                        "RLM structured path did not succeed (%s); trying flat-string path.",
                        str(res.get("results"))[:120],
                    )
            except Exception as e:  # pragma: no cover - depends on live fast-rlm
                logger.warning(
                    "RLM structured expert path failed (%s); falling back to flat-string.",
                    e,
                )

        # Flat-string path (legacy, or structured fallback).
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
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    use_rlm: bool,
    product_id: Optional[str] = None,
    product_name: Optional[str] = None,
    query: Optional[str] = None,
    history: Optional[str] = None,
) -> str:
    """Non-streaming answer: RLM (long context) -> standard LLM fallback -> ""."""
    if use_rlm:
        text = await _rlm_generate(
            prompt, model,
            product_id=product_id,
            product_name=product_name,
            query=query,
            history=history,
        )
        if text:
            return text
        logger.warning("Expert RLM path empty; falling back to standard LLM.")
    llm = _safe_build_llm(model, base_url=base_url, api_key=api_key)
    if llm is None:
        return ""
    try:
        return _clean_llm_text(await llm.generate(prompt))
    except Exception as e:  # pragma: no cover - depends on live LLM
        logger.warning("Expert standard LLM generate failed: %s", e)
        return ""


async def _stream_answer(
    prompt: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    use_rlm: bool,
    product_id: Optional[str] = None,
    product_name: Optional[str] = None,
    query: Optional[str] = None,
    history: Optional[str] = None,
) -> AsyncIterator[ExpertStreamEvent]:
    """Streaming answer: RLM (chunked) -> standard LLM stream (with fallback).

    Yields ``ExpertStreamEvent`` objects so the router can emit typed SSE
    frames (status / reasoning / content). The RLM path wraps each text
    piece as a ``content`` event; the standard LLM path passes through the
    events from ``_ExpertLLM.stream()`` (which may include ``reasoning``
    events for thinking-capable models).
    """
    if use_rlm:
        text = await _rlm_generate(
            prompt, model,
            product_id=product_id,
            product_name=product_name,
            query=query,
            history=history,
        )
        if text:
            for piece in _chunk_text(text):
                yield ExpertStreamEvent(EVENT_CONTENT, piece)
            return
        logger.warning("Expert RLM stream empty; falling back to standard LLM.")
    llm = _safe_build_llm(model, base_url=base_url, api_key=api_key)
    if llm is None:
        return
    async for event in llm.stream(prompt):
        yield event
