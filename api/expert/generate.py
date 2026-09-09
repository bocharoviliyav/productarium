"""Expert answer generation over the standard LLM.

Split out of the former ``api/expert_agent.py`` (Step 6). Owns:
- ``_generate_answer``: non-streaming answer (standard LLM -> "").
- ``_stream_answer``: streaming answer over the expert LLM stream.

The former fast-rlm long-context path was removed with the LangChain
migration: the expert agent is a plain ChatOpenAI call (a LangGraph agent
lands with Wave B). Imports ``_safe_build_llm`` from ``api.expert.llm`` and
``_clean_llm_text`` from ``api.expert.prompt`` as top-level names so they
remain patchable use-site globals (tests monkeypatch these on the module
where the calling function looks them up, not on a facade).
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from api.utils import setup_logging
from api.utils.llm_helpers import aclose_llm as _aclose_llm
from api.expert.llm import _safe_build_llm
from api.expert.prompt import _clean_llm_text
from api.expert.types import ExpertStreamEvent

setup_logging()
logger = logging.getLogger(__name__)


async def _generate_answer(
    prompt: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    product_id: Optional[str] = None,
    product_name: Optional[str] = None,
    query: Optional[str] = None,
    history: Optional[str] = None,
) -> str:
    """Non-streaming answer over the standard LLM; ``""`` on failure."""
    llm = _safe_build_llm(model, base_url=base_url, api_key=api_key)
    if llm is None:
        return ""
    try:
        return _clean_llm_text(await llm.generate(prompt))
    except Exception as e:  # pragma: no cover - depends on live LLM
        logger.warning("Expert standard LLM generate failed: %s", e)
        return ""
    finally:
        # P1-14: release httpx pools. Duck-typed close — ``_safe_build_llm`` is
        # a patch point and may return objects without ``aclose``.
        await _aclose_llm(llm)


async def _stream_answer(
    prompt: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    product_id: Optional[str] = None,
    product_name: Optional[str] = None,
    query: Optional[str] = None,
    history: Optional[str] = None,
) -> AsyncIterator[ExpertStreamEvent]:
    """Streaming answer over the expert LLM stream.

    Yields ``ExpertStreamEvent`` objects so the router can emit typed SSE
    frames (status / reasoning / content), passing through the events from
    ``_ExpertLLM.stream()`` (which may include ``reasoning`` events for
    thinking-capable models and falls back to chunked non-streaming delivery).
    """
    llm = _safe_build_llm(model, base_url=base_url, api_key=api_key)
    if llm is None:
        return
    try:
        async for event in llm.stream(prompt):
            yield event
    finally:
        # P1-14: release httpx pools (also on early SSE disconnect). Duck-typed
        # close — ``_safe_build_llm`` is a patch point; never raise from a
        # generator's finally (it would mask the generation result).
        await _aclose_llm(llm)
