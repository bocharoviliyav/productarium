"""Expert agent public entry points + chat orchestration.

Split out of the former ``api/expert_agent.py`` (Step 6). Owns:
- ``run_expert_chat``: product-scoped conversational chat. Returns an async
  generator (``stream=True``) or a coroutine resolving to the full answer
  string (``stream=False``).
- ``run_expert_doc``: one-shot expert document (full self-contained Markdown).
- ``_run_expert_chat_collect`` / ``_run_expert_chat_stream``: the two chat
  orchestration cores (model + knowledge + history + prompt -> answer).

All expert-internal helpers are imported as top-level names so they are
patchable use-site globals: tests monkeypatch ``api.expert.chat._generate_answer``
etc. on THIS module and the calling functions here see the patched value
(they look deps up in chat.py's globals, not a facade's).

The prompt bodies (``EXPERT_SYSTEM_PROMPT`` / ``EXPERT_DOC_PROMPT``) are
referenced via the ``api.expert.prompt`` module at call time (NOT captured at
import) so ``api.prompts.reload_prompt_file`` can hot-reload them by reloading
``api.expert.prompt`` and the next call here picks up the new text.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from api.utils import setup_logging
from api.expert.generate import (
    _generate_answer,
    _stream_answer,
)
from api.expert.knowledge import (
    _format_history,
    _product_name_by_id,
    _retrieve_product_knowledge,
)
from api.expert.llm import _resolve_expert_model
from api.expert import prompt as _expert_prompt
from api.expert.prompt import _build_prompt
from api.expert.types import (
    EVENT_ANSWERING,
    EVENT_CONTENT,
    EVENT_RETRIEVING,
    EVENT_STATUS,
    EVENT_THINKING,
    ExpertStreamEvent,
)

setup_logging()
logger = logging.getLogger(__name__)


async def _run_expert_chat_collect(
    product_id: str,
    query: str,
    messages: List[Dict[str, Any]],
    model: Optional[str],
) -> str:
    """Non-streaming chat: returns the full answer string."""
    resolved_model, base_url, api_key = _resolve_expert_model(model)
    knowledge = await _retrieve_product_knowledge(product_id, query)
    history = _format_history(messages)
    # P1-13: resolve ctx window off-loop (may query live API metadata).
    try:
        from api.utils import get_model_context_window_async
        ctx_win = await get_model_context_window_async(
            base_url=base_url, model_name=resolved_model, api_key=api_key, task="expert"
        )
    except Exception:
        ctx_win = 8192
    prompt = _build_prompt(
        _expert_prompt.EXPERT_SYSTEM_PROMPT, _product_name_by_id(product_id), knowledge, history, query,
        base_url=base_url, model=resolved_model, api_key=api_key, ctx_win=ctx_win,
    )
    logger.info(
        "Expert chat (collect) product=%s prompt_chars=%d",
        product_id,
        len(prompt),
    )
    return await _generate_answer(
        prompt, resolved_model, base_url, api_key,
        product_id=product_id,
        product_name=_product_name_by_id(product_id),
        query=query,
        history=history,
    )


async def _run_expert_chat_stream(
    product_id: str,
    query: str,
    messages: List[Dict[str, Any]],
    model: Optional[str],
) -> AsyncIterator[ExpertStreamEvent]:
    """Streaming chat: yields status events then answer chunks.

    Emits status events so the frontend can show a phase-aware loader:
    - ``("status", "retrieving")`` before knowledge retrieval.
    - ``("status", "thinking")`` before the first answer chunk.
    - ``("status", "answering")`` when content starts flowing.

    Yields ``ExpertStreamEvent`` objects (status / reasoning / content).
    """
    yield ExpertStreamEvent(EVENT_STATUS, EVENT_RETRIEVING)
    resolved_model, base_url, api_key = _resolve_expert_model(model)
    knowledge = await _retrieve_product_knowledge(product_id, query)
    history = _format_history(messages)
    # P1-13: resolve ctx window off-loop (may query live API metadata).
    try:
        from api.utils import get_model_context_window_async
        ctx_win = await get_model_context_window_async(
            base_url=base_url, model_name=resolved_model, api_key=api_key, task="expert"
        )
    except Exception:
        ctx_win = 8192
    prompt = _build_prompt(
        _expert_prompt.EXPERT_SYSTEM_PROMPT, _product_name_by_id(product_id), knowledge, history, query,
        base_url=base_url, model=resolved_model, api_key=api_key, ctx_win=ctx_win,
    )
    logger.info(
        "Expert chat (stream) product=%s prompt_chars=%d",
        product_id,
        len(prompt),
    )
    yield ExpertStreamEvent(EVENT_STATUS, EVENT_THINKING)
    content_started = False
    async for event in _stream_answer(
        prompt, resolved_model, base_url, api_key,
        product_id=product_id,
        product_name=_product_name_by_id(product_id),
        query=query,
        history=history,
    ):
        # Emit the "answering" status just before the first content chunk
        # (after any reasoning events have been sent).
        if not content_started and event.type == EVENT_CONTENT:
            content_started = True
            yield ExpertStreamEvent(EVENT_STATUS, EVENT_ANSWERING)
        yield event


def run_expert_chat(
    product_id: str,
    query: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    stream: bool = True,
):
    """Product-scoped expert chat.

    Args:
        product_id: The product to ground the answer in.
        query: The current user question.
        messages: Prior conversation history as ``[{"role": "user"|"assistant",
            "content": "..."}]`` (the current query is passed separately).
        model: Optional model override; otherwise resolved from the admin
            ``models.expert`` config (with env fallbacks).
        stream: When True, returns an async generator yielding text chunks.
            When False, returns a coroutine resolving to the full answer string.

    Returns:
        An async generator (``stream=True``) or a coroutine (``stream=False``).
    """
    msgs = list(messages or [])
    if stream:
        return _run_expert_chat_stream(product_id, query, msgs, model)
    return _run_expert_chat_collect(product_id, query, msgs, model)
