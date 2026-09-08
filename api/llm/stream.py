"""Token streaming (content + reasoning deltas) over ChatOpenAI.

Replaces the former ``OpenAIClient.acall(stream=True)`` +
``_extract_chunk_fields`` adalflow path. The typing-tag parser
(``_ThinkingStreamParser``) and chunk-field extraction stay in
``api/expert/llm.py`` untouched — they operate on OpenAI-compatible chunk
shapes, not adalflow types.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional, Tuple

from api.llm.client import build_chat_model

logger = logging.getLogger(__name__)


async def stream_chat(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream answer-text deltas for ``prompt`` from the configured LLM.

    Yields plain text deltas (inline ``<think>`` tags NOT parsed here — the
    expert layer owns that). Raises on transport errors; callers implement
    their own fallbacks (e.g. non-streaming generate + chunked delivery).
    """
    if not prompt:
        return
    from langchain_core.messages import HumanMessage

    chat = build_chat_model(model=model, base_url=base_url, api_key=api_key)
    async for chunk in chat.astream([HumanMessage(content=prompt)]):
        text = _chunk_text(chunk)
        if text:
            yield text


def _chunk_text(chunk: Any) -> str:
    """Extract the content delta from a streamed AIMessageChunk."""
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


async def stream_chat_fields(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AsyncIterator[Tuple[str, str]]:
    """Stream ``(content, reasoning)`` delta pairs from the configured LLM.

    Yields tuples where either element may be empty (never both empty). The
    reasoning delta is read from the OpenAI-compatible ``reasoning_content``
    / ``reasoning`` / ``thinking`` fields (DeepSeek / vLLM / Qwen shapes);
    everything else is content. This keeps the expert SSE stream contract
    (``{"reasoning": ...}`` / ``{"content": ...}`` typed frames) intact.
    """
    if not prompt:
        return
    from langchain_core.messages import HumanMessage

    chat = build_chat_model(model=model, base_url=base_url, api_key=api_key)
    async for chunk in chat.astream([HumanMessage(content=prompt)]):
        content, reasoning = _chunk_fields(chunk)
        if content or reasoning:
            yield content, reasoning


def _chunk_fields(chunk: Any) -> Tuple[str, str]:
    """Extract ``(content, reasoning)`` text deltas from a streamed chunk.

    LangChain's AIMessageChunk exposes the raw OpenAI delta fields via
    ``additional_kwargs`` (and the base ``content``), so the same shapes the
    former ``_extract_chunk_fields`` handled are covered here.
    """
    content = _chunk_text(chunk)
    reasoning = ""
    additional = getattr(chunk, "additional_kwargs", None) or {}
    if isinstance(additional, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            val = additional.get(key)
            if isinstance(val, str) and val:
                reasoning = val
                break
    if not reasoning:
        # Some servers attach reasoning on the response-metadata payload.
        meta = getattr(chunk, "response_metadata", None) or {}
        if isinstance(meta, dict):
            val = meta.get("reasoning_content") or meta.get("reasoning")
            if isinstance(val, str) and val:
                reasoning = val
    return content, reasoning
