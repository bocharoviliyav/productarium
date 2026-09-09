"""Non-streaming text generation with backoff retry over ChatOpenAI.

Replaces the adalflow ``Generator`` + ``OpenAIClient.call`` retry stack with a
thin async wrapper around :class:`langchain_openai.ChatOpenAI`. Transient
errors (timeouts, 5xx, rate limits, unprocessable/bad-request payloads) are
retried with backoff within the central ``llm_retry_max_time`` budget
(admin > env > default; default 900s, floor 30s) — the same policy the
former ``OpenAIClient`` applied via ``backoff.on_exception``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from api.llm.client import build_chat_model

logger = logging.getLogger(__name__)

# Transient errors worth retrying, matched on the exception class name. Kept
# as names (not imported classes) so this module imports cleanly even when
# langchain-openai internals reorganize their exception hierarchy.
_RETRYABLE_ERRORS = (
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "UnprocessableEntityError",
    "APIConnectionError",
)


def _is_retryable(exc: BaseException) -> bool:
    """True for transient OpenAI-compatible API errors worth retrying."""
    # Walk the exception chain (langchain wraps SDK errors in its own types).
    seen = 0
    current: Optional[BaseException] = exc
    while current is not None and seen < 5:
        if type(current).__name__ in _RETRYABLE_ERRORS:
            return True
        current = current.__cause__ or current.__context__
        seen += 1
    # Fall back to message sniffing for wrapped SDK errors.
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "rate limit", "too many requests", "timeout", "timed out")
    )


class GenerateLLM:
    """Non-streaming text generator with backoff retry over ChatOpenAI.

    The former adalflow-based wrappers (``_ExpertLLM`` /
    ``_StandardLLM`` / ``_SummaryLLM``) each rebuilt a Generator per call
    site; this single class covers their shared contract: ``await
    generate(prompt) -> str`` returning the answer text or ``""`` on failure
    (never raising into callers — they surface graceful fallbacks).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        **overrides: Any,
    ) -> None:
        self._chat = build_chat_model(
            model=model,
            base_url=base_url,
            api_key=api_key,
            **overrides,
        )
        self._max_retries = max(1, int(max_retries))

    async def aclose(self) -> None:
        """Close the per-instance httpx async client (best-effort, idempotent).

        ``build_chat_model`` opens a dedicated ``httpx.AsyncClient`` per
        instance; callers that build a generator per call site (docgen judge,
        spec enrichment, product summary) MUST close it to avoid leaking a
        connection pool per generation run.
        """
        client = getattr(self._chat, "http_async_client", None)
        if client is None:
            return
        try:
            await client.aclose()
        except Exception:
            logger.debug("GenerateLLM: could not close httpx client", exc_info=True)

    async def generate(self, prompt: str) -> str:
        """Generate an answer for ``prompt``; ``""`` on any failure.

        A single HumanMessage is sent; the answer text is returned verbatim
        (stripping of markdown fences / line numbers is the caller's concern).
        Rate-limit / transient failures are retried with linear backoff.
        """
        if not prompt:
            return ""
        from langchain_core.messages import HumanMessage

        last_exc: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            try:
                result = await self._chat.ainvoke([HumanMessage(content=prompt)])
                text = _message_text(result)
                if text:
                    return text
                logger.warning(
                    "GenerateLLM: empty response on attempt %d/%d.",
                    attempt + 1,
                    self._max_retries,
                )
            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < self._max_retries - 1:
                    backoff = (attempt + 1) * 2.5
                    logger.warning(
                        "GenerateLLM: transient error (attempt %d/%d); "
                        "sleeping %.1fs: %s",
                        attempt + 1,
                        self._max_retries,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
                    continue
                logger.warning("GenerateLLM: generation failed: %s", exc)
                return ""
        if last_exc is not None:
            logger.warning("GenerateLLM: exhausted retries: %s", last_exc)
        return ""


def _message_text(message: Any) -> str:
    """Extract the answer text from a langchain AIMessage-like object."""
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal content blocks: concatenate the text parts.
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content) if content else ""
