"""Expert LLM wrapper + factory + chunk-text extraction + model resolution.

Split out of the former ``api/expert_agent.py`` (Step 6). Owns the
``_ExpertLLM`` adalflow Generator wrapper (non-streaming ``generate`` + async
``stream`` with chunked fallback), the ``_safe_build_llm`` graceful-degrade
factory, ``_extract_chunk_fields`` (native + OpenAI
streaming-chunk shape handling including reasoning/thinking fields),
``_ThinkingStreamParser`` (inline ``⬢`` tag splitter), and
``_resolve_expert_model`` (admin-configured model/
base_url/api_key resolution for the ``expert`` task).

``_ExpertLLM`` mirrors ``api.docgen._common._StandardLLM`` for non-streaming
generation (file-disjoint per the Wave 2 scope contract) and adds async
streaming that bypasses the Generator to call the model client's
``acall(stream=True)`` directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from api.utils import setup_logging
from api.expert.types import (
    EVENT_CONTENT,
    EVENT_REASONING,
    ExpertStreamEvent,
)

setup_logging()
logger = logging.getLogger(__name__)

# Inline thinking tag delimiters used by some models (Qwen3 with
# think:false, or older checkpoints) that inline the reasoning trace directly
# in the content field instead of emitting a separate field.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*$", re.DOTALL)


class _ExpertLLM:
    """Thin LLM wrapper over the configured local model.

    Mirrors ``api.docgen._common._StandardLLM`` for non-streaming generation
    (adalflow ``Generator`` with ``template=\"{{input_str}}\"``) and adds an async
    ``stream()`` that bypasses the Generator to call the model client's
    ``acall(stream=True)`` directly. Honors admin-configured ``base_url`` / ``api_key``
    from ``api.config.settings.get_model_for_task`` when provided.
    """

    def __init__(
        self,
        model: Optional[str],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        import adalflow as adal
        from api.config import get_model_config

        generator_config = get_model_config(model)
        client_cls = generator_config["model_client"]
        self.model_kwargs = generator_config["model_kwargs"]
        self.model = self.model_kwargs.get("model", model)

        # Every supported local server (LM Studio, llama.cpp, vLLM, ...)
        # exposes the OpenAI-compatible /v1 API, so OpenAIClient covers all
        # cases. SSL verify is wired via ssl_config.
        client_kwargs: Dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        self.model_client = client_cls(**client_kwargs)
        self.generator = adal.Generator(
            template="{{input_str}}",
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
        )

    async def generate(self, prompt: str) -> str:
        """Non-streaming generation. Returns the raw response text."""

        def _call() -> str:
            # adalflow 1.x Generator.call() takes ``prompt_kwargs`` (the dict
            # that fills the ``{{input_str}}`` template placeholder), NOT a
            # bare ``input_str=`` kwarg. Passing ``input_str=`` raises
            # ``TypeError: Generator.call() got an unexpected keyword argument
            # 'input_str'`` (see api/docgen/wiki.py for the pattern).
            result = self.generator(prompt_kwargs={"input_str": prompt})
            # On a model error (e.g. 401 / connection refused) adalflow returns
            # a GeneratorOutput with ``error`` set and ``data=None`` but stores
            # the full prompt on ``input`` for tracing. Returning
            # ``str(result)`` would leak the entire prompt (system prompt +
            # retrieved context + query) into the expert chat/doc. Treat any
            # error as "no generation" so the caller surfaces a graceful
            # failure message instead of the raw prompt.
            if getattr(result, "error", None):
                logger.warning("Expert LLM returned an error: %s", result.error)
                return ""
            for attr in ("data", "response", "answer", "raw_response", "output"):
                val = getattr(result, attr, None)
                if val:
                    return str(val)
            return ""

        return await asyncio.to_thread(_call)

    async def stream(self, prompt: str) -> AsyncIterator[ExpertStreamEvent]:
        """Async-stream typed events from the local LLM.

        Yields ``ExpertStreamEvent`` objects:
        - ``("reasoning", text)`` for reasoning/thinking deltas (from
          ``reasoning_content`` / ``reasoning`` / ``thinking`` fields or
          inline ``<think>`` tags parsed from content).
        - ``("content", text)`` for answer text deltas.

        Attempts true token streaming via the model client's ``acall``; on any
        failure falls back to non-streaming ``generate`` + chunked delivery so
        the caller always receives incremental events.
        """
        # Late import: prompt.py defines _clean_llm_text + _chunk_text. Kept
        # local to avoid an import cycle (prompt -> llm would otherwise circle).
        from api.expert.prompt import _clean_llm_text, _chunk_text

        try:
            from adalflow.core.types import ModelType

            # Every supported server uses the flat OpenAI-compatible streaming
            # request shape.
            mk = {
                "model": self.model,
                "stream": True,
                "temperature": self.model_kwargs.get("temperature", 0.1),
            }
            if "top_p" in self.model_kwargs:
                mk["top_p"] = self.model_kwargs["top_p"]
            if "seed" in self.model_kwargs:
                mk["seed"] = self.model_kwargs["seed"]

            api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
                input=prompt, model_kwargs=mk, model_type=ModelType.LLM
            )
            response = await self.model_client.acall(
                api_kwargs=api_kwargs, model_type=ModelType.LLM
            )
            parser = _ThinkingStreamParser()
            produced = False
            async for chunk in response:  # type: ignore[union-attr]
                content_text, reasoning_text = _extract_chunk_fields(
                    chunk
                )
                # Separate reasoning field (DeepSeek/vLLM thinking) —
                # yield directly, no inline-tag parsing needed.
                if reasoning_text:
                    produced = True
                    yield ExpertStreamEvent(EVENT_REASONING, reasoning_text)
                # Content may contain inline <think> tags (Qwen3 with
                # think:false). Route through the stateful parser.
                if content_text:
                    produced = True
                    for ev in parser.feed(content_text):
                        yield ev
            # Flush any text still buffered in the parser.
            for ev in parser.flush():
                produced = True
                yield ev
            if produced:
                return
            logger.warning(
                "Expert LLM stream produced no chunks; falling back to generate."
            )
        except Exception as e:  # pragma: no cover - depends on live HTTP
            logger.warning(
                "Expert LLM streaming failed (%s); falling back to chunked generate.",
                e,
            )

        # Fallback: non-streaming generate + chunked yield.
        try:
            text = _clean_llm_text(await self.generate(prompt))
        except Exception as e:  # pragma: no cover - depends on live LLM
            logger.warning("Expert LLM fallback generate failed: %s", e)
            return
        # Strip inline thinking tags — the non-streaming fallback can't show
        # reasoning separately, so remove it from the answer text.
        text = _strip_thinking_tags(text)
        for piece in _chunk_text(text):
            yield ExpertStreamEvent(EVENT_CONTENT, piece)


class _ThinkingStreamParser:
    """Stateful parser for inline ``<think>...</think>`` tags in the content stream.

    Some models (Qwen3 with ``think:false``, or older checkpoints)
    inline the reasoning trace directly in the ``content`` field using
    ``<think>...</think>`` tags instead of emitting a separate
    ``reasoning_content`` / ``reasoning`` / ``thinking`` field. This parser
    detects tag boundaries across chunk boundaries and yields
    ``ExpertStreamEvent`` objects: reasoning events while inside ``<think>``,
    content events outside.

    When no ``<think>`` tags are present (non-reasoning models, or models that
    emit reasoning as a separate field), text passes through as content
    unchanged — near-zero overhead.
    """

    def __init__(self) -> None:
        self._in_thinking = False
        self._buffer = ""

    def feed(self, text: str) -> List[ExpertStreamEvent]:
        """Process new text and return events. Buffers ambiguous tail."""
        if not text:
            return []
        self._buffer += text
        events: List[ExpertStreamEvent] = []

        while self._buffer:
            if self._in_thinking:
                idx = self._buffer.find(_THINK_CLOSE)
                if idx == -1:
                    # No close tag yet — check for a partial tag at the end.
                    partial = self._partial_tag_match(self._buffer, _THINK_CLOSE)
                    if partial < len(self._buffer):
                        events.append(
                            ExpertStreamEvent(
                                EVENT_REASONING, self._buffer[:partial]
                            )
                        )
                        self._buffer = self._buffer[partial:]
                        break
                    # Entire buffer could be the start of the close tag.
                    events.append(
                        ExpertStreamEvent(EVENT_REASONING, self._buffer)
                    )
                    self._buffer = ""
                    break
                else:
                    # Found the close tag.
                    events.append(
                        ExpertStreamEvent(EVENT_REASONING, self._buffer[:idx])
                    )
                    self._buffer = self._buffer[idx + len(_THINK_CLOSE) :]
                    self._in_thinking = False
                    # Strip leading whitespace after </think>.
                    self._buffer = self._buffer.lstrip("\n")
            else:
                idx = self._buffer.find(_THINK_OPEN)
                if idx == -1:
                    # No open tag — check for a partial tag at the end.
                    partial = self._partial_tag_match(self._buffer, _THINK_OPEN)
                    if partial < len(self._buffer):
                        events.append(
                            ExpertStreamEvent(
                                EVENT_CONTENT, self._buffer[:partial]
                            )
                        )
                        self._buffer = self._buffer[partial:]
                        break
                    # Entire buffer could be the start of the open tag.
                    events.append(
                        ExpertStreamEvent(EVENT_CONTENT, self._buffer)
                    )
                    self._buffer = ""
                    break
                else:
                    # Found the open tag.
                    if idx > 0:
                        events.append(
                            ExpertStreamEvent(EVENT_CONTENT, self._buffer[:idx])
                        )
                    self._buffer = self._buffer[idx + len(_THINK_OPEN) :]
                    self._in_thinking = True
                    # Strip leading whitespace after <think>.
                    self._buffer = self._buffer.lstrip("\n")

        return events

    def flush(self) -> List[ExpertStreamEvent]:
        """Yield any remaining buffered text. Called when the stream ends."""
        if not self._buffer:
            return []
        text = self._buffer
        self._buffer = ""
        if self._in_thinking:
            # Stream ended inside thinking — yield as reasoning.
            return [ExpertStreamEvent(EVENT_REASONING, text)]
        return [ExpertStreamEvent(EVENT_CONTENT, text)]

    @staticmethod
    def _partial_tag_match(buffer: str, tag: str) -> int:
        """Return the index where a partial tag prefix starts at the buffer end.

        If the buffer ends with a prefix of ``tag`` (e.g. buffer ends with
        ``"<thi"`` and tag is ``"<think>"``), return the start index of that
        prefix so the caller can keep it buffered for the next chunk.
        Otherwise return ``len(buffer)`` (nothing to keep).
        """
        max_overlap = min(len(buffer), len(tag) - 1)
        for i in range(max_overlap, 0, -1):
            if buffer[-i:] == tag[:i]:
                return len(buffer) - i
        return len(buffer)


def _strip_thinking_tags(text: str) -> str:
    """Remove inline ``<think>...</think>`` blocks (or unclosed ``<think>``)."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def _get_field(obj: Any, *keys: str) -> Optional[str]:
    """Get the first non-empty value for any of *keys* from a dict or object."""
    for key in keys:
        if isinstance(obj, dict):
            val = obj.get(key)
        else:
            val = getattr(obj, key, None)
        if val:
            return val
    return None


def _extract_chunk_fields(
    chunk: Any
) -> Tuple[Optional[str], Optional[str]]:
    """Extract ``(content, reasoning)`` deltas from a streaming chunk.

    Handles native (``message.content`` / ``message.thinking``), OpenAI
    /v1 (``choices[0].delta.reasoning``), DeepSeek/DashScope
    (``choices[0].delta.reasoning_content``), and plain dict shapes. Either
    element of the tuple may be ``None`` if the field is absent or empty.
    """
    # native format (object with .message attribute)
    message = getattr(chunk, "message", None)
    if message is not None:
        if isinstance(message, dict):
            content = message.get("content") or None
            reasoning = _get_field(message, "thinking", "reasoning_content", "reasoning")
        else:
            content = getattr(message, "content", None) or None
            reasoning = _get_field(message, "thinking", "reasoning_content", "reasoning")
        if (isinstance(content, str) and content) or reasoning:
            return content, reasoning

    # dict-shaped chunk
    if isinstance(chunk, dict):
        msg = chunk.get("message")
        if isinstance(msg, dict):
            content = msg.get("content") or None
            reasoning = _get_field(msg, "thinking", "reasoning_content", "reasoning")
            if content or reasoning:
                return content, reasoning

        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            if isinstance(delta, dict):
                content = delta.get("content") or None
                reasoning = _get_field(delta, "reasoning_content", "reasoning", "thinking")
                if content or reasoning:
                    return content, reasoning

        if chunk.get("content"):
            content = chunk["content"]
            reasoning = _get_field(chunk, "thinking", "reasoning_content", "reasoning")
            return content, reasoning

    # OpenAI ChatCompletion chunk format (object with .choices)
    choices = getattr(chunk, "choices", None)
    if choices:
        try:
            delta = getattr(choices[0], "delta", None)
        except Exception:
            delta = None
        if delta is not None:
            content = getattr(delta, "content", None) or None
            reasoning = _get_field(delta, "reasoning_content", "reasoning", "thinking")
            if (isinstance(content, str) and content) or reasoning:
                return content, reasoning

    # adalflow GeneratorOutput-style: .response / .data
    for attr in ("response", "data", "text"):
        val = getattr(chunk, attr, None)
        if isinstance(val, str) and val:
            return val, None

    return None, None


def _safe_build_llm(
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[_ExpertLLM]:
    """Build an ``_ExpertLLM`` or return None on any failure (graceful degrade)."""
    try:
        return _ExpertLLM(model, base_url=base_url, api_key=api_key)
    except Exception as e:  # pragma: no cover - depends on live config/LLM
        logger.warning(
            "Could not initialise expert LLM (%s): %s. "
            "Expert answers will be empty until a model is available.",
            model,
            e,
        )
        return None


def _resolve_expert_model(model: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve (model, base_url, api_key) for the expert task.

    Reads admin-configured ``models.expert.*`` from the Config Abstraction Layer.
    """
    try:
        from api.config.abstraction import get_task_config

        cfg = get_task_config("expert") or {}
        resolved_model = model or cfg.get("model") or "qwen/qwen3.6-27b"
        return resolved_model, cfg.get("base_url"), cfg.get("api_key")
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_task_config(expert) failed; using defaults: %s", e)
        return model or "qwen/qwen3.6-27b", None, None
