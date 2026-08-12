"""Expert LLM wrapper + factory + chunk-text extraction + model resolution.

Split out of the former ``api/expert_agent.py`` (Step 6). Owns the
``_ExpertLLM`` adalflow Generator wrapper (non-streaming ``generate`` + async
``stream`` with chunked fallback), the ``_safe_build_llm`` graceful-degrade
factory, ``_extract_chunk_text`` (Ollama + OpenAI streaming-chunk shape
handling), and ``_resolve_expert_model`` (admin-configured provider/model/
base_url/api_key resolution for the ``expert`` task).

``_ExpertLLM`` mirrors ``api.docgen._common._StandardLLM`` for non-streaming
generation (file-disjoint per the Wave 2 scope contract) and adds async
streaming that bypasses the Generator to call the model client's
``acall(stream=True)`` directly — the same pattern used by
``api/websocket_wiki.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from api.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class _ExpertLLM:
    """Thin LLM wrapper over the configured local model.

    Mirrors ``api.docgen._common._StandardLLM`` for non-streaming generation
    (adalflow ``Generator`` with ``template=\"{{input_str}}\"``) and adds an async
    ``stream()`` that bypasses the Generator to call the model client's
    ``acall(stream=True)`` directly — the same pattern used by
    ``api/websocket_wiki.py``. Honors admin-configured ``base_url`` / ``api_key``
    from ``api.settings_store.get_model_for_task`` when provided.
    """

    def __init__(
        self,
        provider: str,
        model: Optional[str],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        import adalflow as adal
        from api.config import get_model_config

        generator_config = get_model_config(provider, model)
        client_cls = generator_config["model_client"]
        self.model_kwargs = generator_config["model_kwargs"]
        self.model = self.model_kwargs.get("model", model)

        # Every supported local server (Ollama, LM Studio, llama.cpp, vLLM, ...)
        # exposes the OpenAI-compatible /v1 API, so OpenAIClient covers all
        # cases. SSL verify is wired via ssl_config.
        client_kwargs: Dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        self.provider = provider
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
            # 'input_str'`` (see api/wiki_generator.py:283 for the pattern).
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

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Async-stream text chunks from the local LLM.

        Attempts true token streaming via the model client's ``acall``; on any
        failure falls back to non-streaming ``generate`` + chunked delivery so
        the caller always receives incremental chunks.
        """
        # Late import: prompt.py defines _clean_llm_text + _chunk_text. Kept
        # local to avoid an import cycle (prompt -> llm would otherwise circle).
        from api.expert.prompt import _clean_llm_text, _chunk_text

        try:
            from adalflow.core.types import ModelType

            # Every supported server uses the flat OpenAI-compatible streaming
            # request shape (Ollama's /v1 endpoint accepts it too).
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
            produced = False
            async for chunk in response:  # type: ignore[union-attr]
                text = _extract_chunk_text(chunk, self.provider)
                if text:
                    produced = True
                    yield text
            if produced:
                return
            logger.warning(
                "Expert LLM stream produced no chunks; falling back to generate."
            )
        except Exception as e:  # pragma: no cover - depends on live Ollama/HTTP
            logger.warning(
                "Expert LLM streaming failed (%s); falling back to chunked generate.",
                e,
            )

        # Fallback: non-streaming generate + chunked yield.
        try:
            text = _clean_llm_text(await self.generate(prompt))
        except Exception as e:  # pragma: no cover - depends on live Ollama
            logger.warning("Expert LLM fallback generate failed: %s", e)
            return
        for piece in _chunk_text(text):
            yield piece


def _extract_chunk_text(chunk: Any, provider: str) -> Optional[str]:
    """Extract a text delta from a streaming chunk (Ollama or OpenAI shape).

    Mirrors the extraction logic in ``api/websocket_wiki.py`` for both the
    Ollama native format (``message.content``) and the OpenAI streaming format
    (``choices[0].delta.content``).
    """
    # Ollama native format
    message = getattr(chunk, "message", None)
    if message is not None:
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            return content
    # dict-shaped Ollama chunk
    if isinstance(chunk, dict):
        msg = chunk.get("message")
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            if isinstance(delta, dict) and delta.get("content"):
                return delta["content"]
        if chunk.get("content"):
            return chunk["content"]
    # OpenAI ChatCompletion chunk format
    choices = getattr(chunk, "choices", None)
    if choices:
        try:
            delta = getattr(choices[0], "delta", None)
        except Exception:
            delta = None
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                return content
    # adalflow GeneratorOutput-style: .response / .data
    for attr in ("response", "data", "text"):
        val = getattr(chunk, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def _safe_build_llm(
    provider: str,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[_ExpertLLM]:
    """Build an ``_ExpertLLM`` or return None on any failure (graceful degrade)."""
    try:
        return _ExpertLLM(provider, model, base_url=base_url, api_key=api_key)
    except Exception as e:  # pragma: no cover - depends on live config/Ollama
        logger.warning(
            "Could not initialise expert LLM (%s/%s): %s. "
            "Expert answers will be empty until a model is available.",
            provider,
            model,
            e,
        )
        return None


def _resolve_expert_model(model: Optional[str]) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Resolve (provider, model, base_url, api_key) for the expert task.

    Reads admin-configured ``models.expert.*`` from the Config Abstraction Layer.
    """
    try:
        from api.config_abstraction import get_task_config

        cfg = get_task_config("expert") or {}
        provider = cfg.get("provider") or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")
        resolved_model = model or cfg.get("model") or os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b")
        return provider, resolved_model, cfg.get("base_url"), cfg.get("api_key")
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_task_config(expert) failed; using defaults: %s", e)
        return os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"), model or os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b"), None, None
