"""Embedder factory over the OpenAI-compatible /v1/embeddings endpoint.

Every supported local server (LM Studio, llama.cpp, vLLM, ...) exposes an
OpenAI-compatible embeddings endpoint, so a single langchain
``OpenAIEmbeddings`` instance covers all cases. Admin-configured
``models.embedder.{model,base_url,api_key}`` (settings store) is threaded
through with env/JSON fallbacks; TLS verification is wired via
:mod:`api.config.ssl` so the corporate CA bundle reaches the enterprise
gateway.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from api.config import configs

logger = logging.getLogger(__name__)

# P1-25: process-level embedder cache. Building an embedder constructs a new
# model client (connection pool, SSL context) on every call; the pgvector
# memory backend embeds one batch per source, so indexing/reindexing a
# product with many sources previously built hundreds of throwaway clients.
# Instances are cached keyed by the EFFECTIVE (base_url, api_key, model) so an
# admin config change yields a fresh client while unchanged configs reuse the
# cached one. Bounded FIFO so stale configs can't accumulate.
_EMBEDDER_CACHE: dict = {}
_EMBEDDER_CACHE_LOCK = threading.Lock()
_MAX_CACHED_EMBEDDERS = 8

# Default embedding model (matches .env.example / nomic-embed-text-v1.5, 768d).
_DEFAULT_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"


def get_embedder(base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Build a langchain ``OpenAIEmbeddings`` for the configured endpoint.

    Args:
        base_url: Custom base URL for the embedder provider.
        api_key: Custom API key for the embedder provider.

    Returns:
        A configured ``OpenAIEmbeddings`` instance (``embed_documents(list)``
        and ``embed_query(str)`` APIs).
    """
    from langchain_openai import OpenAIEmbeddings

    # Thread admin-configured embedder model/base_url/api_key into the
    # resolved configuration (settings store wins, then explicit args, then
    # env, then embedder.json).
    admin_model: Optional[str] = None
    try:
        from api.config.settings import get_model_for_task, get_setting

        emb_cfg = get_model_for_task("embedder") or {}
        # Only an EXPLICIT admin-configured embedder model overrides the
        # JSON config: ``get_model_for_task`` falls back to the chat default
        # (``qwen/...``) when nothing is stored, which must not shadow the
        # configured 768d embedding model.
        admin_model = get_setting("models.embedder.model")
    except Exception:  # pragma: no cover - settings store is import-safe
        emb_cfg = {}

    embedder_config = configs.get("embedder_openai_local")
    if not embedder_config:
        raise ValueError(
            "No embedder configuration found. Please check your embedder.json config."
        )
    model_kwargs = dict(embedder_config.get("model_kwargs") or {})

    model = admin_model or model_kwargs.get("model") or _DEFAULT_EMBEDDING_MODEL
    if not base_url:
        base_url = emb_cfg.get("base_url")
    if not api_key:
        api_key = emb_cfg.get("api_key")

    # P1-25: reuse the client for unchanged effective configs (the key covers
    # everything that feeds the model client + model_kwargs below).
    cache_key = (base_url or "", api_key or "", model)
    with _EMBEDDER_CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    resolved_base_url = (
        base_url
        or os.environ.get("LOCAL_OPENAI_BASE_URL")
        or "http://localhost:1234/v1"
    )

    # API-key placeholder handling: 'not-needed'/'not_needed' (or missing on
    # a local endpoint) mean no auth — the SDK needs a value, and the
    # Authorization header is stripped via an httpx event hook.
    from api.llm.client import (
        NO_AUTH_PLACEHOLDER,
        build_http_async_client,
        is_local_endpoint,
        resolve_api_key,
    )

    resolved_key = resolve_api_key(api_key or os.environ.get("LOCAL_OPENAI_API_KEY"))
    if not resolved_key:
        resolved_key = NO_AUTH_PLACEHOLDER

    http_async_client = build_http_async_client(
        strip_auth=resolved_key == NO_AUTH_PLACEHOLDER
    )
    # A no-auth REMOTE endpoint (no explicit placeholder) is a likely
    # misconfiguration: an embedder silently sending placeholder creds to a
    # remote gateway would 401 anyway, so mirror the chat-model policy.
    if (
        resolved_key == NO_AUTH_PLACEHOLDER
        and not is_local_endpoint(resolved_base_url)
        and api_key is None
    ):
        logger.warning(
            "Embedder: no real API key configured for remote endpoint %s; "
            "using placeholder (requests will carry no Authorization header).",
            resolved_base_url,
        )

    kwargs = {
        "model": model,
        "api_key": resolved_key,
        "base_url": resolved_base_url,
        "http_async_client": http_async_client,
        # langchain's default client-side context-length check tokenizes
        # inputs with tiktoken (cl100k) and sends token-ID arrays instead of
        # raw strings. Local GGUF servers tokenize server-side and validate
        # ids against their own vocab (~30k for nomic), so most cl100k ids
        # land out of range and the whole request is rejected with 400
        # "Prompt contains invalid tokens". Send plain strings instead;
        # chunk sizes (~350 words) are far below the model context anyway.
        "check_embedding_ctx_length": False,
    }
    # Preserve extra configured params (e.g. encoding_format) as request opts.
    for key in ("encoding_format", "dimensions"):
        if model_kwargs.get(key) is not None:
            kwargs[key] = model_kwargs[key]

    batch_size = embedder_config.get("batch_size")
    if isinstance(batch_size, int) and batch_size > 0:
        kwargs["chunk_size"] = batch_size

    embedder = OpenAIEmbeddings(**kwargs)
    with _EMBEDDER_CACHE_LOCK:
        if len(_EMBEDDER_CACHE) >= _MAX_CACHED_EMBEDDERS:
            # FIFO-evict the oldest entry (dicts preserve insertion order).
            _EMBEDDER_CACHE.pop(next(iter(_EMBEDDER_CACHE)), None)
        _EMBEDDER_CACHE[cache_key] = embedder
    return embedder
