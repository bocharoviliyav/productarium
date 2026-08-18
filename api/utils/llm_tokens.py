"""LLM token / context-window math (former ``api/model_utils.py``).

Pure token-counting and context-window resolution with NO LLM client
dependencies, so the dependency direction stays clean: ``llm_helpers`` may
import this module, but this module imports nothing from the LLM client stack.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Cache detected model context windows for 5 minutes: (base_url, model) -> (timestamp, context_window)
_MODEL_CTX_CACHE: Dict[Tuple[str, str], Tuple[float, int]] = {}
_CACHE_TTL_SECONDS = 300.0


def get_model_context_window(
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    task: Optional[str] = None,
) -> int:
    """Dynamically resolve the context window size (in tokens) for a given model/endpoint.

    Order of precedence:
    1. Explicit env var `RLM_MODEL_CONTEXT_WINDOW`
    2. Task/admin config `models.<task>.max_prompt_tokens` / `context_window`
    3. Live API metadata query (cached for 5 minutes):
       - OpenAI-compatible: GET {base_url}/models -> parse `max_model_len` / `context_window` / `max_tokens`
    4. Model name heuristic hints
    5. Safe fallback default: 8192 tokens
    """
    # 1. Environment variable override
    raw = os.environ.get("RLM_MODEL_CONTEXT_WINDOW")
    if raw:
        try:
            val = int(str(raw).strip())
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass

    # 2. Admin setting override
    if task:
        try:
            from api.config.settings import get_model_for_task
            cfg = get_model_for_task(task) or {}
            mpt = cfg.get("max_prompt_tokens")
            if isinstance(mpt, int) and mpt > 0:
                return mpt
            # Resolve model/base_url/api_key from the admin config so
            # the live API query uses the admin-configured model name.
            if not model_name:
                model_name = cfg.get("model")
            if not base_url:
                base_url = cfg.get("base_url")
            if not api_key:
                api_key = cfg.get("api_key")
        except Exception:
            pass

    model = (model_name or "qwen/qwen3.6-27b").strip()
    url = base_url or os.environ.get("LOCAL_OPENAI_BASE_URL") or "http://localhost:8080/v1"

    # Check cache
    cache_key = (url.rstrip("/"), model)
    now = time.time()
    if cache_key in _MODEL_CTX_CACHE:
        ts, cached_val = _MODEL_CTX_CACHE[cache_key]
        if now - ts < _CACHE_TTL_SECONDS:
            return cached_val

    ctx_found: Optional[int] = None

    # 3. Live endpoint query
    try:
        from api.config.ssl import requests_verify
        from api.config.settings import _sanitize_api_key
        from api.config.timeout import resolve_model_list_timeout

        headers = {}
        clean_key = _sanitize_api_key(api_key or os.environ.get("LOCAL_OPENAI_API_KEY"))
        if clean_key and clean_key.lower() not in ("not-needed", "not_needed"):
            headers["Authorization"] = f"Bearer {clean_key}"

        # OpenAI-compatible /v1/models
        oai_url = url if url.endswith("/v1") else url.rstrip("/") + "/v1"
        try:
            resp = requests.get(f"{oai_url}/models", headers=headers, timeout=resolve_model_list_timeout(), verify=requests_verify())
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("data", []) if isinstance(data, dict) else []
                for mobj in models_list:
                    if isinstance(mobj, dict) and (mobj.get("id") == model or mobj.get("name") == model):
                        for key in ("max_model_len", "context_window", "max_tokens", "max_context_length", "n_ctx"):
                            val = mobj.get(key)
                            if isinstance(val, (int, float)) and val > 0:
                                ctx_found = int(val)
                                break
        except Exception as e:
            logger.debug("OpenAI /v1/models query failed for %s: %s", model, e)
    except Exception as e:
        logger.debug("get_model_context_window fetch exception: %s", e)

    final_ctx = ctx_found or 8192

    _MODEL_CTX_CACHE[cache_key] = (now, final_ctx)
    logger.info("Resolved model context window for %s at %s: %d tokens", model, url, final_ctx)
    return final_ctx


def _count_tokens(text: str) -> int:
    """Approximate token count for prompt budget estimation."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return max(1, len(text) // 4)
