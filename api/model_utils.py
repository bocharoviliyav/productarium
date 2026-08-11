"""Model utility functions for context window detection and text token clamping."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Cache detected model context windows for 5 minutes: (provider, base_url, model) -> (timestamp, context_window)
_MODEL_CTX_CACHE: Dict[Tuple[str, str, str], Tuple[float, int]] = {}
_CACHE_TTL_SECONDS = 300.0


def _parse_context_from_name(model_name: str) -> Optional[int]:
    """Heuristic context window resolution from model name strings."""
    if not model_name:
        return None
    name_lower = model_name.lower()

    # Check explicit context indicators in name e.g. 128k, 64k, 32k, 16k, 8k, 4k
    if "128k" in name_lower:
        return 131072
    if "64k" in name_lower:
        return 65536
    if "32k" in name_lower:
        return 32768
    if "16k" in name_lower:
        return 16384
    if "8k" in name_lower:
        return 8192
    if "4k" in name_lower:
        return 4096

    # Known model families with default contexts
    if any(k in name_lower for k in ("qwen3.5", "qwen3.6", "qwen2.5", "llama3", "mistral", "gemma2", "deepseek")):
        return 32768
    if "qwen3" in name_lower or "qwen2" in name_lower:
        return 32768
    if any(k in name_lower for k in ("8b", "7b", "3b", "1b")):
        return 8192

    return None


def get_model_context_window(
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    task: Optional[str] = None,
) -> int:
    """Dynamically resolve the context window size (in tokens) for a given model/endpoint.

    Order of precedence:
    1. Explicit env vars `RLM_MODEL_CONTEXT_WINDOW` or `OLLAMA_NUM_CTX`
    2. Task/admin config `models.<task>.max_prompt_tokens` / `context_window`
    3. Live API metadata query (cached for 5 minutes):
       - Ollama: POST {host}/api/show -> parse `model_info` / `parameters`
       - OpenAI-compatible: GET {base_url}/models -> parse `max_model_len` / `context_window` / `max_tokens`
    4. Model name heuristic hints
    5. Safe fallback default: 8192 tokens
    """
    # 1. Environment variable override
    for env_key in ("RLM_MODEL_CONTEXT_WINDOW", "OLLAMA_NUM_CTX"):
        raw = os.environ.get(env_key)
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
            from api.settings_store import get_model_for_task
            cfg = get_model_for_task(task) or {}
            mpt = cfg.get("max_prompt_tokens")
            if isinstance(mpt, int) and mpt > 0:
                return mpt
        except Exception:
            pass

    prov = (provider or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER") or "openai_local").lower()
    model = (model_name or os.environ.get("DEEPWIKI_DEFAULT_MODEL") or "qwen/qwen3.6-27b").strip()
    url = base_url or os.environ.get("LOCAL_OPENAI_BASE_URL" if prov == "openai_local" else "OLLAMA_HOST") or "http://localhost:8080/v1"

    # Check cache
    cache_key = (prov, url.rstrip("/"), model)
    now = time.time()
    if cache_key in _MODEL_CTX_CACHE:
        ts, cached_val = _MODEL_CTX_CACHE[cache_key]
        if now - ts < _CACHE_TTL_SECONDS:
            return cached_val

    ctx_found: Optional[int] = None

    # 3. Live endpoint query
    try:
        from api.ssl_config import requests_verify
        from api.settings_store import _sanitize_api_key
        from api.timeout_config import resolve_model_list_timeout

        headers = {}
        clean_key = _sanitize_api_key(api_key or os.environ.get("LOCAL_OPENAI_API_KEY"))
        if clean_key and clean_key.lower() not in ("not-needed", "not_needed"):
            headers["Authorization"] = f"Bearer {clean_key}"

        if prov == "ollama" or ":11434" in url or "ollama" in url.lower():
            ollama_base = url.replace("/v1", "").rstrip("/")
            try:
                resp = requests.post(f"{ollama_base}/api/show", json={"name": model}, headers=headers, timeout=resolve_model_list_timeout(), verify=requests_verify())
                if resp.status_code == 200:
                    data = resp.json()
                    model_info = data.get("model_info", {})
                    for k, v in model_info.items():
                        if "context_length" in k and isinstance(v, (int, float)) and v > 0:
                            ctx_found = int(v)
                            break
                    if not ctx_found:
                        params_str = str(data.get("parameters", ""))
                        m = re.search(r"num_ctx\s+(\d+)", params_str)
                        if m:
                            ctx_found = int(m.group(1))
            except Exception as e:
                logger.debug("Ollama /api/show query failed for %s: %s", model, e)

        else:
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
                            if ctx_found:
                                break
            except Exception as e:
                logger.debug("OpenAI /v1/models query failed for %s: %s", model, e)
    except Exception as e:
        logger.debug("get_model_context_window fetch exception: %s", e)

    # 4. Fallback to name heuristic
    if not ctx_found:
        ctx_found = _parse_context_from_name(model)

    # 5. Default fallback
    final_ctx = ctx_found or 8192

    _MODEL_CTX_CACHE[cache_key] = (now, final_ctx)
    logger.info("Resolved model context window for %s/%s at %s: %d tokens", prov, model, url, final_ctx)
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


def clamp_text_by_tokens(text: str, max_tokens: int, preserve_tail: bool = False) -> str:
    """Clamp a text block so that its token count does not exceed max_tokens.

    If preserve_tail=True, keeps the tail of the text (useful for conversation history).
    Otherwise, keeps the head of the text (useful for documents/code).
    """
    if not text or max_tokens <= 0:
        return ""

    current_tokens = _count_tokens(text)
    if current_tokens <= max_tokens:
        return text

    ratio = max_tokens / current_tokens
    char_limit = max(10, int(len(text) * ratio * 0.95))

    if preserve_tail:
        clamped = "... (история обрезана для контекста)\n" + text[-char_limit:]
    else:
        clamped = text[:char_limit] + "\n... (содержимое обрезано для контекста)"

    while _count_tokens(clamped) > max_tokens and len(clamped) > 50:
        char_limit = int(len(clamped) * 0.9)
        if preserve_tail:
            clamped = "... (история обрезана)\n" + clamped[-char_limit:]
        else:
            clamped = clamped[:char_limit] + "\n... (содержимое обрезано)"

    return clamped
