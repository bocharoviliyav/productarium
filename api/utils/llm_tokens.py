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
from typing import Any, Dict, Optional, Tuple

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
    2. Task/admin config `models.<task>.max_prompt_tokens`
    3. Live API metadata query (cached for 5 minutes), in order:
       - OpenAI-compatible: GET {base_url}/v1/models -> `max_model_len` /
         `context_window` / `max_tokens` / `max_context_length` / `n_ctx`
         (vLLM advertises these; LM Studio / llama.cpp / Ollama do NOT)
       - LM Studio native: GET {origin}/api/v0/models -> `maxContextLength`
       - llama.cpp: GET {origin}/props -> `default_generation_settings.n_ctx`
       - Ollama: POST {origin}/api/show -> `model_info["*.context_length"]`
    4. Safe fallback default: 8192 tokens (logged as a WARNING that names
       the manual overrides — a silent 8192 previously clamped every docgen
       budget on 262k-token models whose server does not advertise the
       window via /v1/models)
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

    # 3. Live endpoint queries
    try:
        from api.config.ssl import requests_verify
        from api.config.settings import _sanitize_api_key
        from api.config.timeout import resolve_model_list_timeout

        headers = {}
        clean_key = _sanitize_api_key(api_key or os.environ.get("LOCAL_OPENAI_API_KEY"))
        if clean_key and clean_key.lower() not in ("not-needed", "not_needed"):
            headers["Authorization"] = f"Bearer {clean_key}"

        request_timeout = resolve_model_list_timeout()
        verify = requests_verify()

        def _get(probe_url: str) -> Optional[dict]:
            try:
                resp = requests.get(
                    probe_url, headers=headers,
                    timeout=request_timeout, verify=verify,
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.debug("Context-window probe %s failed: %s", probe_url, e)
            return None

        model_l = model.lower()

        # 3a. OpenAI-compatible /v1/models (vLLM advertises max_model_len &
        # friends here; LM Studio / llama.cpp / Ollama list bare ids only).
        oai_url = url if url.endswith("/v1") else url.rstrip("/") + "/v1"
        data = _get(f"{oai_url}/models")
        if isinstance(data, dict):
            for mobj in data.get("data", []) or []:
                if not isinstance(mobj, dict):
                    continue
                entry_id = str(mobj.get("id") or mobj.get("name") or "").strip().lower()
                if entry_id != model_l:
                    continue
                for key in ("max_model_len", "context_window", "max_tokens", "max_context_length", "n_ctx"):
                    val = mobj.get(key)
                    if isinstance(val, (int, float)) and val > 0:
                        ctx_found = int(val)
                        break
                if ctx_found:
                    break

        # 3b-d. Server-NATIVE probes: these servers do not publish the
        # context window through /v1/models, only through their own endpoints.
        # origin = the OpenAI base without the /v1 suffix.
        origin = oai_url[: -len("/v1")].rstrip("/") if oai_url.endswith("/v1") else oai_url.rstrip("/")

        if ctx_found is None:
            # LM Studio native REST API.
            data = _get(f"{origin}/api/v0/models")
            if isinstance(data, dict):
                for mobj in data.get("data", []) or []:
                    if not isinstance(mobj, dict):
                        continue
                    entry_id = str(mobj.get("id") or "").strip().lower()
                    if entry_id != model_l:
                        continue
                    val = mobj.get("maxContextLength")
                    if isinstance(val, (int, float)) and val > 0:
                        ctx_found = int(val)
                        break

        if ctx_found is None:
            # llama.cpp server properties.
            data = _get(f"{origin}/props")
            if isinstance(data, dict):
                settings = data.get("default_generation_settings")
                if isinstance(settings, dict):
                    val = settings.get("n_ctx")
                    if isinstance(val, (int, float)) and val > 0:
                        ctx_found = int(val)

        if ctx_found is None:
            # Ollama model info: model_info["<arch>.context_length"].
            try:
                resp = requests.post(
                    f"{origin}/api/show",
                    json={"model": model},
                    timeout=request_timeout,
                    verify=verify,
                )
                if resp.status_code == 200:
                    info = (resp.json() or {}).get("model_info")
                    if isinstance(info, dict):
                        for key, val in info.items():
                            if (
                                isinstance(key, str)
                                and key.endswith(".context_length")
                                and isinstance(val, (int, float))
                                and val > 0
                            ):
                                ctx_found = int(val)
                                break
            except Exception as e:
                logger.debug("Context-window probe %s/api/show failed: %s", origin, e)
    except Exception as e:
        logger.debug("get_model_context_window fetch exception: %s", e)

    final_ctx = ctx_found or 8192
    if ctx_found is None:
        logger.warning(
            "No advertised context window for %s at %s (probed /v1/models, "
            "LM Studio /api/v0/models, llama.cpp /props, Ollama /api/show); "
            "falling back to 8192 tokens. If the real window is larger, set "
            "RLM_MODEL_CONTEXT_WINDOW or the admin Models -> "
            "max_prompt_tokens override.",
            model, url,
        )

    _MODEL_CTX_CACHE[cache_key] = (now, final_ctx)
    logger.info("Resolved model context window for %s at %s: %d tokens", model, url, final_ctx)
    return final_ctx


# P1-23: hybrid token counting. The default is the cheap chars/4 heuristic
# (encoding a whole codebase chunk-by-chunk is CPU-heavy and was O(total_chars)
# per call with per-call get_encoding). Precise tiktoken counting is opt-in via
# the admin setting ``llm.precise_tokens`` or env ``LLM_PRECISE_TOKENS`` — one
# cl100k_base encoder per process (providers/models change, the encoding does
# not need to).
_PRECISE_TOKENS_ENV = "LLM_PRECISE_TOKENS"
_PRECISE_TOKENS_SETTING = "llm.precise_tokens"
_ENCODER: Any = None
_ENCODER_LOADED = False


def _truthy(raw: Any) -> bool:
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _precise_tokens_enabled() -> bool:
    """True when precise tiktoken counting is explicitly enabled."""
    env_raw = os.environ.get(_PRECISE_TOKENS_ENV)
    if env_raw is not None and env_raw.strip():
        return _truthy(env_raw)
    try:
        from api.config.settings import get_setting

        val = get_setting(_PRECISE_TOKENS_SETTING)
        if val is not None and str(val).strip():
            return _truthy(val)
    except Exception:  # pragma: no cover - settings store is import-safe
        pass
    return False


def _get_encoder() -> Any:
    """Singleton cl100k_base encoder (built once per process). None if unavailable."""
    global _ENCODER, _ENCODER_LOADED
    if not _ENCODER_LOADED:
        _ENCODER_LOADED = True
        try:
            import tiktoken  # type: ignore

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - optional dep
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str, precise: Optional[bool] = None) -> int:
    """Approximate token count for prompt budget estimation (P1-23 hybrid).

    - default: ``max(1, len(text) // 4)`` — a fast chars/4 heuristic;
    - precise mode (``precise=True``, or globally enabled via the admin
      setting ``llm.precise_tokens`` / env ``LLM_PRECISE_TOKENS``): exact
      tiktoken cl100k_base count via a process-wide singleton encoder,
      falling back to the heuristic when tiktoken is unavailable.
    """
    if not text:
        return 0
    use_precise = _precise_tokens_enabled() if precise is None else bool(precise)
    if use_precise:
        enc = _get_encoder()
        if enc is not None:
            try:
                return len(enc.encode(text, disallowed_special=()))
            except Exception:  # pragma: no cover - defensive
                pass
    return max(1, len(text) // 4)


# Backwards-compatible alias (callers import ``_count_tokens``).
_count_tokens = count_tokens


async def get_model_context_window_async(
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    task: Optional[str] = None,
) -> int:
    """Async variant of :func:`get_model_context_window` (P1-13).

    The sync resolver may perform a live ``requests.get`` (first call per
    model/endpoint, 5-minute cache afterwards) which would block the event
    loop. This wrapper runs it in a worker thread — safe to await from async
    request handlers.
    """
    import asyncio

    return await asyncio.to_thread(
        get_model_context_window,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        task=task,
    )
