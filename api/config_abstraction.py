"""Central Configuration Abstraction Layer for Productarium.

Architectural Guarantees:
1. DB Settings Store (SettingORM / admin panel UI) has HIGHEST PRECEDENCE over
   environment variables and JSON config files.
2. At application startup, defaults are collected from JSON config files and
   environment variables, populating the settings store DB if empty.
3. When settings are updated in the admin panel UI, new values are written to DB,
   pushed to `os.environ`, and all runtime singletons (`cognee.config`, `_MODEL_CTX_CACHE`,
   model client caches) are updated INSTANTLY without requiring a container restart.
4. All services read task configs through `get_task_config(task)`, guaranteeing
   they always see the highest-precedence active settings.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ALL_MODEL_TASKS = ("docgen", "expert", "summary", "cognee", "embedder")


def sync_runtime_settings() -> None:
    """Force all process subsystems to synchronize with DB settings immediately.

    1. Syncs active task API keys and base URLs to process environment variables.
    2. Re-applies cognee runtime config (mutates LLMConfig & EmbeddingConfig singletons).
    3. Clears model context window cache in `api.model_utils`.
    4. Exports admin-store timeout overrides to their canonical env vars so
       subprocess / module-level readers (e.g. fast-rlm's Pyodide REPL, which
       reads RLM_API_TIMEOUT_MS from the process environment and cannot reach
       host Python) see admin-set values without a restart.
    """
    try:
        from api.model_utils import _MODEL_CTX_CACHE
        _MODEL_CTX_CACHE.clear()
    except Exception:
        pass

    try:
        from api.settings_store import get_model_for_task
        for task in ("docgen", "expert", "summary"):
            cfg = get_model_for_task(task) or {}
            b_url = cfg.get("base_url")
            a_key = cfg.get("api_key")
            if b_url:
                os.environ["LOCAL_OPENAI_BASE_URL"] = b_url
            if a_key and a_key not in ("not-needed", "not_needed"):
                os.environ["LOCAL_OPENAI_API_KEY"] = a_key
                os.environ["OPENAI_API_KEY"] = a_key
    except Exception as e:
        logger.debug("sync_runtime_settings: env sync skipped: %s", e)

    try:
        from api.cognee_manager import apply_cognee_runtime_config
        apply_cognee_runtime_config()
    except Exception as e:
        logger.debug("sync_runtime_settings: apply_cognee_runtime_config skipped: %s", e)

    try:
        from api.timeout_config import sync_timeout_env
        sync_timeout_env()
    except Exception as e:
        logger.debug("sync_runtime_settings: sync_timeout_env skipped: %s", e)

    logger.info("Configuration Abstraction Layer: Synchronized runtime settings across process.")


def get_task_config(task: str) -> Dict[str, Optional[str]]:
    """Get model configuration for a task with HIGHEST PRECEDENCE given to DB settings.

    Precedence:
    1. DB SettingORM (`models.<task>.*`) — Admin UI saves.
    2. Process Environment Variables (`LOCAL_OPENAI_BASE_URL`, `OLLAMA_HOST`, etc.).
    3. Hardcoded defaults.
    """
    try:
        from api.settings_store import get_model_for_task
        return get_model_for_task(task)
    except Exception as e:
        logger.debug("get_task_config(%s) fallback: %s", task, e)
        p = f"models.{task}."
        from api.settings_store import get_setting, get_secret, _sanitize_api_key, _parse_int_setting
        return {
            "provider": get_setting(p + "provider") or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"),
            "model": get_setting(p + "model") or os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b"),
            "base_url": get_setting(p + "base_url") or os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1"),
            "api_key": _sanitize_api_key(get_secret(p + "api_key")) or os.environ.get("LOCAL_OPENAI_API_KEY", "not-needed"),
            "max_prompt_tokens": _parse_int_setting(get_setting(p + "max_prompt_tokens")),
        }


def save_task_config(
    task: str,
    provider: str,
    model: str,
    base_url: str,
    api_key: Optional[str] = None,
    max_prompt_tokens: Optional[int] = None,
) -> None:
    """Save model configuration for a task to the DB settings store and apply instantly."""
    from api.settings_store import set_setting, _sanitize_api_key

    p = f"models.{task}."
    clean_key = _sanitize_api_key(api_key) if api_key else None

    set_setting(p + "provider", provider)
    set_setting(p + "model", model)
    set_setting(p + "base_url", base_url)
    if clean_key is not None:
        set_setting(p + "api_key", clean_key, encrypt=True)
    if max_prompt_tokens is not None:
        set_setting(p + "max_prompt_tokens", str(max_prompt_tokens))

    # Sync environment variables for process readers
    if task == "cognee":
        os.environ["LLM_ENDPOINT"] = base_url
        if clean_key:
            os.environ["LLM_API_KEY"] = clean_key
    elif task == "embedder":
        os.environ["EMBEDDING_ENDPOINT"] = base_url
        if clean_key:
            os.environ["EMBEDDING_API_KEY"] = clean_key
    elif task in ("docgen", "expert", "summary"):
        os.environ["LOCAL_OPENAI_BASE_URL"] = base_url
        if clean_key:
            os.environ["LOCAL_OPENAI_API_KEY"] = clean_key

    sync_runtime_settings()


def bootstrap_config() -> None:
    """Bootstrap configuration at startup, ensuring DB defaults exist and runtime is synced."""
    try:
        from api.settings_store import bootstrap_secret_key
        bootstrap_secret_key()
    except Exception as e:
        logger.warning("bootstrap_config: secret_key bootstrap failed: %s", e)

    # Sync runtime settings immediately at startup
    sync_runtime_settings()
