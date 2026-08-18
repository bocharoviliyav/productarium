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
    3. Clears model context window cache in `api.utils`.
    4. Exports admin-store timeout overrides to their canonical env vars so
       subprocess / module-level readers (e.g. fast-rlm's Pyodide REPL, which
       reads RLM_API_TIMEOUT_MS from the process environment and cannot reach
       host Python) see admin-set values without a restart.
    """
    try:
        from api.utils import _MODEL_CTX_CACHE
        _MODEL_CTX_CACHE.clear()
    except Exception:
        pass

    try:
        from api.config.settings import get_model_for_task
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
        from api.cognee import apply_cognee_runtime_config
        apply_cognee_runtime_config()
    except Exception as e:
        logger.debug("sync_runtime_settings: apply_cognee_runtime_config skipped: %s", e)

    try:
        from api.config.timeout import sync_timeout_env
        sync_timeout_env()
    except Exception as e:
        logger.debug("sync_runtime_settings: sync_timeout_env skipped: %s", e)

    # Invalidate the cached memory backend instance so an admin switch of
    # ``memory.backend`` (pgvector <-> cognee) takes effect on the next call
    # without a process restart.
    try:
        from api.memory.resolver import reset_memory_backend_cache
        reset_memory_backend_cache()
    except Exception as e:
        logger.debug("sync_runtime_settings: reset_memory_backend_cache skipped: %s", e)

    logger.info("Configuration Abstraction Layer: Synchronized runtime settings across process.")


def get_task_config(task: str) -> Dict[str, Optional[str]]:
    """Get model configuration for a task with HIGHEST PRECEDENCE given to DB settings.

    Precedence:
    1. DB SettingORM (`models.<task>.*`) — Admin UI saves.
    2. Process Environment Variables (`LOCAL_OPENAI_BASE_URL`, etc.).
    3. Hardcoded defaults.
    """
    try:
        from api.config.settings import get_model_for_task
        return get_model_for_task(task)
    except Exception as e:
        logger.debug("get_task_config(%s) fallback: %s", task, e)
        p = f"models.{task}."
        from api.config.settings import get_setting, get_secret, _sanitize_api_key, _parse_int_setting
        return {
            "model": get_setting(p + "model") or "qwen/qwen3.6-27b",
            "base_url": get_setting(p + "base_url") or os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1"),
            "api_key": _sanitize_api_key(get_secret(p + "api_key")) or os.environ.get("LOCAL_OPENAI_API_KEY", "not-needed"),
            "max_prompt_tokens": _parse_int_setting(get_setting(p + "max_prompt_tokens")),
        }


def bootstrap_config() -> None:
    """Bootstrap configuration at startup, ensuring DB defaults exist and runtime is synced."""
    try:
        from api.config.settings import bootstrap_secret_key
        bootstrap_secret_key()
    except Exception as e:
        logger.warning("bootstrap_config: secret_key bootstrap failed: %s", e)

    # Sync runtime settings immediately at startup
    sync_runtime_settings()
