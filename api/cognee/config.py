from __future__ import annotations

import asyncio
import logging
import os

from api.cognee._runtime import (
    _COGNEE_AVAILABLE,
    _default_cognee_data_root,
    _default_cognee_model,
    _default_cognee_provider,
    _default_cognee_system_root,
    _host_to_v1,
    _local_llm_host,
    _normalize_model_for_litellm,
    _resolve_embedding_dimensions,
    _resolve_graph_extraction_timeout,
    cognee,
)
from api.cognee.rate_limiter import _apply_cognee_rate_limit_patches
from api.config.ssl import apply_cognee_ssl_patch, apply_ssl_env

logger = logging.getLogger(__name__)

async def init_cognee():
    """Initializes Cognee schema/migrations on startup (non-fatal, non-blocking).

    cognee's startup API has changed across versions: older releases expose
    ``run_startup_migrations()``, newer ones (1.2.x) dropped it in favor of
    ``init()``. Try each in turn and no-op if neither exists, so a cognee
    version bump does not break startup.

    Cognee is a secondary feature (a knowledge-graph index over generated
    docs). The app must start and serve requests regardless of cognee's
    availability: docgen writes generated docs to the product DB directly,
    and the expert agent / summary fall back to artifact docs when cognee is
    unavailable. We therefore wrap the whole init (migrations + setup + stale-
    data reconciliation) in ``asyncio.wait_for`` with the ``cognee_init``
    timeout (default 120s). On timeout we log and return: cognee creates
    tables lazily on first write, so a skipped init is non-fatal.
    """
    if not _COGNEE_AVAILABLE:
        logger.warning("cognee unavailable; skipping startup migrations.")
        return
    apply_cognee_runtime_config()
    try:
        from api.config.timeout import resolve_cognee_init_timeout
        init_timeout = resolve_cognee_init_timeout()
    except Exception:  # pragma: no cover - defensive
        init_timeout = 120.0
    try:
        await asyncio.wait_for(_init_cognee_body(), timeout=init_timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Cognee init timed out after %.0fs. The app will start without a "
            "ready knowledge graph; cognee tables are created lazily on first "
            "write and the admin \"Force Refresh\" button can re-run indexing.",
            init_timeout,
        )
    except Exception as e:
        logger.warning(f"Could not complete Cognee database migrations: {e}. Falling back to SQLite/LanceDB if postgres fails.")


async def _init_cognee_body() -> None:
    """Body of :func:`init_cognee`: migrations + setup + stale-data reconcile.

    Extracted so :func:`init_cognee` can wrap it in ``asyncio.wait_for`` and
    treat a timeout as non-fatal (the app starts without a ready graph).
    """
    try:
        if hasattr(cognee, "run_startup_migrations"):
            await cognee.run_startup_migrations()
            logger.info("Cognee startup migrations completed successfully.")
        elif hasattr(cognee, "init"):
            await cognee.init()
            logger.info("Cognee init() completed successfully.")
        else:
            logger.info("Cognee exposes no run_startup_migrations/init; nothing to do.")
    except Exception as e:
        logger.warning(f"Could not complete Cognee database migrations: {e}. Falling back to SQLite/LanceDB if postgres fails.")
    # Some cognee versions expose a dedicated ``setup()`` that actually creates
    # the relational schema (``init()``/``run_startup_migrations()`` may only
    # register the engine lazily, so the very first query raises
    # ``DatabaseNotCreatedError: The database has not been created yet. Please
    # call `await setup()` first``). Call it defensively when present so the
    # stale-data reconciler below does not trip over a not-yet-created schema.
    # Best-effort and non-fatal: cognee itself creates tables on first write,
    # so a missing setup() just defers schema creation slightly.
    if hasattr(cognee, "setup"):
        try:
            await cognee.setup()
            logger.info("Cognee setup() completed (schema created/migrated if needed).")
        except Exception as e:  # pragma: no cover - depends on cognee version
            logger.debug("Cognee setup() failed (non-fatal; tables created lazily): %s", e)
    # Reconcile stale ``Data`` rows whose backing ``text_<hash>.txt`` files were
    # lost when the old ephemeral data root was wiped. Runs once per startup so a
    # rebuild no longer leaves cognify pointing at vanished files. Non-fatal.
    # Lazy import: _reconcile_stale_cognee_data lives in indexing.py, which
    # imports apply_cognee_runtime_config from this module (avoid circular import).
    from api.cognee.indexing import _reconcile_stale_cognee_data
    await _reconcile_stale_cognee_data()


# --------------------------------------------------------------------------- #
# Apply admin-configured model/embedder settings to cognee at runtime
# --------------------------------------------------------------------------- #
# cognee is a pydantic-settings app: it reads its LLM/embedding config from
# environment variables AT IMPORT TIME (LLMConfig / EmbeddingConfig are
# BaseSettings singletons). That means settings saved through the admin panel
# (the encrypted SettingORM store) NEVER reach cognee unless we push them in
# via cognee's runtime setters (``cognee.config.set_llm_*`` /
# ``set_embedding_*``), which mutate those same cached singletons.
#
# ``apply_cognee_runtime_config`` reads the admin ``models.cognee`` and
# ``models.embedder`` tasks (with env fallbacks) and pushes them onto cognee's
# runtime config. It is called from ``init_cognee`` (startup) and at the start
# of every ``add_and_index_document`` / ``query_cognee`` so an admin save takes
# effect for the very next ingestion/query without a restart. It is also the
# single source of truth for the litellm model-name prefix.
#
# Everything here is best-effort and never raises: if the settings store / DB
# is down, cognee simply keeps whatever config it already has (env defaults).
_ORIG_ACREATE_STRUCTURED_OUTPUT = None


def apply_cognee_retry_patch() -> None:
    """Patch cognee's OpenAIAdapter.acreate_structured_output tenacity retry decorator.

    cognee's default retry decorator on ``OpenAIAdapter.acreate_structured_output``
    uses ``retry_if_not_exception_type((NotFoundError, AuthenticationError))``.
    When an asyncio task is cancelled (e.g. timeout or request cancellation) or
    an unrecoverable BadRequestError is raised, tenacity catches ``CancelledError``
    or ``BadRequestError`` and RETRIES it with exponential backoff (8s -> 128s),
    flooding logs with:
      `Retrying ... as it raised CancelledError`
    and stalling background workers.

    We strip cognee's original @retry decorator by walking `__wrapped__` to the
    raw method body, save it in `_ORIG_ACREATE_STRUCTURED_OUTPUT` (idempotent),
    and wrap it with a 30-second `asyncio.wait_for` timeout and graceful
    fallback to an empty `response_model()` on timeout/error.
    """
    global _ORIG_ACREATE_STRUCTURED_OUTPUT
    if not _COGNEE_AVAILABLE or _ORIG_ACREATE_STRUCTURED_OUTPUT is not None:
        return
    try:
        import asyncio
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
            OpenAIAdapter,
        )

        curr = OpenAIAdapter.acreate_structured_output
        while hasattr(curr, "__wrapped__"):
            curr = getattr(curr, "__wrapped__")

        _ORIG_ACREATE_STRUCTURED_OUTPUT = curr

        async def _patched_acreate_structured_output(
            self, text_input: str, system_prompt: str, response_model: type, **kwargs
        ):
            # Cap text_input conservatively for graph extraction (~1500 chars,
            # roughly 400 tokens) so a single chunk stays well inside the model's
            # context window. cognee's cognify already chunks documents upstream
            # of this call; this is a per-call safety cap only.
            if len(text_input) > 1500:
                text_input = text_input[:1500] + "\n... (truncated)"

            # Drive the underlying coroutine manually instead of a bare
            # ``asyncio.wait_for``. ``wait_for`` cancels the task on timeout and
            # relies on the coroutine propagating ``CancelledError``; cognee's
            # structured-output path (litellm.acompletion -> instructor) does
            # NOT always honor cancellation promptly, so a cancelled call can
            # leave a never-awaited ``acompletion`` coroutine that Python then
            # warns about at GC time ("coroutine '...acompletion' was never
            # awaited"). By creating the coroutine explicitly and closing it on
            # any failure we guarantee the coroutine object is released cleanly.
            coro = _ORIG_ACREATE_STRUCTURED_OUTPUT(
                self, text_input, system_prompt, response_model, **kwargs
            )
            task = asyncio.ensure_future(coro)
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=_resolve_graph_extraction_timeout())
            except Exception as exc:
                # Cancel + await the task so any inner coroutine is properly
                # released rather than leaked as never-awaited.
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                logger.warning(
                    "cognee graph extraction skipped chunk due to %s (%s); returning empty model.",
                    type(exc).__name__,
                    exc,
                )
                try:
                    return response_model()
                except Exception:
                    pass
                raise exc

        OpenAIAdapter.acreate_structured_output = _patched_acreate_structured_output
        logger.info("Successfully applied idempotent cognee OpenAIAdapter retry patch.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not apply cognee OpenAIAdapter retry patch: %s", e)


def apply_cognee_runtime_config() -> None:
    """Push admin-configured model + embedder settings into cognee's runtime.

    Reads ``models.cognee`` (LLM) and ``models.embedder`` (embeddings) from the
    settings store with env fallback, normalizes the LLM model name for litellm,
    and calls cognee's runtime setters. Safe to call repeatedly; safe to call
    when cognee is unavailable (no-op). Never raises.
    """
    if not _COGNEE_AVAILABLE:
        return
    # Re-apply SSL config and retry patch so cognee calls do not retry CancelledError
    try:
        apply_ssl_env()
        apply_cognee_ssl_patch()
        apply_cognee_retry_patch()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("apply_cognee_runtime_config: SSL/retry re-apply failed: %s", e)
    # Re-assert the persistent data/system roots on the cognee config singleton.
    # The env block above (DATA_ROOT_DIRECTORY / SYSTEM_ROOT_DIRECTORY) already
    # sets these at import time, but an admin could change DEEPWIKI_ADALFLOW_ROOT
    # or a custom root at runtime, and cognee's ``config.data_root_directory`` is
    # a SETTER METHOD (not a plain attr) that mutates the cached BaseConfig.
    # Calling it here keeps the singleton in lockstep with the resolved env path
    # for every ingestion/query, not just the first import. Best-effort + safe.
    try:
        _rt_data_root = os.environ.get("DATA_ROOT_DIRECTORY") or _default_cognee_data_root
        _rt_sys_root = os.environ.get("SYSTEM_ROOT_DIRECTORY") or _default_cognee_system_root
        for _mk in (_rt_data_root, _rt_sys_root):
            try:
                os.makedirs(_mk, exist_ok=True)
            except Exception:
                pass
        _fn_dr = getattr(cognee.config, "data_root_directory", None)
        if callable(_fn_dr):
            _fn_dr(_rt_data_root)
        _fn_sr = getattr(cognee.config, "system_root_directory", None)
        if callable(_fn_sr):
            _fn_sr(_rt_sys_root)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("apply_cognee_runtime_config: root re-assert failed: %s", e)
    try:
        from api.config.settings import get_model_for_task, get_setting
        llm_cfg = get_model_for_task("cognee")
        emb_cfg = get_model_for_task("embedder")
    except Exception as e:  # pragma: no cover - settings store / DB unavailable
        logger.debug("apply_cognee_runtime_config: settings lookup failed: %s", e)
        return

    try:
        cfg = cognee.config  # type: ignore[attr-defined]
        # --- LLM ---
        # Every supported local server (Ollama, LM Studio, llama.cpp, vLLM, ...)
        # exposes an OpenAI-compatible /v1 API, so cognee's ``openai`` provider
        # (OpenAIAdapter -> POST {endpoint}/chat/completions) covers all cases.
        # The Productarium provider id (``openai_local`` / ``ollama`` / etc.) is
        # no longer branched on; the admin-configured model/base_url/api_key from
        # the ``cognee`` task always win, with env defaults as the fallback.
        model = llm_cfg.get("model") or _default_cognee_model
        base_url = llm_cfg.get("base_url") or f"{_local_llm_host}/v1"
        api_key = llm_cfg.get("api_key") or os.environ.get("LLM_API_KEY") or "not-needed"

        cognee_provider = "openai"
        model = _normalize_model_for_litellm(cognee_provider, model)
        endpoint = _host_to_v1(base_url or "")

        # Set provider first so the model prefix matches the adapter that will
        # be selected (OpenAIAdapter).
        _safe_set(cfg, "set_llm_provider", cognee_provider)
        _safe_set(cfg, "set_llm_model", model)
        if endpoint:
            _safe_set(cfg, "set_llm_endpoint", endpoint)
        _safe_set(cfg, "set_llm_api_key", api_key)

        # Export to process environment variables for cognee adapters reading os.environ directly
        if endpoint:
            os.environ["LLM_ENDPOINT"] = endpoint
        if api_key and api_key not in ("not-needed", "not_needed"):
            os.environ["LLM_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key

        # Direct mutation of LLMConfig singleton so get_llm_config() sees new values instantly
        try:
            from cognee.infrastructure.llm.config import get_llm_config
            _lc = get_llm_config()
            if endpoint:
                _lc.llm_endpoint = endpoint
            if api_key:
                _lc.llm_api_key = api_key
            _lc.llm_provider = cognee_provider
            _lc.llm_model = model
            _instructor_mode = os.environ.get("LLM_INSTRUCTOR_MODE") or "markdown_json_mode"
            _lc.llm_instructor_mode = _instructor_mode
        except Exception as e:
            logger.debug("apply_cognee_runtime_config: get_llm_config push failed: %s", e)

        # --- Embeddings ---
        # cognee's get_embedding_engine() selects the engine by embedding_provider:
        #   "openai_compatible" -> OpenAICompatibleEmbeddingEngine (openai SDK /v1/embeddings)
        #   "ollama"            -> OllamaEmbeddingEngine  (native /api/embed)
        #   anything else       -> LiteLLMEmbeddingEngine (litellm.aembedding)
        # Every supported local server exposes the OpenAI-compatible
        # /v1/embeddings endpoint (Ollama's :11434 included), so the single
        # ``openai_compatible`` provider covers all cases. CRITICAL: do NOT use
        # ``openai`` here -- it falls through to LiteLLMEmbeddingEngine, whose
        # __init__ calls tiktoken.encoding_for_model(<nomic model>) and raises
        # KeyError. ``openai_compatible`` uses the OpenAI SDK directly and falls
        # back to cl100k_base when transformers is not installed.
        emb_base = (emb_cfg.get("base_url") or "").rstrip("/")
        emb_model = emb_cfg.get("model") or os.environ.get("EMBEDDING_MODEL") or "text-embedding-nomic-embed-text-v1.5"
        emb_key = emb_cfg.get("api_key") or api_key
        emb_provider = "openai_compatible"
        emb_endpoint = _host_to_v1(emb_base) if emb_base else f"{_local_llm_host}/v1"
        emb_dims = _resolve_embedding_dimensions(emb_model)
        # Default tokenizer to empty string so local tiktoken is used
        # without making external network requests to huggingface.co
        emb_tok = os.environ.get("HUGGINGFACE_TOKENIZER", "")

        _safe_set(cfg, "set_embedding_provider", emb_provider)
        _safe_set(cfg, "set_embedding_model", emb_model)
        _safe_set(cfg, "set_embedding_endpoint", emb_endpoint)
        _safe_set(cfg, "set_embedding_api_key", emb_key)
        _safe_set(cfg, "set_embedding_dimensions", emb_dims)
        _safe_set_embedding_dict(cfg, {"huggingface_tokenizer": emb_tok})

        # Export to process environment variables for cognee engines reading os.environ directly
        if emb_endpoint:
            os.environ["EMBEDDING_ENDPOINT"] = emb_endpoint
        if emb_key:
            os.environ["EMBEDDING_API_KEY"] = emb_key
        if emb_model:
            os.environ["EMBEDDING_MODEL"] = emb_model

        # Direct mutation of EmbeddingConfig singleton so get_embedding_config() sees new values instantly
        try:
            from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
            _ec = get_embedding_config()
            if emb_endpoint:
                _ec.embedding_endpoint = emb_endpoint
            if emb_key:
                _ec.embedding_api_key = emb_key
            if emb_model:
                _ec.embedding_model = emb_model
            _ec.embedding_provider = emb_provider
            _ec.embedding_dimensions = emb_dims
            _ec.huggingface_tokenizer = emb_tok
        except Exception as e:
            logger.debug("apply_cognee_runtime_config: get_embedding_config push failed: %s", e)

        # Invalidate cognee's ``create_embedding_engine`` lru_cache so a changed
        # embedder provider/model/endpoint takes effect on the next call instead
        # of reusing a stale (e.g. LiteLLM) engine built from the old config.
        try:
            from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import create_embedding_engine as _cee
            _cc = getattr(_cee, "cache_clear", None)
            if callable(_cc):
                _cc()
        except Exception as e:  # pragma: no cover - depends on cognee version
            logger.debug("apply_cognee_runtime_config: embedding cache_clear failed: %s", e)

        logger.info(
            "cognee runtime config applied: LLM provider=%s model=%s endpoint=%s; "
            "embedder provider=%s model=%s endpoint=%s",
            cognee_provider, model, endpoint, emb_provider, emb_model, emb_endpoint,
        )
        _apply_cognee_rate_limit_patches()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("apply_cognee_runtime_config failed (non-fatal): %s", e)


def _safe_set(cfg, setter_name: str, value) -> None:
    """Call ``cfg.<setter_name>(value)`` if the setter exists; swallow errors."""
    fn = getattr(cfg, setter_name, None)
    if not callable(fn):
        return
    try:
        fn(value)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("cognee %s(%r) failed: %s", setter_name, value, e)


def _safe_set_embedding_dict(cfg, config_dict: dict) -> None:
    """Bulk-set embedding config attrs (e.g. huggingface_tokenizer)."""
    fn = getattr(cfg, "set_embedding_config", None)
    if not callable(fn):
        return
    try:
        fn(config_dict)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("cognee set_embedding_config(%r) failed: %s", config_dict, e)
