import os
import logging
import asyncio

logger = logging.getLogger(__name__)

# Apply SSL/TLS configuration (corporate CA bundle / skip-verify) before cognee
# is imported. cognee's aiohttp adapters build an ssl.SSLContext via
# ``create_secure_ssl_context`` -> ``ssl.create_default_context()``, which honors
# the ``SSL_CERT_FILE`` env var, so setting it here (from the admin panel / env)
# makes cognee trust a corporate root cert. See api/ssl_config.py.
from api.ssl_config import apply_ssl_env, apply_cognee_ssl_patch
apply_ssl_env()

# Postgres settings for cognee metadata/graph storage.
# Defaults assume a local dev run (localhost); inside Docker, docker-compose
# sets DB_HOST=postgres (the compose service name).
DB_PROVIDER = os.environ.get("DB_PROVIDER") or "postgres"
DB_HOST = os.environ.get("DB_HOST") or "localhost"
DB_PORT = os.environ.get("DB_PORT") or "5432"
DB_NAME = os.environ.get("DB_NAME") or "cognee_db"
DB_USERNAME = os.environ.get("DB_USERNAME") or "cognee"
DB_PASSWORD = os.environ.get("DB_PASSWORD") or "cognee"

VECTOR_DB_PROVIDER = os.environ.get("VECTOR_DB_PROVIDER") or "pgvector"

# Set environment variables for cognee
os.environ["DB_PROVIDER"] = DB_PROVIDER
os.environ["DB_HOST"] = DB_HOST
os.environ["DB_PORT"] = DB_PORT
os.environ["DB_NAME"] = DB_NAME
os.environ["DB_USERNAME"] = DB_USERNAME
os.environ["DB_PASSWORD"] = DB_PASSWORD
os.environ["VECTOR_DB_PROVIDER"] = VECTOR_DB_PROVIDER


# --------------------------------------------------------------------------- #
# litellm model-name normalization
# --------------------------------------------------------------------------- #
# cognee's OpenAI/Custom adapters call ``litellm.acompletion(model=<name>)``;
# litellm resolves the provider from the model-name PREFIX, not from a separate
# provider field. A bare local model like ``qwen/qwen3.6-27b`` is therefore
# misread as provider ``qwen`` and raises
# ``litellm.BadRequestError: LLM Provider NOT provided``.
#
# Fix: prefix the model with the litellm provider segment that matches cognee's
# ``llm_provider``:
#   provider "ollama"      -> "ollama/<model>"   (used by OllamaAPIAdapter,
#                                                    which calls the OpenAI SDK
#                                                    directly and tolerates the
#                                                    prefix)
#   provider "openai"/
#   "openai_local"/
#   "custom"             -> "openai/<model>"   (litellm openai route with a
#                                                    custom api_base)
#
# Ollama's own adapter ignores the prefix (it talks to the OpenAI-compatible
# /v1 endpoint directly), so it is safe to add it there too; what matters is
# that litellm never sees an unprefixed local model name on the openai route.
#
# ``_strip_provider_prefix`` first removes any existing ``<provider>/`` segment
# so repeated applies never double-prefix (``openai/openai/qwen3:8b``).
# litellm's known routing prefixes are an allow-list; a stray slash inside a
# model name (rare) is left intact by only stripping a *leading* known prefix.
_LITELLM_PROVIDER_PREFIXES = (
    "openai/",
    "ollama/",
    "anthropic/",
    "gemini/",
    "mistral/",
    "azure/",
    "bedrock/",
    "huggingface/",
    "together_ai/",
)


def _strip_provider_prefix(model: str) -> str:
    """Remove a leading litellm provider prefix (``openai/``, ``ollama/``, …)."""
    if not model:
        return model
    m = model.strip()
    low = m.lower()
    for pfx in _LITELLM_PROVIDER_PREFIXES:
        if low.startswith(pfx):
            return m[len(pfx):]
    return m


def _normalize_model_for_litellm(provider: str, model: str) -> str:
    """Return ``model`` with the litellm provider prefix for ``provider``.

    - ``ollama``        -> ``ollama/<model>``
    - anything else that ends up on the OpenAI/Custom route (openai /
      openai_local / custom) -> ``openai/<model>``

    Idempotent: an already-prefixed name is not double-prefixed.
    """
    bare = _strip_provider_prefix(model or "")
    if not bare:
        return model or ""
    prov = (provider or "").strip().lower()
    if prov == "ollama":
        return f"ollama/{bare}"
    # openai / openai_local / custom / unknown -> route through litellm's
    # openai-compatible path with a custom api_base.
    return f"openai/{bare}"


def _host_to_v1(host: str) -> str:
    """Normalize a local LLM host URL to an OpenAI ``/v1`` base URL.

    Strips a trailing ``/v1`` (if present) then re-appends it, so both
    ``http://localhost:11434`` and ``http://localhost:1234/v1`` resolve to a
    ``.../v1`` OpenAI-compatible base URL.
    """
    h = (host or "").rstrip("/")
    if h.endswith("/v1"):
        h = h[:-3]
    return f"{h}/v1" if h else h


def _resolve_default_provider() -> str:
    """Best-effort default cognee LLM provider with NO .env.

    If ``LOCAL_OPENAI_BASE_URL`` is set (LM Studio / llama.cpp / vLLM) we use
    the OpenAI-compatible route (``openai`` provider). Otherwise we assume a
    local Ollama on ``OLLAMA_HOST`` (default :11434) and use the ``ollama``
    provider, whose adapter talks to Ollama's native OpenAI-compatible ``/v1``
    endpoint directly (no litellm provider resolution involved).
    """
    if os.environ.get("LOCAL_OPENAI_BASE_URL"):
        return "openai"
    return "ollama"


# --- Local LLM configuration for cognee ---
# cognee talks to an OpenAI-compatible LLM endpoint. We point it at the SAME
# local server the rest of DeepWiki uses (Ollama by default, but also LM
# Studio / llama.cpp / vLLM via LOCAL_OPENAI_BASE_URL). cognee's LLM endpoint
# must use the OpenAI-compatible path ("/v1"). A non-empty LLM_API_KEY is
# required even by local servers that ignore it; we default to "not-needed".
# NB: ``OLLAMA_HOST`` historically carried the host URL for every provider;
# we keep that for backwards compat but prefer LOCAL_OPENAI_BASE_URL when set.
_local_llm_host = (
    os.environ.get("LOCAL_OPENAI_BASE_URL")
    or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
).rstrip("/")
# Normalize to a bare host (strip a trailing "/v1") so we can append the right
# path per cognee's expectations below.
_local_llm_host = _local_llm_host[:-3] if _local_llm_host.endswith("/v1") else _local_llm_host

# The default cognee provider must MATCH the local server we point at, so that
# the right adapter (OllamaAPIAdapter vs OpenAIAdapter) is selected and the
# model name is prefixed correctly for litellm. With an empty .env this
# resolves to ``ollama`` (the fully-local default).
_default_cognee_provider = _resolve_default_provider()
os.environ.setdefault("LLM_PROVIDER", _default_cognee_provider)
os.environ.setdefault("LLM_ENDPOINT", f"{_local_llm_host}/v1")
# ``LLM_MODEL`` must carry the litellm provider prefix on the openai route;
# the ollama adapter tolerates the prefix too. Default bare model is the same
# RLM/local default, then normalized.
_default_cognee_model = os.environ.get("RLM_MODEL_NAME") or os.environ.get("LLM_MODEL") or "qwen3:8b"
os.environ.setdefault("LLM_MODEL", _normalize_model_for_litellm(_default_cognee_provider, _default_cognee_model))
if not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = os.environ.get("LOCAL_OPENAI_API_KEY") or os.environ.get("OLLAMA_API_KEY") or "not-needed"

# --- Local embeddings configuration for cognee ---
# cognee requires embeddings to be configured alongside the LLM; if only one is
# set it silently falls back to OpenAI (cloud). For the "openai" embedding
# provider cognee hits EMBEDDING_ENDPOINT with the OpenAI embeddings shape
# (POST {endpoint}/embeddings) -- which is exactly what LM Studio / llama.cpp /
# vLLM expose. The Ollama provider instead uses the NATIVE "/api/embed" path.
#
# We default to the OpenAI-compatible provider so the same LM Studio server that
# serves chat also serves embeddings (its nomic-embed-text model). When the user
# explicitly keeps Ollama (OLLAMA_HOST=:11434 and no LOCAL_OPENAI_BASE_URL), the
# native Ollama embedding provider is used instead. cognee additionally
# requires EMBEDDING_DIMENSIONS and HUGGINGFACE_TOKENIZER, and its config
# validator raises if ANY of {EMBEDDING_PROVIDER, EMBEDDING_MODEL,
# EMBEDDING_DIMENSIONS, HUGGINGFACE_TOKENIZER} is set without the others, so we
# set them all together here (each via setdefault so explicit env/ compose wins).
_use_ollama_native_embed = (
    ":11434" in _local_llm_host
    and not os.environ.get("LOCAL_OPENAI_BASE_URL")
)
if _use_ollama_native_embed:
    os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
    os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text")
    os.environ.setdefault("EMBEDDING_ENDPOINT", f"{_local_llm_host}/api/embed")
else:
    # CRITICAL: use ``openai_compatible`` (NOT ``openai``) for non-Ollama local
    # servers (LM Studio / llama.cpp / vLLM). cognee's embedding dispatch in
    # ``create_embedding_engine`` routes ONLY ``openai_compatible`` and ``ollama``
    # to safe engines; ANY other value (incl. ``openai``) falls through to
    # ``LiteLLMEmbeddingEngine``, whose ``__init__`` calls
    # ``tiktoken.encoding_for_model(embedding_model)``. For a nomic-embed model
    # name tiktoken has no mapping -> ``KeyError: Could not automatically map
    # text-embedding-nomic-embed-text-v1.5 to a tokeniser``, which breaks every
    # cognee.add()/cognify(). ``OpenAICompatibleEmbeddingEngine`` instead uses
    # the OpenAI SDK directly and falls back to ``cl100k_base`` when the
    # HuggingFace tokenizer (``transformers``) isn't installed.
    os.environ.setdefault("EMBEDDING_PROVIDER", "openai_compatible")
    os.environ.setdefault("EMBEDDING_MODEL", os.environ.get("EMBEDDING_MODEL") or "text-embedding-nomic-embed-text-v1.5")
    os.environ.setdefault("EMBEDDING_ENDPOINT", f"{_local_llm_host}/v1")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ.setdefault("EMBEDDING_API_KEY", os.environ.get("LLM_API_KEY") or "not-needed")
os.environ.setdefault("HUGGINGFACE_TOKENIZER", "nomic-ai/nomic-embed-text-v1.5")

# --- Structured-output instructor mode (cognify graph extraction) ------------
# cognee's ``cognify`` extracts a knowledge graph by calling the LLM with a
# Pydantic ``response_model`` via the ``instructor`` library. The mode is set
# by ``LLMConfig.llm_instructor_mode`` (env ``LLM_INSTRUCTOR_MODE``) and passed
# to ``OpenAIAdapter(instructor_mode=...)``. cognee's default is
# ``json_schema_mode`` (== ``instructor.Mode.JSON_SCHEMA``), which sends
# ``response_format: {type: json_schema, ...}``.
#
# Problem: many local OpenAI-compatible gateways (LM Studio, llama.cpp, strict
# corporate proxies) do NOT implement the ``response_format: json_schema`` or
# function-calling (``tool_choice``) structured-output paths and reject them
# with ``BadRequestError`` (e.g. ``Invalid tool_choice type: 'object'.
# Supported string values: none, auto, required``). cognee then retries forever
# inside ``instructor`` (exponential backoff 8s -> 128s), stalling cognify.
#
# Fix: default to ``markdown_json_mode`` (== ``instructor.Mode.MD_JSON``), which
# appends a "return JSON" instruction and parses a JSON block out of the plain
# text completion. Empirically this is the ONLY mode that works against LM
# Studio (JSON_SCHEMA / JSON / TOOLS / TOOLS_STRICT all raise BadRequestError).
# Still overridable via env for gateways that do support json_schema.
os.environ.setdefault("LLM_INSTRUCTOR_MODE", "markdown_json_mode")

# --- Graph database provider (cognee knowledge-graph engine) -----------------
# cognee's GraphConfig defaults ``graph_database_provider`` to ``ladybug`` (an
# embedded LanceDB-like store). On first use LadybugDB tries to DOWNLOAD its
# ``json`` extension from ``extension.ladybugdb.com``; in an offline / locked-
# down container that fails with
# ``IO exception: Failed to load library .../libjson.lbug_extension`` and
# breaks ``cognify``. Default to ``postgres`` instead -- the same Postgres we
# already use for relational + vector storage. GRAPH_DATABASE_* credentials
# fall back to the relational DB_* config when unset, so no extra env needed.
os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "postgres")

# --- Skip cognee's blocking LLM connection test (item 10 bug fix) -------------
# cognee runs a blocking LLM connection test during cognify() that times out
# after ~30s when pointed at a local Ollama that is slow to answer. Setting
# COGNEE_SKIP_CONNECTION_TEST=true (the default here) disables that test so
# cognify() starts immediately. Overridable via env. Also best-effort: shrink
# the cognee LLM connection timeout if cognee honors it.
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
os.environ.setdefault("COGNEE_LLM_CONNECTION_TIMEOUT", "5")
logger.info(
    "cognee connection test %s (COGNEE_SKIP_CONNECTION_TEST=%s).",
    "skipped" if os.environ.get("COGNEE_SKIP_CONNECTION_TEST", "true").lower()
    in ("1", "true", "t", "yes") else "enabled",
    os.environ.get("COGNEE_SKIP_CONNECTION_TEST"),
)

# --- Single-user posture (disable cognee's multi-tenant access control) -------
# cognee 1.2.x defaults to ENABLE_BACKEND_ACCESS_CONTROL=true, which turns on
# per-user/per-tenant dataset isolation + mandatory auth. Productarium runs as a
# single local instance with its OWN auth layer and treats cognee's graph as one
# shared knowledge base (one dataset per product), so multi-tenant isolation is
# neither needed nor desirable. With EBAC on, cognee.add()/cognify()/recall()
# resolve a default user that has no tenant and then crash inside dataset/vector
# routing with ``cannot read properties of undefined tenantID`` (and, because the
# default vector provider under EBAC is LanceDB, a missing
# VECTOR_DATASET_DATABASE_HANDLER would add 401/403s). Switching to single-user
# mode (shared DB, no auth, no tenant routing) resolves all of that. Still
# overridable via env for anyone who genuinely wants multi-tenant cognee.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

# --- Persistent data + system root directories ------------------------------
# cognee's BaseConfig defaults ``data_root_directory`` to
# ``<cognee package>/.data_storage`` and ``system_root_directory`` to
# ``<cognee package>/.cognee_system``. Both live INSIDE the installed package,
# which is EPHEMERAL inside Docker (wiped on every ``docker-compose build``).
# The Postgres ``Data`` table (raw_data_location column) is on a PERSISTENT
# volume, so after a rebuild the table still references ``text_<hash>.txt``
# files that no longer exist -> ``cognify`` raises
# ``File not found: .../.data_storage/text_<hash>.txt`` for every stale row.
#
# Fix: point both roots at a path inside the MOUNTED ``~/.adalflow`` volume
# (the same volume that already persists repos + the settings secret key), so
# the backing files survive rebuilds in lockstep with the Postgres metadata.
# ``DATA_ROOT_DIRECTORY`` / ``SYSTEM_ROOT_DIRECTORY`` are the pydantic-settings
# env overrides for ``BaseConfig.data_root_directory`` / ``system_root_directory``
# and are read at cognee import time (set above, before ``import cognee``).
# Honors DEEPWIKI_CONFIG_DIR / a custom ADALFLOW root when set.
_adalflow_root = os.environ.get("DEEPWIKI_ADALFLOW_ROOT") or os.path.expanduser("~/.adalflow")
_default_cognee_data_root = os.path.join(_adalflow_root, "cognee_data")
_default_cognee_system_root = os.path.join(_adalflow_root, "cognee_system")
os.environ.setdefault("DATA_ROOT_DIRECTORY", _default_cognee_data_root)
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", _default_cognee_system_root)
try:
    os.makedirs(_default_cognee_data_root, exist_ok=True)
    os.makedirs(_default_cognee_system_root, exist_ok=True)
except Exception as _e:  # pragma: no cover - defensive
    logger.debug("could not pre-create cognee data/system roots: %s", _e)
logger.info(
    "cognee storage roots: data=%s system=%s",
    os.environ.get("DATA_ROOT_DIRECTORY"), os.environ.get("SYSTEM_ROOT_DIRECTORY"),
)

# Import cognee defensively: if the package is unavailable (or fails to import)
# the module still loads and every cognee entrypoint degrades gracefully.
try:
    import cognee  # type: ignore
    _COGNEE_AVAILABLE = True
except Exception as _cognee_import_err:  # pragma: no cover - dep missing
    cognee = None  # type: ignore
    _COGNEE_AVAILABLE = False
    logger.warning(
        "cognee package could not be imported; cognee features disabled: %s",
        _cognee_import_err,
    )

# Apply the cognee SSL monkeypatch now that cognee is imported. Covers the
# skip-verify case (ssl.verify=false) by patching
# ``cognee.shared.utils.create_secure_ssl_context`` to return an unverified
# context. The CA-bundle case is already handled by SSL_CERT_FILE set in
# apply_ssl_env() above. No-op when cognee is unavailable.
apply_cognee_ssl_patch()


async def init_cognee():
    """Initializes Cognee schema/migrations on startup (non-fatal).

    cognee's startup API has changed across versions: older releases expose
    ``run_startup_migrations()``, newer ones (1.2.x) dropped it in favor of
    ``init()``. Try each in turn and no-op if neither exists, so a cognee
    version bump does not break startup.
    """
    if not _COGNEE_AVAILABLE:
        logger.warning("cognee unavailable; skipping startup migrations.")
        return
    apply_cognee_runtime_config()
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
    # Reconcile stale ``Data`` rows whose backing ``text_<hash>.txt`` files were
    # lost when the old ephemeral data root was wiped. Runs once per startup so a
    # rebuild no longer leaves cognify pointing at vanished files. Non-fatal.
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
def apply_cognee_runtime_config() -> None:
    """Push admin-configured model + embedder settings into cognee's runtime.

    Reads ``models.cognee`` (LLM) and ``models.embedder`` (embeddings) from the
    settings store with env fallback, normalizes the LLM model name for litellm,
    and calls cognee's runtime setters. Safe to call repeatedly; safe to call
    when cognee is unavailable (no-op). Never raises.
    """
    if not _COGNEE_AVAILABLE:
        return
    # Re-apply SSL config so an admin save of ssl.ca_bundle / ssl.verify takes
    # effect for the next cognee call: re-export the CA env vars and refresh the
    # cognee aiohttp SSL monkeypatch (skip-verify toggle).
    try:
        apply_ssl_env()
        apply_cognee_ssl_patch()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("apply_cognee_runtime_config: SSL re-apply failed: %s", e)
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
        from api.settings_store import get_model_for_task, get_setting
        llm_cfg = get_model_for_task("cognee")
        emb_cfg = get_model_for_task("embedder")
    except Exception as e:  # pragma: no cover - settings store / DB unavailable
        logger.debug("apply_cognee_runtime_config: settings lookup failed: %s", e)
        return

    try:
        cfg = cognee.config  # type: ignore[attr-defined]
        # --- LLM ---
        # ``get_model_for_task`` always returns a provider (defaulting to
        # ``openai_local`` i.e. LM Studio :1234). But with an EMPTY .env the
        # real local server is Ollama (:11434), so the openai_local default
        # would point cognee at a non-existent LM Studio. Detect this: if NEITHER
        # the admin store NOR ``DEEPWIKI_DEFAULT_PROVIDER`` explicitly set the
        # cognee task's provider, use ``_default_cognee_provider`` (which
        # resolves to ``ollama`` for an empty .env). This makes the system work
        # out-of-the-box with no .env and no admin config.
        explicit_provider = get_setting("models.cognee.provider") or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER")
        if explicit_provider:
            provider = explicit_provider.strip().lower()
            model = llm_cfg.get("model") or ""
            base_url = llm_cfg.get("base_url") or ""
            api_key = llm_cfg.get("api_key") or "not-needed"
        else:
            provider = _default_cognee_provider
            model = _default_cognee_model
            base_url = f"{_local_llm_host}/v1"
            api_key = os.environ.get("LLM_API_KEY") or "not-needed"

        # Map Productarium provider ids to cognee's LLMProvider enum values.
        # cognee supports: openai, ollama, anthropic, custom, gemini, mistral,
        # azure, bedrock, llama_cpp. ``openai_local`` (LM Studio/llama.cpp/vLLM)
        # is NOT a cognee provider -- route it through the openai adapter with a
        # custom endpoint, which is exactly what those servers expose.
        if provider in ("openai_local", "openai"):
            cognee_provider = "openai"
        elif provider in ("ollama",):
            cognee_provider = "ollama"
        else:
            cognee_provider = provider  # custom / anthropic / etc.
        model = _normalize_model_for_litellm(cognee_provider, model)
        endpoint = _host_to_v1(base_url or "")

        # Set provider first so the model prefix matches the adapter that will
        # be selected (OllamaAPIAdapter vs OpenAIAdapter).
        _safe_set(cfg, "set_llm_provider", cognee_provider)
        _safe_set(cfg, "set_llm_model", model)
        if endpoint:
            _safe_set(cfg, "set_llm_endpoint", endpoint)
        _safe_set(cfg, "set_llm_api_key", api_key)

        # --- Structured-output instructor mode ---
        # cognee's cognify extracts a graph via ``instructor`` with a Pydantic
        # ``response_model``. The mode comes from ``LLMConfig.llm_instructor_mode``
        # (env ``LLM_INSTRUCTOR_MODE``) -> ``OpenAIAdapter(instructor_mode=...)``.
        # The env default (set above) is ``markdown_json_mode`` because that's
        # the only mode local OpenAI-compatible gateways (LM Studio) accept;
        # JSON_SCHEMA/TOOLS raise ``BadRequestError: Invalid tool_choice``.
        # cognee has no ``set_llm_instructor_mode`` setter, so we mutate the
        # ``LLMConfig`` singleton directly. ``get_llm_client`` reads the config
        # fresh on every call (it is NOT lru_cache), so the new mode takes
        # effect on the next cognify without any cache invalidation.
        try:
            from cognee.infrastructure.llm.config import get_llm_config
            _instructor_mode = os.environ.get("LLM_INSTRUCTOR_MODE") or "markdown_json_mode"
            _lc = get_llm_config()
            if getattr(_lc, "llm_instructor_mode", "") != _instructor_mode:
                _lc.llm_instructor_mode = _instructor_mode
                logger.info("cognee LLM instructor_mode set to %r.", _instructor_mode)
        except Exception as e:  # pragma: no cover - depends on cognee version
            logger.debug("apply_cognee_runtime_config: instructor_mode push failed: %s", e)

        # --- Embeddings ---
        # The embedder task uses the same provider vocabulary as the LLM task.
        # cognee's get_embedding_engine() selects the engine by embedding_provider:
        #   "ollama"            -> OllamaEmbeddingEngine  (native /api/embed)
        #   "openai_compatible" -> OpenAICompatibleEmbeddingEngine (openai SDK /v1/embeddings)
        #   anything else       -> LiteLLMEmbeddingEngine (litellm.aembedding)
        # For local servers:
        #   - Ollama host  -> provider "ollama", endpoint <host>/api/embed
        #   - LM Studio etc -> provider "openai_compatible", endpoint <host>/v1
        #
        # Same empty-.env logic as the LLM block: ``get_model_for_task`` defaults
        # to ``openai_local`` (LM Studio :1234), but with no .env the real
        # embedder is Ollama. Only trust the task config when the admin or env
        # explicitly set the embedder provider; otherwise use _default_cognee_provider.
        explicit_emb_provider = get_setting("models.embedder.provider") or os.environ.get("DEEPWIKI_EMBEDDER_TYPE") or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER")
        if explicit_emb_provider:
            emb_provider_raw = explicit_emb_provider.strip().lower()
            emb_base = (emb_cfg.get("base_url") or "").rstrip("/")
            emb_model = emb_cfg.get("model") or ""
            emb_key = emb_cfg.get("api_key") or api_key
        else:
            emb_provider_raw = _default_cognee_provider
            emb_base = f"{_local_llm_host}".rstrip("/")
            emb_model = ""
            emb_key = api_key

        if emb_provider_raw in ("ollama",):
            emb_provider = "ollama"
            emb_endpoint = emb_base if emb_base else f"{_local_llm_host}/api/embed"
            # Ollama embeds via /api/embed; if the admin pasted a /v1 URL, swap it.
            if emb_endpoint.endswith("/v1"):
                emb_endpoint = emb_endpoint[:-3] + "/api/embed"
            emb_dims = 768
            emb_tok = "nomic-ai/nomic-embed-text-v1.5"
            if not emb_model:
                emb_model = "nomic-embed-text"
        else:
            # openai_local / openai -> use the openai-compatible engine, which
            # uses the openai SDK directly (avoids litellm embedding quirks).
            emb_provider = "openai_compatible"
            emb_endpoint = _host_to_v1(emb_base) if emb_base else f"{_local_llm_host}/v1"
            emb_dims = 768
            emb_tok = "nomic-ai/nomic-embed-text-v1.5"
            if not emb_model:
                emb_model = "text-embedding-nomic-embed-text-v1.5"

        _safe_set(cfg, "set_embedding_provider", emb_provider)
        _safe_set(cfg, "set_embedding_model", emb_model)
        _safe_set(cfg, "set_embedding_endpoint", emb_endpoint)
        _safe_set(cfg, "set_embedding_api_key", emb_key)
        _safe_set(cfg, "set_embedding_dimensions", emb_dims)
        # huggingface_tokenizer is required by cognee's Ollama tokenizer; set it
        # via the generic embedding config dict setter.
        _safe_set_embedding_dict(cfg, {"huggingface_tokenizer": emb_tok})
        # Invalidate cognee's ``create_embedding_engine`` lru_cache so a changed
        # embedder provider/model/endpoint takes effect on the next call instead
        # of reusing a stale (e.g. LiteLLM) engine built from the old config.
        # Best-effort: the function is @lru_cache; cache_clear() resets it.
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


async def _empty_cognee_dataset(dataset_name: str) -> bool:
    """Clear all data + graph for a cognee dataset by name. Returns True on success.

    Uses the CURRENT cognee API (``cognee.datasets.empty_dataset``), which takes
    a dataset UUID. The legacy ``cognee.delete([name])`` is DEPRECATED and its
    real signature is ``delete(data_id, dataset_id)`` -- passing a list as
    ``data_id`` silently fails, so we do NOT use it. Resolves the name to a
    dataset row via ``cognee.datasets.list_datasets`` (one round-trip, no
    user_id needed because EBAC is off in our single-user posture).

    Fallback: if ``empty_dataset`` raises (e.g. the embedding/tokenizer
    machinery it drags in for graph cleanup hits a KeyError), we fall back to a
    DIRECT SQLAlchemy delete of the ``data`` + ``dataset_data`` junction rows
    for that dataset. That is enough to let a re-add succeed (cognify only
    needs the ``data`` rows gone); any orphaned graph nodes from a PARTIALLY
    cognified dataset are harmless (they reference a now-absent dataset).
    Best-effort: returns False only if BOTH paths fail.
    """
    try:
        datasets_obj = getattr(cognee, "datasets", None)
        if datasets_obj is None:
            return False
        datasets = await datasets_obj.list_datasets()
        target = None
        for d in datasets:
            if getattr(d, "name", None) == dataset_name:
                target = d
                break
        if target is None:
            # Nothing to clear -- treat as success (dataset simply doesn't exist yet).
            return True
        empty_fn = getattr(datasets_obj, "empty_dataset", None)
        if callable(empty_fn):
            try:
                await empty_fn(target.id)
                return True
            except Exception as e:
                # ``empty_dataset`` pulls in graph cleanup -> embedding pipeline,
                # which can fail with a tokenizer KeyError when the embedder is
                # misconfigured. Fall through to the direct-DB fallback below.
                logger.warning(
                    "_empty_cognee_dataset(%r): empty_dataset failed (%s); "
                    "falling back to direct DB delete of data rows.",
                    dataset_name, e,
                )
        # Fallback: direct DB delete of this dataset's data + junction rows.
        return await _direct_delete_dataset_data(target.id)
    except Exception as e:  # pragma: no cover - depends on cognee version
        logger.warning("_empty_cognee_dataset(%r) failed: %s", dataset_name, e)
        return False


async def _direct_delete_dataset_data(dataset_id) -> bool:
    """Direct SQLAlchemy delete of a dataset's ``data`` + ``dataset_data`` rows.

    Bypasses ``cognee.datasets.delete_data`` / ``empty_dataset`` (which pull in
    the embedding pipeline and can fail with a tokenizer KeyError). Enough to
    let a re-add succeed -- cognify only needs the ``data`` rows gone. Returns
    True on success, False on error. Does NOT touch graph nodes (orphans are
    harmless: they reference an absent dataset).
    """
    try:
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.data.models import Data
        from cognee.modules.data.models.Dataset import DatasetData
        from sqlalchemy import delete, select
        eng = get_relational_engine()
        async with eng.get_async_session() as session:
            # Collect this dataset's data_ids via the junction, then delete.
            res = await session.scalars(
                select(DatasetData.data_id).where(DatasetData.dataset_id == dataset_id)
            )
            data_ids = list(res.all())
            if data_ids:
                await session.execute(
                    delete(DatasetData).where(DatasetData.dataset_id == dataset_id)
                )
                await session.execute(
                    delete(Data).where(Data.id.in_(data_ids))
                )
            await session.commit()
        return True
    except Exception as e:  # pragma: no cover - depends on cognee version
        logger.warning("_direct_delete_dataset_data(%s) failed: %s", dataset_id, e)
        return False


def _resolve_raw_data_location_path(raw_data_location: str) -> str:
    """Normalize a cognee ``raw_data_location`` into a plain filesystem path.

    cognee 1.2.x stores ``raw_data_location`` WITH a ``file://`` URI scheme
    prefix (e.g. ``file:///root/.adalflow/cognee_data/text_<hash>.txt``), but
    older rows and some code paths store a bare absolute path. ``os.path.exists``
    returns False on a ``file://`` string, so we MUST strip the scheme before
    any existence check -- otherwise the reconciler would prune every healthy
    row. Returns the original string if it's not a ``file://`` URI.
    """
    if not raw_data_location:
        return raw_data_location
    if raw_data_location.startswith("file://"):
        return raw_data_location[len("file://"):]
    return raw_data_location


async def _reconcile_stale_cognee_data() -> None:
    """One-time startup cleanup of cognee ``Data`` rows whose backing file is gone.

    Background: cognee stores each ingested document as a row in the ``data``
    table with ``raw_data_location`` pointing at a ``text_<hash>.txt`` file under
    ``data_root_directory``. If that directory is ephemeral (the old default was
    inside the installed cognee package, wiped on every Docker rebuild) but the
    Postgres ``data`` table is on a persistent volume, the table ends up
    referencing files that no longer exist. The next ``cognify`` then raises
    ``File not found: .../text_<hash>.txt`` for every stale row.

    This scans every dataset's data rows and deletes the ones whose
    ``raw_data_location`` file is missing, so the dataset can be re-ingested
    cleanly. Best-effort and non-fatal: a failure here only means some stale
    rows linger until the cat 3 / file-not-found retry path clears them lazily.

    Implementation notes:
    - ``raw_data_location`` may carry a ``file://`` URI scheme (cognee 1.2.x) OR
      a bare path. ``_resolve_raw_data_location_path`` normalizes it before the
      existence check so healthy rows are never pruned by mistake.
    - We delete the row DIRECTLY via SQLAlchemy (``data`` + ``dataset_data``
      junction) rather than ``cognee.datasets.delete_data``, because the latter
      drags in the embedding/tokenizer machinery (``has_data_related_nodes`` ->
      tokenize) and fails with a tokenizer KeyError when the embedding pipeline
      is misconfigured. A stale row (missing backing file) by definition never
      completed ``cognify``, so it has NO graph nodes/edges -- there is nothing
      to clean up in the graph, only the relational rows.
    """
    if not _COGNEE_AVAILABLE:
        return
    try:
        datasets_obj = getattr(cognee, "datasets", None)
        if datasets_obj is None:
            return
        datasets = await datasets_obj.list_datasets()
        # Collect stale data_ids first, then delete in one session.
        stale_ids = []
        for d in datasets:
            list_data_fn = getattr(datasets_obj, "list_data", None)
            if not callable(list_data_fn):
                continue
            try:
                data_rows = await list_data_fn(d.id)
            except Exception as e:  # pragma: no cover - depends on cognee version
                logger.debug("reconcile: list_data(%s) failed: %s", d.id, e)
                continue
            for row in data_rows:
                loc = getattr(row, "raw_data_location", None)
                if not loc:
                    continue
                # Strip a ``file://`` URI scheme before the existence check so
                # healthy rows with the scheme-prefixed location are not pruned.
                path = _resolve_raw_data_location_path(loc)
                if os.path.exists(path):
                    continue  # backing file present -- healthy row
                stale_ids.append((d.id, row.id, loc))
        if not stale_ids:
            logger.debug("cognee stale-data reconciliation: no stale rows found.")
            return
        # Direct DB delete of stale ``data`` + ``dataset_data`` rows. Bypasses
        # cognee.datasets.delete_data (which needs the embedding pipeline).
        try:
            from cognee.infrastructure.databases.relational import get_relational_engine
            from cognee.modules.data.models import Data
            from cognee.modules.data.models.Dataset import DatasetData
            from sqlalchemy import delete
            eng = get_relational_engine()
            data_ids = [row_id for (_ds, row_id, _loc) in stale_ids]
            async with eng.get_async_session() as session:
                await session.execute(
                    delete(DatasetData).where(DatasetData.data_id.in_(data_ids))
                )
                await session.execute(
                    delete(Data).where(Data.id.in_(data_ids))
                )
                await session.commit()
        except Exception as e:  # pragma: no cover - depends on cognee version
            logger.warning(
                "cognee stale-data reconciliation: direct DB delete failed: %s. "
                "Stale rows will be cleared lazily by the add/retry path.", e,
            )
            return
        logger.info(
            "cognee stale-data reconciliation: pruned %d row(s) with missing "
            "backing files across %d dataset(s).",
            len(stale_ids), len(datasets),
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("cognee stale-data reconciliation failed (non-fatal): %s", e)


# --- Repo-path text extraction (cat 2: exclude .git + binary files) -------------
# cognee's ``text_loader.load`` does ``open(file_path, encoding=utf-8).read()``
# on every file it traverses, which raises UnicodeDecodeError on binary files
# like ``.git/index``. When the caller hands us a directory (the cloned repo),
# we read the text files ourselves (skipping .git/binary/non-text) and hand
# cognee the concatenated text blob instead of the raw path.
_COGNEE_TEXT_EXTENSIONS = {
    # code
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".cs", ".scala", ".clj", ".ex",
    ".exs", ".erl", ".hs", ".ml", ".fs", ".lua", ".pl", ".r", ".dart", ".vue",
    ".svelte", ".m", ".mm",
    # docs / config (text)
    ".md", ".markdown", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".properties", ".xml", ".html", ".htm", ".css", ".scss",
    ".sass", ".less", ".csv", ".tsv", ".env.example", ".gitignore", ".dockerignore",
    ".editorconfig", ".sql", ".graphql", ".gql", ".proto", ".thrift", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".makefile", ".mk",
}
# Basenames without an extension that are worth indexing.
_COGNEE_TEXT_BASENAMES = {
    "readme", "readme.md", "readme.rst", "readme.txt", "readme",
    "license", "license.md", "contributing", "contributing.md",
    "changelog", "changelog.md", "makefile", "dockerfile", "rakefile",
    "gemfile", "procfile", "vagrantfile", "jenkinsfile", "brewfile",
}
# Directory names to always skip when walking a repo for cognee indexing.
# ``.git`` is the primary culprit (its ``index``/``objects`` are binary and
# raise UnicodeDecodeError in cognee's text loader).
_COGNEE_SKIP_DIRS = {
    ".git", ".svn", ".hg", ".bzr", "node_modules", "bower_components",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".venv", "venv", "env", "virtualenv", "dist", "build", "out", "target",
    "bin", "obj", ".idea", ".vscode", ".vs", ".next", ".nuxt", ".cache",
    ".adalflow", "logs", "log", "tmp", "temp", "coverage", ".coverage",
}
# Per-file read cap so a giant minified file doesn't dominate the blob.
_COGNEE_PER_FILE_MAX_CHARS = 16_000
# Cap the whole blob so cognee ingestion stays bounded.
_COGNEE_BLOB_MAX_CHARS = 200_000


def _is_likely_text_file(file_path: str) -> bool:
    """Quick extension/basename allow-list check for text files."""
    import os as _os
    name = _os.path.basename(file_path)
    low = name.lower()
    if low in _COGNEE_TEXT_BASENAMES:
        return True
    ext = _os.path.splitext(low)[1]
    return ext in _COGNEE_TEXT_EXTENSIONS


def _read_repo_text_for_cognee(repo_dir: str) -> str:
    """Walk ``repo_dir`` and return a concatenated text blob of text files.

    Skips ``.git`` and other non-source dirs (``_COGNEE_SKIP_DIRS``) and any
    file whose extension isn't in the text allow-list. Each file is read as
    UTF-8 (errors replaced) and capped per-file + per-blob. Returns "" if the
    path isn't a directory or no text files were found.
    """
    import os as _os
    if not repo_dir or not _os.path.isdir(repo_dir):
        return ""
    parts = []
    total = 0
    for root, dirs, files in _os.walk(repo_dir):
        # Mutate dirs in place to prune skipped directories (os.walk contract).
        dirs[:] = [d for d in dirs if d not in _COGNEE_SKIP_DIRS]
        for fname in files:
            fpath = _os.path.join(root, fname)
            if not _is_likely_text_file(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(_COGNEE_PER_FILE_MAX_CHARS + 1)
                if len(text) > _COGNEE_PER_FILE_MAX_CHARS:
                    text = text[:_COGNEE_PER_FILE_MAX_CHARS] + "\n... (file truncated)\n"
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("cognee repo read skipped %s: %s", fpath, e)
                continue
            if not text:
                continue
            rel = _os.path.relpath(fpath, repo_dir)
            block = f"### File: {rel}\n``\n{text}\n```\n"
            if total + len(block) > _COGNEE_BLOB_MAX_CHARS:
                remaining = _COGNEE_BLOB_MAX_CHARS - total
                if remaining > 200:
                    parts.append(block[:remaining] + "\n... (repo blob truncated)\n")
                parts = [p for p in parts if p]  # keep what we have
                logger.info(
                    "cognee repo blob capped at %d chars while reading %s.",
                    _COGNEE_BLOB_MAX_CHARS, repo_dir,
                )
                return "\n".join(parts)
            parts.append(block)
            total += len(block)
    return "\n".join(parts)


async def add_and_index_document(content_or_path: str, dataset_name: str):
    """
    Adds document text or file path into Cognee memory and processes it to
    knowledge graph. NEVER raises: errors are logged so callers (e.g.
    background docgen) are not crashed by cognee import/timeout issues.

    Two robustness fixes over the naive ``cognee.add(content_or_path)``:
    - **Directory input** (cat 2): when ``content_or_path`` is a directory
      (the cloned repo), cognee's text loader would traverse ``.git/index``
      and raise ``UnicodeDecodeError`` on the binary file. Instead we read the
      text files ourselves (allow-listed extensions, skip ``.git``/binary) and
      hand cognee a concatenated text blob.
    - **Re-index duplicate key** (cat 3): re-docgen re-adds the same content
      to an existing dataset, which raises ``UniqueViolationError`` on cognee
      1.2.x's ``data`` table (fixed content hash -> fixed PK). On that error
      we delete the dataset's data and retry the add once.
    """
    if not _COGNEE_AVAILABLE:
        logger.warning("cognee unavailable; skipping index of dataset %r.", dataset_name)
        return
    apply_cognee_runtime_config()

    # Cat 2: if the input is a directory (cloned repo), read text files
    # ourselves and pass a blob to cognee. This avoids cognee's text loader
    # choking on binary files like ``.git/index``.
    payload = content_or_path
    import os as _os
    if content_or_path and _os.path.isdir(content_or_path):
        blob = _read_repo_text_for_cognee(content_or_path)
        if not blob:
            logger.warning(
                "cognee index: no text files found in %r; skipping dataset %r.",
                content_or_path, dataset_name,
            )
            return
        payload = blob
        logger.info(
            "cognee index: read %d chars of text from repo %r for dataset %r.",
            len(blob), content_or_path, dataset_name,
        )

    try:
        logger.info(f"Ingesting into Cognee (dataset: {dataset_name})...")
        await cognee.add(payload, dataset_name=dataset_name)
        await cognee.cognify(datasets=[dataset_name])
        logger.info(f"Cognee: Ingested and cognified dataset '{dataset_name}' successfully.")
    except Exception as e:
        # Cat 3 + ephemeral-root recovery: two recoverable failure classes share
        # the same clear-and-retry path:
        #  - **Duplicate key** (cat 3): re-indexing the same content raises a
        #    duplicate-key violation on cognee 1.2.x's ``data`` table (PK is a
        #    content hash). Clearing the dataset lets the retry re-ingest.
        #  - **File not found** (stale backing file): if a ``Data`` row's
        #    ``raw_data_location`` points at a ``text_<hash>.txt`` that no longer
        #    exists (old ephemeral data root wiped on rebuild), ``cognify`` raises
        #    ``File not found: .../text_<hash>.txt`` / ``FileNotFoundError``.
        #    Clearing the stale dataset data lets the retry re-create the file.
        # Any OTHER error is logged and swallowed (never fatal to docgen).
        msg = str(e)
        is_dup_key = (
            "duplicate key value" in msg
            or "data_pkey" in msg
            or "UniqueViolationError" in type(e).__name__
            or "IntegrityError" in type(e).__name__
        )
        is_file_not_found = (
            "File not found" in msg
            or "FileNotFoundError" in type(e).__name__
            or isinstance(e, FileNotFoundError)
        )
        if not (is_dup_key or is_file_not_found):
            logger.error(
                f"Error ingesting content into Cognee for dataset '{dataset_name}': {e}",
                exc_info=True,
            )
            return
        reason = (
            "duplicate-key violation (re-index of existing content)"
            if is_dup_key and not is_file_not_found
            else "stale backing file (File not found)"
            if is_file_not_found and not is_dup_key
            else "duplicate-key + stale backing file"
        )
        logger.warning(
            "cognee add for dataset %r hit a %s; clearing dataset data and "
            "retrying once. Error: %s",
            dataset_name, reason, msg,
        )
        # Clear the dataset's data + graph so the retry can re-ingest cleanly.
        # Uses the CURRENT cognee API (``datasets.empty_dataset``); the legacy
        # ``cognee.delete([name])`` is deprecated and its real signature is
        # ``delete(data_id, dataset_id)`` so passing a list silently fails.
        cleared = await _empty_cognee_dataset(dataset_name)
        if not cleared:
            logger.warning(
                "could not clear dataset %r before retry; aborting re-index.",
                dataset_name,
            )
            return
        try:
            await cognee.add(payload, dataset_name=dataset_name)
            await cognee.cognify(datasets=[dataset_name])
            logger.info(
                "Cognee: re-ingested dataset '%s' successfully after %s.",
                dataset_name, reason,
            )
        except Exception as e2:
            logger.error(
                f"cognee re-index retry failed for dataset '{dataset_name}': {e2}",
                exc_info=True,
            )

async def query_cognee(query: str, dataset_name: str, top_k: int = 20) -> str:
    """
    Queries Cognee knowledge graph and returns retrieved contextual
    triplets/text. NEVER raises; returns "" on any error.
    """
    if not _COGNEE_AVAILABLE:
        return ""
    apply_cognee_runtime_config()
    try:
        from cognee import SearchType
        logger.info(f"Querying Cognee knowledge graph (dataset: {dataset_name})...")
        results = await cognee.recall(
            query_text=query,
            query_type=SearchType.GRAPH_COMPLETION,
            top_k=top_k
        )
        if results:
            # Format results nicely
            return "\n\n".join([str(r) for r in results])
        return ""
    except Exception as e:
        logger.error(f"Error querying Cognee: {e}", exc_info=True)
        return ""
