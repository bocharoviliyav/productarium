from __future__ import annotations

import logging
import os
import warnings

logger = logging.getLogger(__name__)


# Apply SSL/TLS configuration (corporate CA bundle / skip-verify) before cognee
# is imported. cognee's aiohttp adapters build an ssl.SSLContext via
# ``create_secure_ssl_context`` -> ``ssl.create_default_context()``, which honors
# the ``SSL_CERT_FILE`` env var, so setting it here (from the admin panel / env)
# makes cognee trust a corporate root cert. See api/ssl_config.py.
from api.config.ssl import apply_ssl_env, apply_cognee_ssl_patch
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
#   provider "openai"/
#   "openai_local"/
#   "custom"             -> "openai/<model>"   (litellm openai route with a
#                                                    custom api_base)
#
# ``_strip_provider_prefix`` first removes any existing ``<provider>/`` segment
# so repeated applies never double-prefix (``openai/openai/qwen/qwen3.6-27b``).
# litellm's known routing prefixes are an allow-list; a stray slash inside a
# model name (rare) is left intact by only stripping a *leading* known prefix.
_LITELLM_PROVIDER_PREFIXES = (
    "openai/",
    "anthropic/",
    "gemini/",
    "mistral/",
    "azure/",
    "bedrock/",
    "huggingface/",
    "together_ai/",
)


def _strip_provider_prefix(model: str) -> str:
    """Remove a leading litellm provider prefix (``openai/``, ``anthropic/``, …)."""
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

    - anything that ends up on the OpenAI/Custom route (openai /
      openai_local / custom / unknown) -> ``openai/<model>``

    Idempotent: an already-prefixed name is not double-prefixed.
    """
    bare = _strip_provider_prefix(model or "")
    if not bare:
        return model or ""
    # openai / openai_local / custom / unknown -> route through litellm's
    # openai-compatible path with a custom api_base.
    return f"openai/{bare}"


def _host_to_v1(host: str) -> str:
    """Normalize a local or remote LLM/embedder host URL to an OpenAI ``/v1`` base URL.

    Strips trailing slashes, `/embeddings`, and `/v1` so that `f"{h}/v1"` resolves cleanly
    without duplicating paths (e.g. `https://ai.gw/v1/embeddings` -> `https://ai.gw/v1`).
    """
    if not host:
        return ""
    h = host.strip().rstrip("/")
    if h.lower().endswith("/embeddings"):
        h = h[:-11].rstrip("/")
    if h.lower().endswith("/v1"):
        h = h[:-3].rstrip("/")
    return f"{h}/v1" if h else h


def _resolve_embedding_dimensions(model_name: str) -> int:
    """Infer embedding dimensions from model name or admin settings/env."""
    try:
        from api.config.settings import get_setting
        stored = get_setting("models.embedder.dimensions")
        if stored:
            return int(stored.strip())
    except Exception:
        pass
    env_dim = os.environ.get("EMBEDDING_DIMENSIONS") or os.environ.get("DEEPWIKI_EMBEDDING_DIMENSIONS")
    if env_dim:
        try:
            return int(env_dim.strip())
        except ValueError:
            pass

    if not model_name:
        return 768
    m = model_name.lower()
    if "3-large" in m or "3072" in m:
        return 3072
    if "3-small" in m or "ada-002" in m or "1536" in m:
        return 1536
    if "qwen" in m or "bge-m3" in m or "1024" in m:
        return 1024
    if "bge-small" in m or "minilm" in m or "384" in m:
        return 384
    return 768
def _resolve_default_provider() -> str:
    """Default cognee LLM provider.

    Every supported local server (LM Studio, llama.cpp, vLLM, ...)
    exposes an OpenAI-compatible ``/v1`` API, so cognee's ``openai`` provider
    (which POSTs to {LLM_ENDPOINT}/chat/completions via the OpenAI SDK) covers
    all cases.
    """
    return "openai"


# --- Local LLM configuration for cognee ---
# cognee talks to an OpenAI-compatible LLM endpoint. We point it at the SAME
# local server the rest of DeepWiki uses (LM Studio by default, but also
# llama.cpp / vLLM via LOCAL_OPENAI_BASE_URL). cognee's LLM endpoint
# must use the OpenAI-compatible path ("/v1"). A non-empty LLM_API_KEY is
# required even by local servers that ignore it; we default to "not-needed".
_local_llm_host = (
    os.environ.get("LOCAL_OPENAI_BASE_URL")
    or "http://localhost:1234/v1"
).rstrip("/")
# Normalize to a bare host (strip a trailing "/v1") so we can append the right
# path per cognee's expectations below.
_local_llm_host = _local_llm_host.removesuffix("/v1")

# The default cognee provider must MATCH the local server we point at, so that
# the right adapter (OpenAIAdapter) is selected and the model name is prefixed
# correctly for litellm. With an empty .env this resolves to ``openai`` (the
# fully-local default).
_default_cognee_provider = _resolve_default_provider()
os.environ.setdefault("LLM_PROVIDER", _default_cognee_provider)
os.environ.setdefault("LLM_ENDPOINT", f"{_local_llm_host}/v1")
# ``LLM_MODEL`` must carry the litellm provider prefix on the openai route.
# Default bare model is the same RLM/local default, then normalized.
_default_cognee_model = os.environ.get("RLM_MODEL_NAME") or os.environ.get("LLM_MODEL") or "qwen/qwen3.6-27b"
os.environ.setdefault("LLM_MODEL", _normalize_model_for_litellm(_default_cognee_provider, _default_cognee_model))
if not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = os.environ.get("LOCAL_OPENAI_API_KEY") or "not-needed"

# --- Local embeddings configuration for cognee ---
# cognee requires embeddings to be configured alongside the LLM; if only one is
# set it silently falls back to OpenAI (cloud). Every supported local server
# (LM Studio, llama.cpp, vLLM, ...) exposes the OpenAI-compatible
# embeddings shape (POST {endpoint}/embeddings), so the single
# ``openai_compatible`` provider covers all cases.
#
# CRITICAL: use ``openai_compatible`` (NOT ``openai``). cognee's embedding
# dispatch in ``create_embedding_engine`` routes ONLY ``openai_compatible`` to
# safe engines; ANY other value (incl. ``openai``) falls through
# to ``LiteLLMEmbeddingEngine``, whose ``__init__`` calls
# ``tiktoken.encoding_for_model(embedding_model)``. For a nomic-embed model
# name tiktoken has no mapping -> ``KeyError: Could not automatically map
# text-embedding-nomic-embed-text-v1.5 to a tokeniser``, which breaks every
# cognee.add()/cognify(). ``OpenAICompatibleEmbeddingEngine`` instead uses
# the OpenAI SDK directly and falls back to ``cl100k_base`` when the
# HuggingFace tokenizer (``transformers``) isn't installed.
# cognee additionally requires EMBEDDING_DIMENSIONS and HUGGINGFACE_TOKENIZER,
# and its config validator raises if ANY of {EMBEDDING_PROVIDER, EMBEDDING_MODEL,
# EMBEDDING_DIMENSIONS, HUGGINGFACE_TOKENIZER} is set without the others, so we
# set them all together here (each via setdefault so explicit env/ compose wins).
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
# Set the graph DB credentials EXPLICITLY. cognee's GraphEngine warns
# ("Postgres graph credentials are not fully configured; falling back to the
# relational database configuration") whenever GRAPH_DATABASE_HOST/PORT/NAME/
# USERNAME/PASSWORD are all unset, even though it then falls back to DB_*.
# Mirroring DB_* here silences that warning and makes the graph backend use
# the SAME Postgres instance intentionally (relational + vector + graph in
# one DB). All via setdefault so explicit env / compose wins.
os.environ.setdefault("GRAPH_DATABASE_HOST", DB_HOST)
os.environ.setdefault("GRAPH_DATABASE_PORT", DB_PORT)
os.environ.setdefault("GRAPH_DATABASE_NAME", DB_NAME)
os.environ.setdefault("GRAPH_DATABASE_USERNAME", DB_USERNAME)
os.environ.setdefault("GRAPH_DATABASE_PASSWORD", DB_PASSWORD)
# Set the vector DB credentials EXPLICITLY. cognee's VectorEngine warns
# ("PGVector credentials are not fully configured; falling back to the
# relational database configuration") whenever VECTOR_DB_HOST/PORT/NAME/
# USERNAME/PASSWORD are all unset, even though it then falls back to DB_*.
# Mirroring DB_* here silences that warning and makes the pgvector backend
# use the SAME Postgres instance intentionally (relational + vector + graph
# in one DB). All via setdefault so explicit env / compose wins.
os.environ.setdefault("VECTOR_DB_HOST", DB_HOST)
os.environ.setdefault("VECTOR_DB_PORT", DB_PORT)
os.environ.setdefault("VECTOR_DB_NAME", DB_NAME)
os.environ.setdefault("VECTOR_DB_USERNAME", DB_USERNAME)
os.environ.setdefault("VECTOR_DB_PASSWORD", DB_PASSWORD)
# --- Fix #2: asyncpg cross-loop termination --------------------------------
# cognee's relational, graph, and vector Postgres engines use SQLAlchemy's
# AsyncAdaptedQueuePool with asyncpg. A pooled asyncpg connection is bound to
# the event loop that first checked it out; when the pool later tries to
# terminate/recycle that connection from a DIFFERENT loop (which happens here
# because cognee's own background tasks, the FastAPI request loop, and
# asyncio.to_thread worker threads all touch the shared pool), asyncpg raises
# ``RuntimeError: ... got Future ... attached to a different loop`` during
# ``do_terminate`` -> ``_terminate_graceful_close``.
#
# The robust fix is to use NullPool so no asyncpg connections are retained in
# a pool across tasks/loops. Each DB session opens a fresh connection and closes
# it on the current loop.
#
# To support string `"nullpool"` across ALL cognee adapters (relational, graph,
# and vector stores), we monkeypatch `sqlalchemy.ext.asyncio.create_async_engine`
# so that any string `"nullpool"` in `kwargs["poolclass"]` is converted to the
# `sqlalchemy.pool.NullPool` class before SQLAlchemy inspects it.
try:
    import sqlalchemy.ext.asyncio as _sa_asyncio
    from sqlalchemy.pool import NullPool as _NullPool

    _orig_create_async_engine = _sa_asyncio.create_async_engine

    def _patched_create_async_engine(*args, **kwargs):
        pc = kwargs.get("poolclass")
        if isinstance(pc, str) and pc.strip().lower() == "nullpool":
            kwargs["poolclass"] = _NullPool
        return _orig_create_async_engine(*args, **kwargs)

    _sa_asyncio.create_async_engine = _patched_create_async_engine
except Exception as _patch_err:  # pragma: no cover - defensive
    logger.warning("Could not patch create_async_engine for NullPool: %s", _patch_err)

os.environ.setdefault(
    "POOL_ARGS",
    '{"poolclass": "nullpool", "pool_pre_ping": true}',
)
os.environ.setdefault(
    "DATABASE_POOL_ARGS",
    '{"poolclass": "nullpool", "pool_pre_ping": true}',
)

# --- Skip cognee's blocking LLM connection test (item 10 bug fix) -------------
# cognee runs a blocking LLM connection test during cognify() that times out
# after ~30s when pointed at a local model that is slow to answer. Setting
# COGNEE_SKIP_CONNECTION_TEST=true (the default here) disables that test so
# cognify() starts immediately. Overridable via env. Also best-effort: shrink
# the cognee LLM connection timeout if cognee honors it.
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
# Best-effort: shrink the cognee LLM connection timeout if cognee honors it.
# Resolved through the central timeout config (admin > env > default).
try:
    from api.config.timeout import resolve_cognee_llm_connection_timeout
    os.environ.setdefault("COGNEE_LLM_CONNECTION_TIMEOUT", str(int(resolve_cognee_llm_connection_timeout())))
except Exception:  # pragma: no cover - defensive
    os.environ.setdefault("COGNEE_LLM_CONNECTION_TIMEOUT", "10")
logger.info(
    "cognee connection test %s (COGNEE_SKIP_CONNECTION_TEST=%s).",
    "skipped" if os.environ.get("COGNEE_SKIP_CONNECTION_TEST", "true").lower()
    in ("1", "true", "t", "yes") else "enabled",
    os.environ.get("COGNEE_SKIP_CONNECTION_TEST"),
)

# --- Per-call timeout for cognee graph extraction (structured-output LLM) -------
# cognee's ``cognify`` extracts a knowledge graph by calling the LLM with a
# Pydantic ``response_model`` via ``instructor`` (one call per text chunk).
# A slow local/corporate LLM can take several minutes for a single structured
# extraction (markdown_json_mode: the model emits a full section's worth of
# tokens before the JSON block is parsed out). The patched wrapper below clamps
# each call to this timeout and returns an empty model on timeout so cognify
# keeps making progress instead of stalling. Resolved through the central
# timeout config (admin > env > default). Default 1800s (30 min), floor 60s.
# Must be >= the per-request HTTP timeout (llm_request) so a genuinely
# slow-but-progressing call is not killed before it can return.
def _resolve_graph_extraction_timeout() -> float:
    from api.config.timeout import resolve_cognee_graph_extraction_timeout
    return resolve_cognee_graph_extraction_timeout()


# --- Overall timeout for a full ``cognify()`` run -----------------------------
# A cognify pass over a large product dataset can run for hours on a local
# model (many chunks, each a structured-output LLM call). cognee itself has no
# top-level timeout, so a hung chunk / dead LLM connection could stall the
# background indexer forever. Wrap the whole run in ``asyncio.wait_for`` with
# this ceiling. Resolved through the central timeout config (admin > env >
# default). Default 7200s (2h) to accommodate very large repos, floor 300s.
def _resolve_cognify_timeout() -> float:
    from api.config.timeout import resolve_cognee_cognify_timeout
    return resolve_cognee_cognify_timeout()


# --- Suppress the benign "coroutine was never awaited" RuntimeWarning ----------
# cognee's structured-output path (OpenAIAdapter.acreate_structured_output ->
# litellm.acompletion / instructor) can, on a timeout or a structured-output
# validation failure, create a litellm ``acompletion`` coroutine that is never
# awaited before being replaced by the retry. Python's runtime then emits:
#   RuntimeWarning: coroutine 'OpenAIChatCompletion.acompletion' was never awaited
# This is internal to cognee/litellm and harmless here: the patched wrapper
# below already returns an empty ``response_model()`` on failure, so the dropped
# coroutine has no observable effect. Filter just this warning so the logs are
# not flooded during a long cognify run. Applied once at import.
try:
    warnings.filterwarnings(
        "ignore",
        message=r"coroutine '.*acompletion' was never awaited",
        category=RuntimeWarning,
    )
except Exception:  # pragma: no cover - defensive
    pass

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
