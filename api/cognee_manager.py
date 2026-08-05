from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

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
        from api.settings_store import get_setting
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
_local_llm_host = _local_llm_host.removesuffix("/v1")

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

class CogneeRateLimiter:
    """Async Semaphore, Rate Limiter, and 429 Retry Handler for Cognee calls."""

    def __init__(self):
        # Map: loop_id -> (Semaphore, Lock, max_concurrency)
        self._loop_primitives: Dict[int, Tuple[asyncio.Semaphore, asyncio.Lock, int]] = {}
        self.last_call_time: float = 0.0

    def get_rate_settings(self) -> Tuple[int, float]:
        """Read rate limit settings from DB settings store or environment.

        - cognee.max_concurrency: int (default 2)
        - cognee.delay_seconds: float (default 0.5s -> max 2 requests/sec)
        - cognee.rate_limit_rps: float (e.g. 2.0 -> 0.5s delay)
        """
        max_conc = 2
        delay_sec = 0.5
        try:
            from api.settings_store import get_setting
            mc = get_setting("cognee.max_concurrency") or os.environ.get("COGNEE_MAX_CONCURRENCY")
            if mc:
                try:
                    max_conc = max(1, int(str(mc).strip()))
                except ValueError:
                    pass
            ds = get_setting("cognee.delay_seconds") or get_setting("cognee.rate_limit_delay") or os.environ.get("COGNEE_DELAY_SECONDS")
            if ds:
                try:
                    delay_sec = max(0.0, float(str(ds).strip()))
                except ValueError:
                    pass
            rps = get_setting("cognee.rate_limit_rps") or os.environ.get("COGNEE_RATE_LIMIT_RPS")
            if rps:
                try:
                    val = float(str(rps).strip())
                    if val > 0:
                        delay_sec = max(delay_sec, 1.0 / val)
                except ValueError:
                    pass
        except Exception:
            pass
        return max_conc, delay_sec

    def _get_loop_primitives(self, max_concurrency: int) -> Tuple[Optional[asyncio.Semaphore], Optional[asyncio.Lock]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None, None

        loop_id = id(loop)
        if loop_id in self._loop_primitives:
            sem, lock, cached_max = self._loop_primitives[loop_id]
            if cached_max == max_concurrency:
                return sem, lock

        sem = asyncio.Semaphore(max_concurrency)
        lock = asyncio.Lock()
        self._loop_primitives[loop_id] = (sem, lock, max_concurrency)
        return sem, lock

    async def execute(self, func, *args, **kwargs):
        max_conc, delay_sec = self.get_rate_settings()
        sem, lock = self._get_loop_primitives(max_conc)

        async def _run():
            if lock and delay_sec > 0:
                async with lock:
                    now = asyncio.get_running_loop().time()
                    elapsed = now - self.last_call_time
                    if elapsed < delay_sec:
                        await asyncio.sleep(delay_sec - elapsed)
                    self.last_call_time = asyncio.get_running_loop().time()

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg) and attempt < max_retries - 1:
                        backoff = (attempt + 1) * 2.5
                        logger.warning(
                            "Cognee LLM/Embedding call hit rate limit (attempt %d/%d). Sleeping %.1fs: %s",
                            attempt + 1, max_retries, backoff, e,
                        )
                        await asyncio.sleep(backoff)
                    else:
                        raise

        if sem:
            async with sem:
                return await _run()
        else:
            return await _run()


_cognee_rate_limiter = CogneeRateLimiter()


def _apply_cognee_rate_limit_patches():
    """Apply rate limiter monkeypatches to litellm and openai clients for cognee."""
    try:
        import litellm
        if not getattr(litellm, "_productarium_rate_limited", False):
            orig_acompletion = getattr(litellm, "acompletion", None)
            orig_aembedding = getattr(litellm, "aembedding", None)

            if callable(orig_acompletion):
                async def _patched_acompletion(*args, **kwargs):
                    return await _cognee_rate_limiter.execute(orig_acompletion, *args, **kwargs)
                setattr(litellm, "acompletion", _patched_acompletion)

            if callable(orig_aembedding):
                async def _patched_aembedding(*args, **kwargs):
                    return await _cognee_rate_limiter.execute(orig_aembedding, *args, **kwargs)
                setattr(litellm, "aembedding", _patched_aembedding)

            setattr(litellm, "_productarium_rate_limited", True)
            logger.info("Cognee litellm rate-limiter & 429 retry patch applied.")
    except Exception as e:
        logger.debug("Could not patch litellm for cognee rate limiting: %s", e)

    try:
        import openai
        if hasattr(openai, "resources") and hasattr(openai.resources.chat, "AsyncCompletions"):
            ac_cls = openai.resources.chat.AsyncCompletions
            if not getattr(ac_cls, "_productarium_rate_limited", False):
                orig_ac_create = ac_cls.create
                async def _patched_ac_create(self_obj, *args, **kwargs):
                    return await _cognee_rate_limiter.execute(orig_ac_create, self_obj, *args, **kwargs)
                ac_cls.create = _patched_ac_create
                ac_cls._productarium_rate_limited = True

        if hasattr(openai, "resources") and hasattr(openai.resources.embeddings, "AsyncEmbeddings"):
            ae_cls = openai.resources.embeddings.AsyncEmbeddings
            if not getattr(ae_cls, "_productarium_rate_limited", False):
                orig_ae_create = ae_cls.create
                async def _patched_ae_create(self_obj, *args, **kwargs):
                    return await _cognee_rate_limiter.execute(orig_ae_create, self_obj, *args, **kwargs)
                ae_cls.create = _patched_ae_create
                ae_cls._productarium_rate_limited = True
    except Exception as e:
        logger.debug("Could not patch openai AsyncCompletions for cognee rate limiting: %s", e)
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
            # Clamp text_input conservatively for graph extraction (max 400 tokens / ~1500 chars)
            try:
                from api.model_utils import clamp_text_by_tokens
                text_input = clamp_text_by_tokens(text_input, max_tokens=400)
            except Exception:
                if len(text_input) > 1500:
                    text_input = text_input[:1500] + "\n... (truncated)"

            try:
                # 30-second timeout so cognify graph extraction never hangs on a slow LLM
                return await asyncio.wait_for(
                    _ORIG_ACREATE_STRUCTURED_OUTPUT(self, text_input, system_prompt, response_model, **kwargs),
                    timeout=30.0,
                )
            except Exception as exc:
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
            emb_dims = _resolve_embedding_dimensions(emb_model)
            emb_tok = os.environ.get("HUGGINGFACE_TOKENIZER", "")
            if not emb_model:
                emb_model = "nomic-embed-text"
        else:
            # openai_local / openai -> use the openai-compatible engine, which
            # uses the openai SDK directly (avoids litellm embedding quirks).
            emb_provider = "openai_compatible"
            emb_endpoint = _host_to_v1(emb_base) if emb_base else f"{_local_llm_host}/v1"
            if not emb_model:
                emb_model = os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"
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
    except Exception as e:
        # On a fresh DB the schema may not have been created yet when this runs
        # (cognee.init()/setup() can defer table creation to the first write).
        # cognee raises ``DatabaseNotCreatedError: The database has not been
        # created yet. Please call `await setup()` first`` in that case. There
        # is nothing stale to reconcile on a fresh DB, so log at debug (not
        # error) and bail out cleanly instead of spamming the startup log.
        msg = str(e)
        if (
            "DatabaseNotCreatedError" in type(e).__name__
            or "has not been created" in msg
            or "call `await setup()`" in msg
            or "DatabaseNotCreatedError" in msg
        ):
            logger.debug(
                "cognee stale-data reconciliation skipped: relational schema not "
                "created yet (nothing stale on a fresh DB). Detail: %s", e,
            )
            return
        raise
    try:
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


def _looks_like_file_path(payload: str) -> bool:
    """True if ``payload`` is an existing file cognee should load from disk.

    ``add_and_index_document`` accepts BOTH genuine file/dir paths (which
    cognee's file loader reads from disk) AND raw text blobs (the repo text we
    read ourselves, or markdown from a caller). The two need different
    treatment when wrapping in ``DataItem``: a file path must stay a string so
    cognee opens the file, while a text blob must be wrapped in ``DataItem``
    (see cat 4 in ``add_and_index_document``) or the pipeline crashes with
    ``'str' object has no attribute '__dict__'``.

    We treat a payload as a file path only when it is a short-ish string that
    resolves to an EXISTING file on disk. Multi-KB text blobs (even if they
    happen to contain a newline-free path-like substring) never satisfy this.
    """
    import os as _os
    if not payload or len(payload) > 4096 or "\n" in payload:
        return False
    try:
        return _os.path.isfile(payload)
    except (OSError, ValueError):
        return False


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


async def add_document(content_or_path: str, dataset_name: str) -> bool:
    """Add a document text or directory path to a cognee dataset (WITHOUT running cognify).

    Safe and non-fatal. Returns True if the payload was added successfully.
    """
    if not _COGNEE_AVAILABLE or not content_or_path:
        return False
    apply_cognee_runtime_config()

    payload = content_or_path
    import os as _os
    if content_or_path and _os.path.isdir(content_or_path):
        blob = _read_repo_text_for_cognee(content_or_path)
        if not blob:
            logger.warning("cognee index: no text files found in %r; skipping dataset %r.", content_or_path, dataset_name)
            return False
        payload = blob
        logger.info("cognee index: read %d chars of text from repo %r for dataset %r.", len(blob), content_or_path, dataset_name)

    ingest_payload = payload
    if isinstance(payload, str) and not _looks_like_file_path(payload):
        try:
            from cognee.tasks.ingestion.data_item import DataItem
            ingest_payload = DataItem(data=payload)
        except Exception:
            pass

    try:
        await cognee.add(ingest_payload, dataset_name=dataset_name)
        return True
    except Exception as e:
        msg = str(e)
        if "duplicate key value" in msg or "data_pkey" in msg or "UniqueViolationError" in type(e).__name__:
            cleared = await _empty_cognee_dataset(dataset_name)
            if cleared:
                try:
                    await cognee.add(ingest_payload, dataset_name=dataset_name)
                    return True
                except Exception:
                    pass
        logger.warning("cognee add_document failed for dataset %r: %s", dataset_name, e)
        return False


async def cognify_dataset(dataset_name: str) -> bool:
    """Run cognee.cognify() ONCE over an entire dataset to build knowledge graph."""
    if not _COGNEE_AVAILABLE:
        return False
    apply_cognee_runtime_config()

    try:
        from api.model_utils import get_model_context_window
        ctx_win = get_model_context_window(task="cognee")
    except Exception:
        ctx_win = 8192
    safe_chunk_size = max(300, min(1200, (ctx_win - 3000) // 2))

    try:
        logger.info("Cognifying Cognee dataset %r (chunk_size: %d)...", dataset_name, safe_chunk_size)
        await cognee.cognify(datasets=[dataset_name], chunk_size=safe_chunk_size)
        logger.info("Cognee: Ingested and cognified dataset %r successfully.", dataset_name)
        return True
    except Exception as e:
        logger.error("Error cognifying dataset %r: %s", dataset_name, e, exc_info=True)
        return False


async def add_documents_and_cognify_once(items: List[str], dataset_name: str) -> None:
    """Add multiple documents/paths to a dataset and run cognify ONCE at the end."""
    if not _COGNEE_AVAILABLE or not items:
        return
    added_any = False
    for item in items:
        if item and item.strip():
            ok = await add_document(item, dataset_name)
            if ok:
                added_any = True
    if added_any:
        await cognify_dataset(dataset_name)


async def add_and_index_document(content_or_path: str, dataset_name: str) -> None:
    """Convenience function: add single document and run cognify once."""
    ok = await add_document(content_or_path, dataset_name)
    if ok:
        await cognify_dataset(dataset_name)

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


async def reindex_product_knowledge_graph(product_id: Optional[str] = None) -> Dict[str, Any]:
    """Force re-indexing of product artifacts and knowledge nodes into Cognee."""
    if not _COGNEE_AVAILABLE:
        return {"success": False, "message": "Cognee package is not available.", "reindexed_count": 0}

    apply_cognee_runtime_config()

    try:
        from api.db import SessionLocal
        from api.models import ProductORM
        from sqlalchemy.orm import selectinload

        with SessionLocal() as db:
            if product_id:
                products = db.query(ProductORM).options(
                    selectinload(ProductORM.artifacts),
                    selectinload(ProductORM.knowledge_nodes),
                ).filter(ProductORM.id == product_id).all()
            else:
                products = db.query(ProductORM).options(
                    selectinload(ProductORM.artifacts),
                    selectinload(ProductORM.knowledge_nodes),
                ).all()

        if not products:
            return {"success": True, "message": "No products found to reindex.", "reindexed_count": 0}

        reindexed_count = 0
        for p in products:
            dataset_name = f"prod_{p.id}"
            items = []
            for a in p.artifacts:
                docs = getattr(a, "generated_docs", None) or getattr(a, "content", None) or ""
                if docs and docs.strip():
                    items.append(docs.strip())
            for n in p.knowledge_nodes:
                md = getattr(n, "content", None) or getattr(n, "content_md", None) or ""
                if md and md.strip():
                    items.append(md.strip())

            if items:
                logger.info("Force re-indexing %d items for product %s (%s)...", len(items), p.name, dataset_name)
                await _empty_cognee_dataset(dataset_name)
                await add_documents_and_cognify_once(items, dataset_name)
                reindexed_count += 1

        return {
            "success": True,
            "message": f"Successfully reindexed {reindexed_count} product(s) into Cognee knowledge graph.",
            "reindexed_count": reindexed_count,
        }
    except Exception as e:
        logger.error("Error during force cognee reindex: %s", e, exc_info=True)
        return {"success": False, "message": f"Reindex error: {e}", "reindexed_count": 0}
