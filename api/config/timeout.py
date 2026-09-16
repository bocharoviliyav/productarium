"""Central timeout configuration for Productarium.

Single source of truth for every timeout parameter in the application. Every
module that needs a timeout reads it through :func:`resolve_timeout` /
:func:`resolve_timeout_int` (or one of the thin convenience wrappers exposed
by name), instead of hardcoding a literal or parsing an env var inline.

Precedence (highest -> lowest):

1. Admin settings store (``timeouts.<key>`` SettingORM row) -- set from the
   admin "Timeouts" panel; takes effect on the next :func:`resolve_*` call
   without a restart (resolvers are read-through; no caching). Exported to the
   canonical env var by :func:`sync_timeout_env` so module-level / subprocess
   readers also see it.
2. Environment variable (the ``env_var`` for the key) -- the fallback when the
   admin store is unset or the DB is down. Also documented in ``.env.example``.
3. Default value -- a sensible per-key constant, raised so long-running work
   on large repos (multi-hour docgen runs, long-context LLM generation) is not
   prematurely aborted.

Every resolver is defensive: an invalid value (non-numeric, negative, empty)
at any precedence level falls back to the next level, never raises. Each key
also has a per-key ``floor`` so a typo can't make a timeout dangerously small.

The :data:`TIMEOUT_KEYS` list is the authoritative registry: the admin router
builds its ``resolved`` view from it, the admin UI renders a field per key, and
the regression test in ``tests/unit/test_timeout_config.py`` asserts every
timeout referenced by the codebase has an entry here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeoutKey:
    """Metadata for one configurable timeout.

    Fields:
        key: The admin-store / resolver key (e.g. ``"llm_request"``). Stored
            under ``timeouts.<key>`` in the SettingORM table.
        env_var: The environment variable used as the env-level fallback.
        default: The default value used when neither the admin store nor the
            env var is set.
        floor: The minimum clamped value so a typo can't make a timeout
            dangerously small.
        label: A short human-readable label shown in the admin UI.
        unit: ``"seconds"`` or ``"milliseconds"`` -- the unit of the value,
            shown in the admin UI helper text.
        group: UI grouping label (e.g. ``"LLM"``, ``"Memory"``).
    """

    key: str
    env_var: str
    default: float
    floor: float
    label: str
    unit: str = "seconds"
    group: str = "LLM"


# Authoritative registry of every configurable timeout. Add a new entry here
# when introducing a new timeout; the regression test asserts every key
# referenced by the codebase is present.
TIMEOUT_KEYS: List[TimeoutKey] = [
    # --- LLM (langchain ChatOpenAI + patched openai SDK clients) ---------
    TimeoutKey(
        key="llm_request",
        env_var="LLM_REQUEST_TIMEOUT_SECONDS",
        default=3600.0,
        floor=60.0,
        label="LLM request timeout",
        group="LLM",
    ),
    TimeoutKey(
        key="llm_retry_max_time",
        env_var="LLM_RETRY_MAX_TIME_SECONDS",
        default=900.0,
        floor=30.0,
        label="LLM retry max time",
        group="LLM",
    ),
    # Not a timeout: the process-wide requests-per-second ceiling for LLM
    # calls (see api/llm/client.py). A concurrency semaphore cannot express
    # an RPS limit — N concurrent calls can all START within one second and
    # trip the gateway's per-second budget with 429.
    TimeoutKey(
        key="llm_rate_limit_rps",
        env_var="LLM_RATE_LIMIT_RPS",
        default=5.0,
        floor=0.0,
        label="LLM requests per second (process-wide)",
        unit="requests/sec",
        group="LLM",
    ),
    # --- Docgen worker ---------------------------------------------------
    # Best-effort ceiling for draining leftover worker-loop tasks after the
    # docs are committed; a drain timeout is non-fatal by design.
    TimeoutKey(
        key="docgen_indexing_drain",
        env_var="DOCGEN_INDEXING_DRAIN_SECONDS",
        default=300.0,
        floor=5.0,
        label="Docgen indexing drain",
        group="LLM",
    ),
    # Not a timeout but a tunable knob that shares the registry's precedence
    # (admin store > env > default) + floor machinery: max parallel LLM calls
    # in the agentic docgen MAP phase (P1-24).
    TimeoutKey(
        key="docgen_map_concurrency",
        env_var="DOCGEN_MAP_CONCURRENCY",
        default=3.0,
        floor=1.0,
        label="Docgen map-phase parallel LLM calls",
        unit="parallel calls",
        group="LLM",
    ),
    # --- Docgen graph budgets (deepagents, api/docgen/codebase.py) ------
    # LangGraph recursion limits. The unit-agent default is raised from the
    # old hardcoded 60: exploring a large repo legitimately needs more than
    # 60 graph steps (larger repos died with RecursionError before). The
    # orchestrator budget also counts its subagent hand-offs, so it is larger.
    TimeoutKey(
        key="docgen_unit_recursion_limit",
        env_var="DOCGEN_UNIT_RECURSION_LIMIT",
        default=256.0,
        floor=16.0,
        label="Docgen: unit agent recursion limit",
        unit="graph steps",
        group="LLM",
    ),
    TimeoutKey(
        key="docgen_orchestrator_recursion_limit",
        env_var="DOCGEN_ORCHESTRATOR_RECURSION_LIMIT",
        default=400.0,
        floor=32.0,
        label="Docgen: orchestrator recursion limit",
        unit="graph steps",
        group="LLM",
    ),
    # Spec enrichment react agent (api/docgen/spec.py); former hardcoded 40.
    TimeoutKey(
        key="spec_recursion_limit",
        env_var="SPEC_RECURSION_LIMIT",
        default=40.0,
        floor=8.0,
        label="Docgen: spec enrichment agent recursion limit",
        unit="graph steps",
        group="LLM",
    ),
    # Promoted from the env-only DOCGEN_SECTION_CONCURRENCY knob (bounds the
    # section agents of the python-parallel fallback). Env semantics are
    # unchanged; the registry only adds the admin-store layer above it.
    TimeoutKey(
        key="docgen_section_concurrency",
        env_var="DOCGEN_SECTION_CONCURRENCY",
        default=3.0,
        floor=1.0,
        label="Docgen: parallel section agents (fallback path)",
        unit="agents",
        group="LLM",
    ),
    # Per-run ceiling on concurrently in-flight LLM calls through the ONE
    # chat instance shared by the deepagents orchestrator and its subagents
    # (langgraph's ToolNode fans task calls out with asyncio.gather; an
    # unbounded burst trips local servers' parallel-request limit with 429).
    TimeoutKey(
        key="docgen_llm_concurrency",
        env_var="DOCGEN_LLM_CONCURRENCY",
        default=8.0,
        floor=1.0,
        label="Docgen: max parallel LLM calls",
        unit="parallel calls",
        group="LLM",
    ),
    # Cross-context 0/1 toggles for codebase docgen briefs (spec digest +
    # spec_lookup tool / DB digest). Not timeouts; they share the registry
    # precedence machinery and are read through ``resolve_timeout_bool`` so
    # the legacy env truthy-string semantics of the formerly env-only
    # ``DOCGEN_DB_CONTEXT_ENABLED`` are preserved byte-for-byte (a plain
    # numeric key would silently flip existing ``=false`` deployments).
    TimeoutKey(
        key="docgen_spec_context_enabled",
        env_var="DOCGEN_SPEC_CONTEXT_ENABLED",
        default=1.0,
        floor=0.0,
        label="Docgen: spec contract context in codebase briefs",
        unit="0/1",
        group="LLM",
    ),
    TimeoutKey(
        key="docgen_db_context_enabled",
        env_var="DOCGEN_DB_CONTEXT_ENABLED",
        default=1.0,
        floor=0.0,
        label="Docgen: database context in codebase briefs",
        unit="0/1",
        group="LLM",
    ),
    # --- Expert chat (detached ask turns, api/expert/turns.py) ----------
    # Wall-clock budget for ONE expert ask turn (regular agent or deep
    # research) including retrieval, tool calls, and streaming. The turn is
    # a detached background task: on expiry the partial answer is persisted
    # and subscribers get an error frame (issue #9 — such turns legitimately
    # run for minutes on a local LLM).
    TimeoutKey(
        key="expert_stream",
        env_var="EXPERT_STREAM_TIMEOUT_SECONDS",
        default=1800.0,
        floor=60.0,
        label="Expert ask turn budget (wall-clock)",
        group="Expert",
    ),
    # LangGraph recursion budget for the expert react agents (ask stream,
    # /ask/doc, deep-research researcher). Raised from langgraph's default of
    # 25 that killed tool-heavy expert turns (memory + MCP tool loops).
    TimeoutKey(
        key="expert_recursion_limit",
        env_var="EXPERT_RECURSION_LIMIT",
        default=64.0,
        floor=8.0,
        label="Expert: agent recursion limit",
        unit="graph steps",
        group="Expert",
    ),
    # --- Memory backend (pgvector recall) -------------------------------
    TimeoutKey(
        key="memory_query",
        env_var="MEMORY_QUERY_TIMEOUT_SECONDS",
        default=30.0,
        floor=5.0,
        label="Memory backend semantic query (pgvector cosine)",
        group="Memory",
    ),
    # --- Model listing / existence checks -------------------------------
    TimeoutKey(
        key="model_list",
        env_var="MODEL_LIST_TIMEOUT_SECONDS",
        default=10.0,
        floor=1.0,
        label="Model list / existence check",
        group="LLM",
    ),
    # --- Integrations (HTTP) -------------------------------------------
    TimeoutKey(
        key="integration_http",
        env_var="INTEGRATION_HTTP_TIMEOUT_SECONDS",
        default=30.0,
        floor=5.0,
        label="Integration HTTP request",
        group="Integrations",
    ),
    TimeoutKey(
        key="git_file_content",
        env_var="GIT_FILE_CONTENT_TIMEOUT_SECONDS",
        default=30.0,
        floor=5.0,
        label="Git file-content fetch (GitHub/GitLab API)",
        group="Integrations",
    ),
    TimeoutKey(
        key="mcp_stdio_wait",
        env_var="MCP_STDIO_WAIT_SECONDS",
        default=10.0,
        floor=1.0,
        label="MCP stdio subprocess wait",
        group="Integrations",
    ),
    TimeoutKey(
        key="mermaid_verify",
        env_var="MERMAID_VERIFY_TIMEOUT",
        default=15.0,
        floor=3.0,
        label="Mermaid diagram verification",
        group="Mermaid",
    ),
    TimeoutKey(
        key="mermaid_repair",
        env_var="MERMAID_REPAIR_TIMEOUT",
        default=180.0,
        floor=10.0,
        label="Mermaid diagram LLM repair",
        group="Mermaid",
    ),
    TimeoutKey(
        key="mermaid_max_repair_attempts",
        env_var="MERMAID_MAX_REPAIR_ATTEMPTS",
        default=3.0,
        floor=1.0,
        label="Mermaid max repair attempts",
        group="Mermaid",
    ),
    TimeoutKey(
        key="mermaid_repair_deadline",
        env_var="MERMAID_REPAIR_DEADLINE_SECONDS",
        default=600.0,
        floor=10.0,
        label="Mermaid repair loop wall-clock budget",
        group="Mermaid",
    ),
    # --- Provider connection test (admin panel "Test" button) ----------
    TimeoutKey(
        key="provider_test",
        env_var="PROVIDER_TEST_TIMEOUT_SECONDS",
        default=15.0,
        floor=3.0,
        label="Provider connection test",
        group="LLM",
    ),
    # --- Database presets (api/mcp/presets.py) --------------------------
    # Wall-clock budget for the preset connection check: a REAL MCP
    # handshake (launcher spawn + initialize + tools/list + one probe tool
    # call). The default is generous because the docker-fallback launcher
    # may pull the server image on the very first check.
    TimeoutKey(
        key="db_connect_check",
        env_var="DB_CONNECT_CHECK_SECONDS",
        default=180.0,
        floor=5.0,
        label="Database preset connection check (MCP handshake)",
        group="Databases",
    ),
    # --- Database RE docgen budgets (walk + render, api/docgen/database.py) --
    # Not timeouts but tunable knobs sharing the registry's precedence
    # (admin store > env > default) + floor machinery, like
    # docgen_map_concurrency. Values are read on every generation, so an
    # admin change applies to the next run without a restart.
    TimeoutKey(
        key="db_docgen_enrich_batch",
        env_var="DB_DOCGEN_ENRICH_BATCH",
        default=40.0,
        floor=5.0,
        label="Database docgen: tables per LLM description batch",
        unit="tables per LLM call",
        group="Databases",
    ),
    # 0 = unlimited on the three caps below: every non-system entity gets a
    # subpage, a description batch and (where the server can resolve it) a
    # source fetch. The walk stays bounded by MAX_TOOL_CALLS + the wall-clock
    # budget, the render side only builds deterministic strings.
    TimeoutKey(
        key="db_docgen_max_subpages",
        env_var="DB_DOCGEN_MAX_SUBPAGES",
        default=0.0,
        floor=0.0,
        label="Database docgen: max entity subpages (0 = unlimited)",
        unit="pages",
        group="Databases",
    ),
    TimeoutKey(
        key="db_docgen_max_descriptions",
        env_var="DB_DOCGEN_MAX_DESCRIPTIONS",
        default=0.0,
        floor=0.0,
        label="Database docgen: max LLM-described objects (0 = unlimited)",
        unit="objects",
        group="Databases",
    ),
    TimeoutKey(
        key="db_fk_evidence_tables",
        env_var="DB_FK_EVIDENCE_TABLES",
        default=1000.0,
        floor=1.0,
        label="Database RE: tables probed for constraints/indexes",
        unit="tables",
        group="Databases",
    ),
    TimeoutKey(
        key="db_source_objects",
        env_var="DB_SOURCE_OBJECTS",
        default=0.0,
        floor=0.0,
        label="Database RE: objects with fetched source (0 = unlimited)",
        unit="objects",
        group="Databases",
    ),
]

# Fast lookup by key + by env var.
_BY_KEY: dict = {k.key: k for k in TIMEOUT_KEYS}
_BY_ENV: dict = {k.env_var: k for k in TIMEOUT_KEYS}


def _parse_float(raw: object) -> Optional[float]:
    """Parse a value to float, returning None on any failure (never raises)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        s = str(raw).strip()
        if not s:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _resolve_with_key(key: str) -> float:
    """Resolve a timeout by admin key with full precedence + floor + fallback.

    1. admin store ``timeouts.<key>``  -> if set + parses + >= 0, use it
    2. env var ``env_var``             -> if set + parses + >= 0, use it
    3. default                         -> the per-key constant

    Any invalid value falls through to the next level. The final value is
    clamped to the key's floor so a typo can't make a timeout dangerously small.
    """
    spec = _BY_KEY.get(key)
    if spec is None:
        logger.debug("resolve_timeout: unknown key %r; returning 0", key)
        return 0.0

    setting_key = f"timeouts.{key}"

    # 1. Admin settings store (import-safe, DB-down-safe).
    try:
        from api.config.settings import get_setting

        store_val = get_setting(setting_key)
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_setting(%r) failed: %s", setting_key, e)
        store_val = None

    parsed = _parse_float(store_val)
    if parsed is not None and parsed >= 0:
        # Admin store wins, but still honor the floor.
        return max(spec.floor, parsed)

    # 2. Environment variable.
    env_val = os.environ.get(spec.env_var)
    parsed = _parse_float(env_val)
    if parsed is not None and parsed >= 0:
        return max(spec.floor, parsed)

    # 3. Default.
    return max(spec.floor, spec.default)


def _parse_bool_strict(raw: object) -> Optional[bool]:
    """Parse an ADMIN-STORE boolean value; None = unset or a typo.

    Accepts the numeric forms the admin UI writes (``0``/``1``) plus the
    common word forms; anything else is treated as a typo and skipped, so
    the precedence level falls through instead of guessing.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return float(raw) != 0.0
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def resolve_timeout_bool(key: str) -> bool:
    """Resolve a 0/1 registry flag with the full precedence chain.

    1. admin store — strict parse (``0/1/true/false/yes/no/on/off``); unset
       or a typo falls through to the next level.
    2. env var — the LEGACY truthy-string semantics of the former env-only
       toggles: anything except ``0/false/no/off`` (including an empty
       value) is ON. This preserves existing deployments byte-for-byte —
       ``_parse_float("false")`` is None, so a plain numeric key would
       silently flip an ``DOCGEN_DB_CONTEXT_ENABLED=false`` install.
    3. default — the per-key constant.

    Never raises; an unknown key returns ``False`` (logged).
    """
    spec = _BY_KEY.get(key)
    if spec is None:
        logger.debug("resolve_timeout_bool: unknown key %r; returning False", key)
        return False

    try:
        from api.config.settings import get_setting

        store_val = get_setting(f"timeouts.{key}")
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_setting(%r) failed: %s", key, e)
        store_val = None

    parsed = _parse_bool_strict(store_val)
    if parsed is not None:
        return parsed

    env_val = os.environ.get(spec.env_var)
    if env_val is not None:
        return (env_val or "").strip().lower() not in ("0", "false", "no", "off")

    return bool(spec.default)


def resolve_timeout(key: str) -> float:
    """Resolve a timeout (seconds or milliseconds) to a float.

    Never raises. Unknown keys return ``0.0`` (and log a debug warning) so a
    typo never crashes the caller.
    """
    return _resolve_with_key(key)


def resolve_timeout_int(key: str) -> int:
    """Resolve a timeout to an int (e.g. for ``mermaid_max_repair_attempts``)."""
    return int(_resolve_with_key(key))


# --- Thin named wrappers (the public API most modules import directly) ----
# Keep the wrapper names aligned with the legacy env-only resolvers so the
# existing call sites + tests keep working unchanged.
def resolve_llm_request_timeout() -> float:
    """Per-request HTTP timeout for OpenAI-compatible clients (seconds)."""
    return resolve_timeout("llm_request")


def resolve_llm_retry_max_time() -> float:
    """Total backoff retry budget for transient errors on OpenAI clients (seconds)."""
    return resolve_timeout("llm_retry_max_time")


def resolve_llm_rate_limit_rps() -> float:
    """Process-wide LLM requests-per-second ceiling (0 = spacing disabled).

    Enforced by ``ServerCompatChatOpenAI`` on every async call through any
    chat instance built by :func:`api.llm.client.build_chat_model` — one
    monotonic slot queue per (base_url, model), shared across all event
    loops in the process. Read per call so an admin save applies without a
    restart.
    """
    return resolve_timeout("llm_rate_limit_rps")


def resolve_docgen_indexing_drain_seconds() -> float:
    """Best-effort ceiling for the docgen worker-loop indexing drain (seconds).

    Leftover worker-loop tasks (e.g. a memory-indexing task that wasn't handed
    off to the main event loop) get this budget instead of being cancelled
    immediately; a drain timeout is non-fatal because the docs are already
    committed.
    """
    return resolve_timeout("docgen_indexing_drain")


def resolve_docgen_map_concurrency() -> int:
    """Max parallel LLM calls in the agentic docgen MAP phase (P1-24).

    The Phase-1 map loop (``_agentic_bottom_up_docgen``) issues one independent
    LLM call per codebase chunk; awaiting them strictly sequentially wastes wall
    clock on a local model. This knob bounds the parallelism (asyncio.Semaphore
    + gather) so the local server is not flooded. The RLM map loop stays
    sequential (shared REPL session). Floor 1 = sequential fallback.
    """
    return resolve_timeout_int("docgen_map_concurrency")


def resolve_docgen_unit_recursion_limit() -> int:
    """LangGraph ``recursion_limit`` for one unit-agent run (section subpage).

    Raised from the old hardcoded 60: deepagents explorations of large repos
    legitimately exceed 60 graph steps. Read per agent run, so an admin change
    applies to the next docgen run without a restart.
    """
    return resolve_timeout_int("docgen_unit_recursion_limit")


def resolve_docgen_orchestrator_recursion_limit() -> int:
    """LangGraph ``recursion_limit`` for the deepagents orchestrator run.

    The orchestrator budget also covers its subagent hand-offs (task calls
    inherit the parent config's recursion_limit), so it must stay above the
    unit limit.
    """
    return resolve_timeout_int("docgen_orchestrator_recursion_limit")


def resolve_spec_recursion_limit() -> int:
    """LangGraph ``recursion_limit`` for the spec enrichment react agent."""
    return resolve_timeout_int("spec_recursion_limit")


def resolve_docgen_section_concurrency() -> int:
    """Max parallel section agents in the python-parallel fallback path.

    Formerly the env-only ``DOCGEN_SECTION_CONCURRENCY``; the caller still
    clamps the result to the number of sections.
    """
    return resolve_timeout_int("docgen_section_concurrency")


def resolve_docgen_llm_concurrency() -> int:
    """Max concurrently in-flight LLM calls per docgen run.

    Bounds the shared chat instance (orchestrator + subagents + fallback
    sections) with an asyncio.Semaphore so a burst of task-dispatched
    subagents cannot exceed the local server's parallel-request limit.
    Floor 1 = fully sequential fallback.
    """
    return resolve_timeout_int("docgen_llm_concurrency")


def resolve_docgen_spec_context_enabled() -> bool:
    """Cross-context toggle: spec digest + ``spec_lookup`` in codebase docgen.

    Own API contracts and external client contracts ride into the codebase
    docgen brief as a deterministic menu, with schema/operation details
    available on demand through the ``spec_lookup`` tool.
    """
    return resolve_timeout_bool("docgen_spec_context_enabled")


def resolve_docgen_db_context_enabled() -> bool:
    """Cross-context toggle: DB digest in codebase docgen briefs.

    Wraps the formerly env-only ``DOCGEN_DB_CONTEXT_ENABLED``; legacy env
    truthy-string semantics are preserved (see ``resolve_timeout_bool``).
    """
    return resolve_timeout_bool("docgen_db_context_enabled")


def resolve_expert_stream_timeout() -> float:
    """Wall-clock budget for one detached expert ask turn (seconds).

    Bounds the background task in ``api/expert/turns.py`` (regular agent or
    deep research). On expiry the turn is cancelled, the partial answer is
    persisted, and subscribers receive an error frame.
    """
    return resolve_timeout("expert_stream")


def resolve_expert_recursion_limit() -> int:
    """LangGraph ``recursion_limit`` for the expert react agents.

    One budget for the ask stream, the one-shot doc turn, and the
    deep-research researcher — all the same react-agent shape. Replaces the
    old hardcoded 25 (langgraph's default); read per request so an admin
    change applies to the next turn without a restart.
    """
    return resolve_timeout_int("expert_recursion_limit")


def resolve_memory_query_timeout() -> float:
    """pgvector cosine-recall query timeout (seconds).

    Caps the top-k semantic query (embed query + ORDER BY embedding <=> :q) so
    a slow embedder or a contended Postgres cannot stall the expert SSE stream.
    On timeout the pgvector backend returns "" and the expert falls back to
    artifact docs.
    """
    return resolve_timeout("memory_query")


def resolve_model_list_timeout() -> float:
    """HTTP timeout for model listing / existence checks (seconds)."""
    return resolve_timeout("model_list")


def resolve_integration_http_timeout() -> float:
    """HTTP timeout for integration connectors (seconds)."""
    return resolve_timeout("integration_http")


def resolve_git_file_content_timeout() -> float:
    """HTTP timeout for GitHub/GitLab file-content API fetches (seconds)."""
    return resolve_timeout("git_file_content")


def resolve_mcp_stdio_wait_timeout() -> float:
    """Timeout for MCP stdio subprocess ``.wait()`` (seconds)."""
    return resolve_timeout("mcp_stdio_wait")


def resolve_mermaid_verify_timeout() -> float:
    """Per-diagram Node verification timeout (seconds)."""
    return resolve_timeout("mermaid_verify")


def resolve_mermaid_repair_timeout() -> float:
    """Per-LLM-repair-call timeout for mermaid (seconds)."""
    return resolve_timeout("mermaid_repair")


def resolve_mermaid_max_repair_attempts() -> int:
    """Max LLM repair attempts per unique mermaid diagram body."""
    return resolve_timeout_int("mermaid_max_repair_attempts")


def resolve_mermaid_repair_deadline() -> float:
    """Wall-clock budget for the whole mermaid repair drain (seconds).

    P1-15: caps the TOTAL time run_repair_loop may spend draining repairs for
    one page, so a mutating LLM that produces a fresh broken body on every
    call cannot spin the loop indefinitely (each new body hash previously
    earned its own per-body budget).
    """
    return resolve_timeout("mermaid_repair_deadline")


def resolve_provider_test_timeout() -> float:
    """HTTP timeout for the admin panel provider connection test (seconds)."""
    return resolve_timeout("provider_test")


def resolve_db_connect_check_timeout() -> float:
    """Wall-clock budget for the preset DB connection check (seconds).

    Bounds the full MCP handshake (launcher spawn + initialize + tools/list
    + one probe tool call) in ``api/mcp/presets.check_preset_connection``.
    """
    return resolve_timeout("db_connect_check")


def resolve_db_docgen_enrich_batch() -> int:
    """Tables per LLM description batch in database docgen.

    Table descriptions are enriched in strict-JSON batches (not per page) so
    a 61-table Postgres or a multi-thousand-table Oracle monolith costs
    dozens — not thousands — of LLM calls. Read per generation: an admin
    change applies to the next run without a restart.
    """
    return resolve_timeout_int("db_docgen_enrich_batch")


def resolve_db_docgen_max_subpages() -> int:
    """Max entity subpages (tables + category objects) rendered per database.

    0 = unlimited (default). Above a finite cap the surplus stays on the
    category root page in a folded (``<details>``) list — hidden, never
    dropped.
    """
    return resolve_timeout_int("db_docgen_max_subpages")


def resolve_db_docgen_max_descriptions() -> int:
    """Max objects that get an LLM description per database run.

    0 = unlimited (default); objects per LLM call follow the batch knob.
    """
    return resolve_timeout_int("db_docgen_max_descriptions")


def resolve_db_fk_evidence_tables() -> int:
    """Max tables probed for constraints/indexes during the RE walk.

    Walk-affecting budget: participates in the introspection cache key.
    """
    return resolve_timeout_int("db_fk_evidence_tables")


def resolve_db_source_objects() -> int:
    """Max non-table objects (views/routines/triggers/…) with fetched source.

    0 = unlimited (default). Walk-affecting budget: participates in the
    introspection cache key.
    """
    return resolve_timeout_int("db_source_objects")


def sync_timeout_env() -> None:
    """Export admin-store timeout overrides to their canonical env vars.

    Called from :func:`api.config.abstraction.sync_runtime_settings` at startup
    and after every admin save. This makes module-level / subprocess readers
    (which read the canonical env vars from the process environment) see
    admin-set values without a restart.

    Best-effort and never raises: a missing settings store or an invalid value
    just leaves the env var untouched.
    """
    for spec in TIMEOUT_KEYS:
        try:
            from api.config.settings import get_setting

            store_val = get_setting(f"timeouts.{spec.key}")
        except Exception as e:  # pragma: no cover - settings store is import-safe
            logger.debug("sync_timeout_env: get_setting(%r) failed: %s", spec.key, e)
            continue
        parsed = _parse_float(store_val)
        if parsed is not None and parsed >= 0:
            # Env vars hold raw numbers; for integer-ish values keep them clean.
            if spec.unit == "milliseconds" or float(parsed).is_integer():
                os.environ[spec.env_var] = str(int(parsed))
            else:
                os.environ[spec.env_var] = str(parsed)


def get_timeout_resolved_view() -> dict:
    """Build the ``resolved`` view for the admin ``timeouts`` group GET.

    Returns a dict keyed by timeout key with the effective value, default, and
    floor, so the UI can show the current effective value and label each field.
    """
    out: dict = {}
    for spec in TIMEOUT_KEYS:
        effective = _resolve_with_key(spec.key)
        out[spec.key] = {
            "value": str(int(effective)) if float(effective).is_integer() else str(effective),
            "default": str(int(spec.default)) if float(spec.default).is_integer() else str(spec.default),
            "floor": str(int(spec.floor)) if float(spec.floor).is_integer() else str(spec.floor),
            "env_var": spec.env_var,
            "label": spec.label,
            "unit": spec.unit,
            "group": spec.group,
        }
    return out


__all__ = [
    "TIMEOUT_KEYS",
    "TimeoutKey",
    "resolve_timeout",
    "resolve_timeout_int",
    "resolve_llm_request_timeout",
    "resolve_llm_retry_max_time",
    "resolve_llm_rate_limit_rps",
    "resolve_expert_stream_timeout",
    "resolve_expert_recursion_limit",
    "resolve_docgen_indexing_drain_seconds",
    "resolve_docgen_map_concurrency",
    "resolve_docgen_unit_recursion_limit",
    "resolve_docgen_orchestrator_recursion_limit",
    "resolve_spec_recursion_limit",
    "resolve_docgen_section_concurrency",
    "resolve_docgen_llm_concurrency",
    "resolve_docgen_spec_context_enabled",
    "resolve_docgen_db_context_enabled",
    "resolve_timeout_bool",
    "resolve_memory_query_timeout",
    "resolve_model_list_timeout",
    "resolve_integration_http_timeout",
    "resolve_git_file_content_timeout",
    "resolve_mcp_stdio_wait_timeout",
    "resolve_mermaid_verify_timeout",
    "resolve_mermaid_repair_timeout",
    "resolve_mermaid_max_repair_attempts",
    "resolve_mermaid_repair_deadline",
    "resolve_provider_test_timeout",
    "resolve_db_connect_check_timeout",
    "resolve_db_docgen_enrich_batch",
    "resolve_db_docgen_max_subpages",
    "resolve_db_docgen_max_descriptions",
    "resolve_db_fk_evidence_tables",
    "resolve_db_source_objects",
    "sync_timeout_env",
    "get_timeout_resolved_view",
]
