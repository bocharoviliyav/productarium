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
   readers (e.g. fast-rlm Pyodide reads ``RLM_API_TIMEOUT_MS``) also see it.
2. Environment variable (the ``env_var`` for the key) -- the fallback when the
   admin store is unset or the DB is down. Also documented in ``.env.example``.
3. Default value -- a sensible per-key constant, raised so long-running work
   on large repos (multi-hour cognify, long-context RLM generation) is not
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
            env var is set. For ``docgen_indexing_drain`` this is a sentinel
            consumed by its resolver (the default is derived from the cognify
            timeout at call time).
        floor: The minimum clamped value so a typo can't make a timeout
            dangerously small.
        label: A short human-readable label shown in the admin UI.
        unit: ``"seconds"`` or ``"milliseconds"`` -- the unit of the value,
            shown in the admin UI helper text.
        group: UI grouping label (e.g. ``"LLM"``, ``"Cognee"``).
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
    # --- LLM (adalflow OpenAIClient + patched openai SDK clients) ---------
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
    # --- Cognee (knowledge graph) ---------------------------------------
    TimeoutKey(
        key="cognee_graph_extraction",
        env_var="COGNEE_GRAPH_EXTRACTION_TIMEOUT",
        # Per-chunk structured-output LLM call (instructor + markdown_json_mode)
        # over a cognee chunk. A slow corporate gateway can take several minutes
        # for a single JSON-schema extraction (the mode appends a "return JSON"
        # instruction and parses a JSON block out of the plain-text completion,
        # so the model emits a full section's worth of tokens before the parse).
        # The previous 600s (10 min) default bit on corporate LLMs and logged
        # "graph extraction skipped chunk due to TimeoutError" repeatedly.
        default=1800.0,
        floor=60.0,
        label="Cognee graph extraction (per chunk)",
        group="Cognee",
    ),
    TimeoutKey(
        key="cognee_cognify",
        env_var="COGNEE_COGNIFY_TIMEOUT",
        default=7200.0,
        floor=300.0,
        label="Cognee cognify (full run)",
        group="Cognee",
    ),
    TimeoutKey(
        key="cognee_llm_connection",
        env_var="COGNEE_LLM_CONNECTION_TIMEOUT",
        default=10.0,
        floor=1.0,
        label="Cognee LLM connection test",
        group="Cognee",
    ),
    TimeoutKey(
        key="cognee_init",
        env_var="COGNEE_INIT_TIMEOUT",
        default=120.0,
        floor=10.0,
        label="Cognee startup init (migrations)",
        group="Cognee",
    ),
    TimeoutKey(
        key="cognee_recall",
        env_var="COGNEE_RECALL_TIMEOUT",
        default=120.0,
        floor=15.0,
        label="Cognee recall (graph query)",
        group="Cognee",
    ),
    # --- Docgen worker ---------------------------------------------------
    # default is a sentinel; the resolver derives the effective default from
    # cognee_cognify at call time so a leftover cognify task gets the full
    # budget instead of being killed at 30s.
    TimeoutKey(
        key="docgen_indexing_drain",
        env_var="DOCGEN_INDEXING_DRAIN_SECONDS",
        default=-1.0,
        floor=5.0,
        label="Docgen indexing drain",
        group="Cognee",
    ),
    # --- RLM (fast-rlm) --------------------------------------------------
    TimeoutKey(
        key="rlm_api_ms",
        env_var="RLM_API_TIMEOUT_MS",
        default=3_600_000.0,
        floor=30_000.0,
        label="RLM per-API-call timeout",
        unit="milliseconds",
        group="RLM",
    ),
    TimeoutKey(
        key="rlm_section",
        env_var="RLM_SECTION_TIMEOUT",
        default=1800.0,
        floor=60.0,
        label="RLM per-section timeout (docgen)",
        group="RLM",
    ),
    TimeoutKey(
        key="rlm_expert",
        env_var="RLM_EXPERT_TIMEOUT",
        default=1800.0,
        floor=60.0,
        label="RLM expert timeout",
        group="RLM",
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
    # --- Provider connection test (admin panel "Test" button) ----------
    TimeoutKey(
        key="provider_test",
        env_var="PROVIDER_TEST_TIMEOUT_SECONDS",
        default=15.0,
        floor=3.0,
        label="Provider connection test",
        group="LLM",
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
    3. default                         -> for ``docgen_indexing_drain`` the
       default is derived from the cognify timeout at call time

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
        from api.settings_store import get_setting

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

    # 3. Default. docgen_indexing_drain derives from the cognify timeout so a
    # leftover cognify task (that wasn't handed off to the main loop) gets the
    # full cognify budget instead of being cancelled at a fixed 30s.
    if key == "docgen_indexing_drain":
        return max(spec.floor, _resolve_with_key("cognee_cognify"))

    return max(spec.floor, spec.default)


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


def resolve_cognee_graph_extraction_timeout() -> float:
    """Per-chunk cognee graph-extraction LLM call timeout (seconds)."""
    return resolve_timeout("cognee_graph_extraction")


def resolve_cognee_cognify_timeout() -> float:
    """Overall timeout for a full cognee.cognify() run (seconds)."""
    return resolve_timeout("cognee_cognify")


def resolve_cognee_llm_connection_timeout() -> float:
    """Cognee LLM connection test timeout (seconds)."""
    return resolve_timeout("cognee_llm_connection")


def resolve_cognee_init_timeout() -> float:
    """Overall timeout for ``init_cognee()`` (startup migrations + setup).

    ``cognee.init()`` / ``run_startup_migrations()`` / ``setup()`` can each
    stall on a slow/unreachable Postgres, blocking app startup. This ceiling
    lets the app start and serve requests (docgen reads/writes the product DB
    directly; expert/summary fall back to artifact docs) while cognee finishes
    initializing -- or gives up -- in the background. cognee creates tables
    lazily on first write, so a skipped init is non-fatal.
    """
    return resolve_timeout("cognee_init")


def resolve_cognee_recall_timeout() -> float:
    """Overall timeout for a single ``cognee.recall()`` query (seconds).

    recall's GRAPH_COMPLETION makes an LLM completion call internally; on a
    slow/contended local model this can hang indefinitely and block the
    expert SSE stream. This ceiling returns "" (-> artifact-docs fallback)
    instead of letting the request hang until the HTTP proxy times out.
    Default 120s (down from 300s): cognee is non-blocking, so a recall that
    hasn't answered in two minutes is treated as unavailable and the expert
    falls back to the artifact-docs baseline -- the user should NOT wait 5
    minutes for a graph query before seeing an answer.
    """
    return resolve_timeout("cognee_recall")


def resolve_docgen_indexing_drain_seconds() -> float:
    """Best-effort ceiling for the docgen worker-loop indexing drain (seconds).

    Default derives from the cognee cognify timeout so a leftover cognify task
    that wasn't handed off to the main event loop gets the full cognify budget
    instead of being cancelled at a fixed 30s (which previously dropped the
    connection mid-graph-build).
    """
    return resolve_timeout("docgen_indexing_drain")


def resolve_rlm_api_timeout_ms() -> int:
    """fast-rlm per-API-call timeout (milliseconds)."""
    return resolve_timeout_int("rlm_api_ms")


def resolve_rlm_section_timeout() -> float:
    """Per-section RLM timeout for the docgen path (seconds)."""
    return resolve_timeout("rlm_section")


def resolve_rlm_expert_timeout() -> float:
    """Per-section RLM timeout for the expert-agent path (seconds)."""
    return resolve_timeout("rlm_expert")


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


def resolve_provider_test_timeout() -> float:
    """HTTP timeout for the admin panel provider connection test (seconds)."""
    return resolve_timeout("provider_test")


def sync_timeout_env() -> None:
    """Export admin-store timeout overrides to their canonical env vars.

    Called from :func:`api.config_abstraction.sync_runtime_settings` at startup
    and after every admin save. This makes module-level / subprocess readers
    (e.g. fast-rlm's Pyodide REPL, which reads ``RLM_API_TIMEOUT_MS`` from the
    process environment and cannot reach host Python) see admin-set values
    without a restart.

    Best-effort and never raises: a missing settings store or an invalid value
    just leaves the env var untouched.
    """
    for spec in TIMEOUT_KEYS:
        try:
            from api.settings_store import get_setting

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
    "resolve_cognee_graph_extraction_timeout",
    "resolve_cognee_cognify_timeout",
    "resolve_cognee_llm_connection_timeout",
    "resolve_docgen_indexing_drain_seconds",
    "resolve_rlm_api_timeout_ms",
    "resolve_rlm_section_timeout",
    "resolve_rlm_expert_timeout",
    "resolve_model_list_timeout",
    "resolve_integration_http_timeout",
    "resolve_git_file_content_timeout",
    "resolve_mcp_stdio_wait_timeout",
    "resolve_mermaid_verify_timeout",
    "resolve_mermaid_repair_timeout",
    "resolve_mermaid_max_repair_attempts",
    "resolve_provider_test_timeout",
    "sync_timeout_env",
    "get_timeout_resolved_view",
]
