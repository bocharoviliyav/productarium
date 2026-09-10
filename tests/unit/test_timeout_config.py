"""Tests for the central timeout resolver (api.timeout_config).

Covers:
- Precedence: admin store (timeouts.<key>) > env var > default.
- Invalid-value fallback at every precedence level (never raises).
- Per-key floor enforcement (a typo can't make a timeout dangerously small).
- docgen_indexing_drain resolves from its own registry entry (own default
  + floor; no derivation from removed legacy keys).
- TIMEOUT_KEYS regression guard: every wrapper imported by the routed files
  maps to an entry in TIMEOUT_KEYS, so a new timeout can't be added to
  the codebase without being registered here (and thus surfaced in the admin
  panel + .env.example docs).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from api.config.timeout import (
    TIMEOUT_KEYS,
    TimeoutKey,
    _BY_ENV,
    _BY_KEY,
    _resolve_with_key,
    get_timeout_resolved_view,
    resolve_db_connect_check_timeout,
    resolve_db_docgen_enrich_batch,
    resolve_db_docgen_max_descriptions,
    resolve_db_docgen_max_subpages,
    resolve_db_fk_evidence_tables,
    resolve_db_source_objects,
    resolve_docgen_db_context_enabled,
    resolve_docgen_indexing_drain_seconds,
    resolve_docgen_llm_concurrency,
    resolve_docgen_map_concurrency,
    resolve_docgen_orchestrator_recursion_limit,
    resolve_docgen_section_concurrency,
    resolve_docgen_spec_context_enabled,
    resolve_docgen_unit_recursion_limit,
    resolve_expert_stream_timeout,
    resolve_timeout,
    resolve_timeout_bool,
    resolve_timeout_int,
    sync_timeout_env,
)


# Every wrapper name imported by the routed files. Kept in sync with the
# grep over api/ for `resolve_*_timeout` / `resolve_*_ms` / `resolve_*_attempts`.
# If a new wrapper is added to the codebase, add it here AND to TIMEOUT_KEYS.
WRAPPER_TO_KEY = {
    "resolve_llm_request_timeout": "llm_request",
    "resolve_llm_retry_max_time": "llm_retry_max_time",
    "resolve_docgen_indexing_drain_seconds": "docgen_indexing_drain",
    "resolve_docgen_map_concurrency": "docgen_map_concurrency",
    "resolve_docgen_unit_recursion_limit": "docgen_unit_recursion_limit",
    "resolve_docgen_orchestrator_recursion_limit": "docgen_orchestrator_recursion_limit",
    "resolve_docgen_section_concurrency": "docgen_section_concurrency",
    "resolve_docgen_llm_concurrency": "docgen_llm_concurrency",
    "resolve_docgen_spec_context_enabled": "docgen_spec_context_enabled",
    "resolve_docgen_db_context_enabled": "docgen_db_context_enabled",
    "resolve_expert_stream_timeout": "expert_stream",
    "resolve_memory_query_timeout": "memory_query",
    "resolve_model_list_timeout": "model_list",
    "resolve_integration_http_timeout": "integration_http",
    "resolve_git_file_content_timeout": "git_file_content",
    "resolve_mcp_stdio_wait_timeout": "mcp_stdio_wait",
    "resolve_mermaid_verify_timeout": "mermaid_verify",
    "resolve_mermaid_repair_timeout": "mermaid_repair",
    "resolve_mermaid_max_repair_attempts": "mermaid_max_repair_attempts",
    "resolve_mermaid_repair_deadline": "mermaid_repair_deadline",
    "resolve_provider_test_timeout": "provider_test",
    "resolve_db_connect_check_timeout": "db_connect_check",
    "resolve_db_docgen_enrich_batch": "db_docgen_enrich_batch",
    "resolve_db_docgen_max_subpages": "db_docgen_max_subpages",
    "resolve_db_docgen_max_descriptions": "db_docgen_max_descriptions",
    "resolve_db_fk_evidence_tables": "db_fk_evidence_tables",
    "resolve_db_source_objects": "db_source_objects",
}


class _EnvGuard:
    """Snapshot/restore a set of env vars + admin-store overrides."""

    def __init__(self, env_vars, store_keys):
        self._env = list(env_vars)
        self._store = list(store_keys)
        self._saved_env: dict = {}
        self._saved_store: list = []

    def __enter__(self):
        for k in self._env:
            self._saved_env[k] = os.environ.pop(k, None)
        from api.config.settings import list_settings
        self._saved_store = [r["key"] for r in list_settings(prefix="timeouts.")]
        # Clear any stored override so the env/default layers are tested clean.
        from api.config.settings import set_setting
        for key in self._saved_store:
            set_setting(key, "", encrypt=False)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        from api.config.settings import set_setting
        for key in self._saved_store:
            set_setting(key, "", encrypt=False)
        return False


class TestTimeoutConfig(unittest.TestCase):
    # ------------------------------------------------------------------
    # Registry / structural invariants
    # ------------------------------------------------------------------
    def test_timeout_keys_has_every_wrapper_key(self):
        """Every wrapper imported by the routed files is in TIMEOUT_KEYS."""
        keys = {k.key for k in TIMEOUT_KEYS}
        for wrapper, key in WRAPPER_TO_KEY.items():
            self.assertIn(
                key, keys,
                f"wrapper {wrapper} maps to key {key!r} missing from TIMEOUT_KEYS",
            )

    def test_timeout_keys_env_vars_unique(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        self.assertEqual(len(env_vars), len(set(env_vars)), "duplicate env vars")

    def test_timeout_keys_lookup_dicts(self):
        for k in TIMEOUT_KEYS:
            self.assertIs(_BY_KEY[k.key], k)
            self.assertIs(_BY_ENV[k.env_var], k)

    def test_unknown_key_returns_zero(self):
        # A typo in a resolve_timeout("...") call must not crash the caller.
        self.assertEqual(resolve_timeout("does_not_exist"), 0.0)
        self.assertEqual(resolve_timeout_int("does_not_exist"), 0)

    # ------------------------------------------------------------------
    # Precedence: admin store > env var > default
    # ------------------------------------------------------------------
    def test_default_used_when_nothing_set(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            for k in TIMEOUT_KEYS:
                self.assertEqual(
                    _resolve_with_key(k.key),
                    max(k.floor, k.default),
                    f"default mismatch for {k.key}",
                )

    def test_env_var_overrides_default(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "9999"
            self.assertEqual(resolve_timeout("llm_request"), 9999.0)

    def test_admin_store_overrides_env_and_default(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with patch("api.config.settings.get_setting") as mock_get, _EnvGuard(env_vars, []):
            # First call (admin store) returns a stored override; the env var
            # is also set, but the admin store must win.
            def _get(key, *a, **kw):
                if key == "timeouts.llm_request":
                    return "7777"
                return None
            mock_get.side_effect = _get
            os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "9999"
            self.assertEqual(resolve_timeout("llm_request"), 7777.0)

    # ------------------------------------------------------------------
    # Invalid-value fallback (never raises)
    # ------------------------------------------------------------------
    def test_invalid_env_falls_back_to_default(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "not-a-number"
            self.assertEqual(resolve_timeout("llm_request"), 3600.0)
            os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = ""
            self.assertEqual(resolve_timeout("llm_request"), 3600.0)

    def test_negative_env_falls_back_to_default(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "-5"
            self.assertEqual(resolve_timeout("llm_request"), 3600.0)

    def test_invalid_admin_store_falls_back_to_env(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with patch("api.config.settings.get_setting") as mock_get, _EnvGuard(env_vars, []):
            def _get(key, *a, **kw):
                if key == "timeouts.llm_request":
                    return "garbage"
                return None
            mock_get.side_effect = _get
            os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "4321"
            self.assertEqual(resolve_timeout("llm_request"), 4321.0)

    # ------------------------------------------------------------------
    # Floor enforcement
    # ------------------------------------------------------------------
    def test_env_below_floor_is_clamped(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            # floor for llm_request is 60.
            os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "1"
            self.assertEqual(resolve_timeout("llm_request"), 60.0)

    def test_admin_store_below_floor_is_clamped(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with patch("api.config.settings.get_setting") as mock_get, _EnvGuard(env_vars, []):
            def _get(key, *a, **kw):
                if key == "timeouts.llm_request":
                    return "1"
                return None
            mock_get.side_effect = _get
            self.assertEqual(resolve_timeout("llm_request"), 60.0)

    # ------------------------------------------------------------------
    # docgen_indexing_drain (own registry entry: default 300, floor 5)
    # ------------------------------------------------------------------
    def test_docgen_drain_default_when_nothing_set(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            self.assertEqual(resolve_docgen_indexing_drain_seconds(), 300.0)

    def test_docgen_drain_env_override(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_INDEXING_DRAIN_SECONDS"] = "120"
            self.assertEqual(resolve_docgen_indexing_drain_seconds(), 120.0)

    # ------------------------------------------------------------------
    # docgen_map_concurrency (P1-24): bounded MAP-phase parallelism
    # ------------------------------------------------------------------
    def test_docgen_map_concurrency_default(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            self.assertEqual(resolve_docgen_map_concurrency(), 3)

    def test_docgen_map_concurrency_env_override(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_MAP_CONCURRENCY"] = "8"
            self.assertEqual(resolve_docgen_map_concurrency(), 8)

    def test_docgen_map_concurrency_floor_one(self):
        # 0 / negative / garbage all clamp/fall back to at least 1
        # (sequential), never to unlimited or zero.
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_MAP_CONCURRENCY"] = "0"
            self.assertEqual(resolve_docgen_map_concurrency(), 1)
            os.environ["DOCGEN_MAP_CONCURRENCY"] = "garbage"
            self.assertEqual(resolve_docgen_map_concurrency(), 3)

    # ------------------------------------------------------------------
    # docgen agent budgets: recursion limits + parallelism (admin-tunable)
    # ------------------------------------------------------------------
    def test_docgen_recursion_limit_defaults(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            self.assertEqual(resolve_docgen_unit_recursion_limit(), 256)
            self.assertEqual(resolve_docgen_orchestrator_recursion_limit(), 400)

    def test_docgen_recursion_limit_env_override(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_UNIT_RECURSION_LIMIT"] = "128"
            self.assertEqual(resolve_docgen_unit_recursion_limit(), 128)
            os.environ["DOCGEN_ORCHESTRATOR_RECURSION_LIMIT"] = "600"
            self.assertEqual(resolve_docgen_orchestrator_recursion_limit(), 600)

    def test_docgen_recursion_limit_floors(self):
        # A typo can't drop a recursion limit below its floor (16 / 32).
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_UNIT_RECURSION_LIMIT"] = "1"
            self.assertEqual(resolve_docgen_unit_recursion_limit(), 16)
            os.environ["DOCGEN_ORCHESTRATOR_RECURSION_LIMIT"] = "5"
            self.assertEqual(resolve_docgen_orchestrator_recursion_limit(), 32)

    def test_docgen_concurrency_knob_defaults(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            self.assertEqual(resolve_docgen_section_concurrency(), 3)
            self.assertEqual(resolve_docgen_llm_concurrency(), 8)

    def test_docgen_concurrency_knob_env_override(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_SECTION_CONCURRENCY"] = "5"
            self.assertEqual(resolve_docgen_section_concurrency(), 5)
            os.environ["DOCGEN_LLM_CONCURRENCY"] = "2"
            self.assertEqual(resolve_docgen_llm_concurrency(), 2)

    def test_docgen_concurrency_knob_floors(self):
        # 0 clamps to 1 (sequential); garbage falls back to the default.
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_SECTION_CONCURRENCY"] = "0"
            self.assertEqual(resolve_docgen_section_concurrency(), 1)
            os.environ["DOCGEN_LLM_CONCURRENCY"] = "0"
            self.assertEqual(resolve_docgen_llm_concurrency(), 1)
            os.environ["DOCGEN_LLM_CONCURRENCY"] = "garbage"
            self.assertEqual(resolve_docgen_llm_concurrency(), 8)

    # ------------------------------------------------------------------
    # Cross-context toggles (bool flags through the same registry)
    # ------------------------------------------------------------------
    def test_cross_context_toggles_default_on(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            self.assertIs(resolve_docgen_spec_context_enabled(), True)
            self.assertIs(resolve_docgen_db_context_enabled(), True)

    def test_cross_context_toggles_env_off(self):
        # Regression: DOCGEN_DB_CONTEXT_ENABLED=false deployments must NOT
        # flip to on when the flag moves into the registry — a plain float
        # parse would read "false" as unset and fall back to the default.
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            for off in ("false", "0", "no", "off"):
                os.environ["DOCGEN_SPEC_CONTEXT_ENABLED"] = off
                os.environ["DOCGEN_DB_CONTEXT_ENABLED"] = off
                self.assertIs(resolve_docgen_spec_context_enabled(), False)
                self.assertIs(resolve_docgen_db_context_enabled(), False)

    def test_cross_context_toggles_env_legacy_truthy(self):
        # Legacy env semantics: anything except the explicit off-words is ON
        # (including an empty value and unrecognised garbage).
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            for on in ("true", "1", "yes", "garbage", ""):
                os.environ["DOCGEN_SPEC_CONTEXT_ENABLED"] = on
                self.assertIs(resolve_docgen_spec_context_enabled(), True)

    def test_cross_context_toggle_admin_beats_env(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with patch("api.config.settings.get_setting") as mock_get, _EnvGuard(env_vars, []):
            def _get(key, *a, **kw):
                if key == "timeouts.docgen_spec_context_enabled":
                    return "0"
                return None
            mock_get.side_effect = _get
            os.environ["DOCGEN_SPEC_CONTEXT_ENABLED"] = "1"
            self.assertIs(resolve_docgen_spec_context_enabled(), False)

    def test_cross_context_toggle_admin_word_forms(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with patch("api.config.settings.get_setting") as mock_get, _EnvGuard(env_vars, []):
            state = {"v": None}

            def _get(key, *a, **kw):
                if key == "timeouts.docgen_db_context_enabled":
                    return state["v"]
                return None
            mock_get.side_effect = _get
            for off in ("0", "false", "no", "off"):
                state["v"] = off
                self.assertIs(resolve_docgen_db_context_enabled(), False)
            for on in ("1", "true", "yes", "on"):
                state["v"] = on
                self.assertIs(resolve_docgen_db_context_enabled(), True)

    def test_cross_context_toggle_admin_typo_falls_to_env(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with patch("api.config.settings.get_setting") as mock_get, _EnvGuard(env_vars, []):
            def _get(key, *a, **kw):
                if key == "timeouts.docgen_spec_context_enabled":
                    return "garbage"
                return None
            mock_get.side_effect = _get
            os.environ["DOCGEN_SPEC_CONTEXT_ENABLED"] = "false"
            self.assertIs(resolve_docgen_spec_context_enabled(), False)

    def test_resolve_timeout_bool_unknown_key_is_false(self):
        self.assertIs(resolve_timeout_bool("does_not_exist"), False)

    def test_docgen_drain_explicit_override_below_floor_is_clamped(self):
        # An explicit DOCGEN_INDEXING_DRAIN_SECONDS below the drain floor (5)
        # is clamped up to the floor.
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_INDEXING_DRAIN_SECONDS"] = "1"
            self.assertEqual(resolve_docgen_indexing_drain_seconds(), 5.0)

    # ------------------------------------------------------------------
    # expert_stream (issue #9): wall-clock budget of one detached ask turn
    # ------------------------------------------------------------------
    def test_expert_stream_default_is_generous(self):
        # Multi-minute answers are the norm for the detached expert turn —
        # the default must stay at 1800s (30 min), not a per-request timeout.
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            self.assertEqual(resolve_expert_stream_timeout(), 1800.0)

    def test_expert_stream_env_override(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["EXPERT_STREAM_TIMEOUT_SECONDS"] = "600"
            self.assertEqual(resolve_expert_stream_timeout(), 600.0)

    def test_expert_stream_floor_sixty_seconds(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["EXPERT_STREAM_TIMEOUT_SECONDS"] = "1"
            self.assertEqual(resolve_expert_stream_timeout(), 60.0)

    # ------------------------------------------------------------------
    # sync_timeout_env exports admin-store overrides to env vars
    # ------------------------------------------------------------------
    def test_sync_timeout_env_exports_admin_store_to_env(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with patch("api.config.settings.get_setting") as mock_get, _EnvGuard(env_vars, []):
            def _get(key, *a, **kw):
                if key == "timeouts.llm_request":
                    return "5555"
                return None
            mock_get.side_effect = _get
            sync_timeout_env()
            self.assertEqual(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS"), "5555")

    def test_sync_timeout_env_skips_invalid_admin_store(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with patch("api.config.settings.get_setting") as mock_get, _EnvGuard(env_vars, []):
            def _get(key, *a, **kw):
                if key == "timeouts.llm_request":
                    return "garbage"
                return None
            mock_get.side_effect = _get
            sync_timeout_env()
            # Invalid admin value is skipped; env var left unset (None here).
            self.assertIsNone(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS"))

    # ------------------------------------------------------------------
    # DB RE docgen budgets (counts, not seconds; admin > env > default)
    # ------------------------------------------------------------------
    def test_db_docgen_budget_defaults(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            self.assertEqual(resolve_db_docgen_enrich_batch(), 40)
            self.assertEqual(resolve_db_docgen_max_subpages(), 200)
            self.assertEqual(resolve_db_docgen_max_descriptions(), 250)
            self.assertEqual(resolve_db_fk_evidence_tables(), 300)
            self.assertEqual(resolve_db_source_objects(), 100)

    def test_db_docgen_budget_env_override(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DB_DOCGEN_ENRICH_BATCH"] = "7"
            self.assertEqual(resolve_db_docgen_enrich_batch(), 7)
            os.environ["DB_DOCGEN_MAX_SUBPAGES"] = "3"
            self.assertEqual(resolve_db_docgen_max_subpages(), 3)
            os.environ["DB_FK_EVIDENCE_TABLES"] = "12"
            self.assertEqual(resolve_db_fk_evidence_tables(), 12)

    def test_db_docgen_budget_floors(self):
        # A typo can't shrink a batch below its floor or kill all subpages.
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DB_DOCGEN_ENRICH_BATCH"] = "1"
            self.assertEqual(resolve_db_docgen_enrich_batch(), 5)
            os.environ["DB_DOCGEN_MAX_SUBPAGES"] = "0"
            self.assertEqual(resolve_db_docgen_max_subpages(), 1)
            os.environ["DB_DOCGEN_MAX_DESCRIPTIONS"] = "-3"
            self.assertEqual(resolve_db_docgen_max_descriptions(), 250)

    def test_db_connect_check_wrapper(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            self.assertEqual(resolve_db_connect_check_timeout(), 180.0)

    # ------------------------------------------------------------------
    # resolved view for the admin panel
    # ------------------------------------------------------------------
    def test_resolved_view_has_every_key(self):
        view = get_timeout_resolved_view()
        self.assertEqual(set(view.keys()), {k.key for k in TIMEOUT_KEYS})
        for k in TIMEOUT_KEYS:
            entry = view[k.key]
            self.assertIn("value", entry)
            self.assertIn("default", entry)
            self.assertIn("floor", entry)
            self.assertIn("env_var", entry)
            self.assertEqual(entry["env_var"], k.env_var)
            self.assertEqual(entry["group"], k.group)
            self.assertEqual(entry["unit"], k.unit)


if __name__ == "__main__":
    unittest.main()
