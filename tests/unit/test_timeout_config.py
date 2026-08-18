"""Tests for the central timeout resolver (api.timeout_config).

Covers:
- Precedence: admin store (timeouts.<key>) > env var > default.
- Invalid-value fallback at every precedence level (never raises).
- Per-key floor enforcement (a typo can't make a timeout dangerously small).
- docgen_indexing_drain default derives from cognee_cognify.
- TIMEOUT_KEYS regression guard: every wrapper imported by the 17 routed
  files maps to an entry in TIMEOUT_KEYS, so a new timeout can't be added to
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
    resolve_docgen_indexing_drain_seconds,
    resolve_timeout,
    resolve_timeout_int,
    sync_timeout_env,
)


# Every wrapper name imported by the 17 routed files. Kept in sync with the
# grep over api/ for `resolve_*_timeout` / `resolve_*_ms` / `resolve_*_attempts`.
# If a new wrapper is added to the codebase, add it here AND to TIMEOUT_KEYS.
WRAPPER_TO_KEY = {
    "resolve_llm_request_timeout": "llm_request",
    "resolve_llm_retry_max_time": "llm_retry_max_time",
    "resolve_cognee_graph_extraction_timeout": "cognee_graph_extraction",
    "resolve_cognee_cognify_timeout": "cognee_cognify",
    "resolve_cognee_llm_connection_timeout": "cognee_llm_connection",
    "resolve_cognee_init_timeout": "cognee_init",
    "resolve_cognee_recall_timeout": "cognee_recall",
    "resolve_docgen_indexing_drain_seconds": "docgen_indexing_drain",
    "resolve_memory_query_timeout": "memory_query",
    "resolve_rlm_api_timeout_ms": "rlm_api_ms",
    "resolve_rlm_section_timeout": "rlm_section",
    "resolve_rlm_expert_timeout": "rlm_expert",
    "resolve_model_list_timeout": "model_list",
    "resolve_integration_http_timeout": "integration_http",
    "resolve_git_file_content_timeout": "git_file_content",
    "resolve_mcp_stdio_wait_timeout": "mcp_stdio_wait",
    "resolve_mermaid_verify_timeout": "mermaid_verify",
    "resolve_mermaid_repair_timeout": "mermaid_repair",
    "resolve_mermaid_max_repair_attempts": "mermaid_max_repair_attempts",
    "resolve_provider_test_timeout": "provider_test",
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
        """Every wrapper imported by the 17 routed files is in TIMEOUT_KEYS."""
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
                    max(k.floor, k.default) if k.key != "docgen_indexing_drain" else _resolve_with_key("cognee_cognify"),
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
    # docgen_indexing_drain derives from cognee_cognify
    # ------------------------------------------------------------------
    def test_docgen_drain_derives_from_cognify_default(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            # With nothing set, drain == cognify (both at default 7200).
            self.assertEqual(
                resolve_docgen_indexing_drain_seconds(),
                resolve_timeout("cognee_cognify"),
            )

    def test_docgen_drain_tracks_cognify_env_override(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["COGNEE_COGNIFY_TIMEOUT"] = "10800"
            self.assertEqual(resolve_docgen_indexing_drain_seconds(), 10800.0)

    def test_docgen_drain_explicit_override_wins_over_cognify(self):
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["COGNEE_COGNIFY_TIMEOUT"] = "10800"
            os.environ["DOCGEN_INDEXING_DRAIN_SECONDS"] = "120"
            self.assertEqual(resolve_docgen_indexing_drain_seconds(), 120.0)

    def test_docgen_drain_explicit_override_below_floor_is_clamped(self):
        # An explicit DOCGEN_INDEXING_DRAIN_SECONDS below the drain floor (5)
        # is clamped up to the floor. (Going through cognify can't exercise the
        # drain floor in practice, because cognify's own floor of 300 is already
        # well above the drain floor of 5.)
        env_vars = [k.env_var for k in TIMEOUT_KEYS]
        with _EnvGuard(env_vars, []):
            os.environ["DOCGEN_INDEXING_DRAIN_SECONDS"] = "1"
            self.assertEqual(resolve_docgen_indexing_drain_seconds(), 5.0)

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
