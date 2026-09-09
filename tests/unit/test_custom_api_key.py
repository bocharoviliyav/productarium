"""Tests for custom (non-placeholder) API-key handling in the LLM stack.

Covers:
- ``api.config.settings._sanitize_api_key``: UUID / quoted / Bearer-prefixed /
  whitespace-wrapped keys, placeholders, empty / None.
- ``api.llm.client`` key + endpoint policy: ``is_no_auth_placeholder``,
  ``is_local_endpoint``, ``resolve_api_key``.
- ``api.config.abstraction.sync_runtime_settings``: admin-configured
  ``models.<task>.*`` values are exported to the canonical env vars.
- ``api.config.timeout``: llm_request / llm_retry_max_time defaults, env
  overrides, and floors.
- ``api.docgen.codebase._raise_if_all_sections_unavailable``: an
  all-placeholder generation must raise (not commit a fake "success").
"""

import os
import unittest
from unittest.mock import patch

from api.config.settings import _sanitize_api_key
from api.llm.client import (
    is_local_endpoint,
    is_no_auth_placeholder,
    resolve_api_key,
)
from api.config.abstraction import sync_runtime_settings


class TestCustomAPIKeyHandling(unittest.TestCase):
    def test_sanitize_api_key_formats(self):
        # UUID format key
        uuid_key = "550e8400-e29b-41d4-a716-446655440000"
        self.assertEqual(_sanitize_api_key(uuid_key), uuid_key)

        # Quoted UUID key
        quoted_key = '"550e8400-e29b-41d4-a716-446655440000"'
        self.assertEqual(_sanitize_api_key(quoted_key), uuid_key)

        # Key with leading Bearer prefix
        bearer_key = "Bearer 550e8400-e29b-41d4-a716-446655440000"
        self.assertEqual(_sanitize_api_key(bearer_key), uuid_key)

        # Key with whitespace
        space_key = "  550e8400-e29b-41d4-a716-446655440000 \n"
        self.assertEqual(_sanitize_api_key(space_key), uuid_key)

        # Empty / None / placeholder
        self.assertIsNone(_sanitize_api_key(None))
        self.assertEqual(_sanitize_api_key(""), "")
        self.assertEqual(_sanitize_api_key("not-needed"), "not-needed")

    def test_llm_client_resolve_custom_key(self):
        uuid_key = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        # Check resolution
        self.assertEqual(resolve_api_key(uuid_key), uuid_key)
        # Check placeholder flag: ONLY the explicit placeholder values count.
        # None / "" are NOT a no-auth signal (a missing key on a remote
        # endpoint must raise, not silently downgrade to no-auth).
        self.assertFalse(is_no_auth_placeholder(uuid_key))
        self.assertTrue(is_no_auth_placeholder("not-needed"))
        self.assertFalse(is_no_auth_placeholder(None))
        self.assertTrue(is_no_auth_placeholder("not_needed"))
        self.assertFalse(is_no_auth_placeholder(""))


class TestResolveCredentialsContract(unittest.TestCase):
    """``_resolve_credentials``: keyless REMOTE endpoints need an explicit
    no-auth signal (the ``api_key`` argument or the ``LOCAL_OPENAI_API_KEY``
    env var). The module-level config default (``api.config
    .LOCAL_OPENAI_API_KEY``) only authorizes LOCAL endpoints — a missing key
    on a remote endpoint raises ``ValueError`` at build time so the
    misconfiguration is loud.
    """

    REMOTE = "https://ai-gateway.company.com/v1"
    LOCAL = "http://localhost:1234/v1"
    UUID_KEY = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def setUp(self):
        from api.llm.client import _resolve_credentials

        self._resolve = _resolve_credentials
        # Pin both key layers: pop the env overrides and patch the module
        # config default to the placeholder (its normal value).
        self._saved = {}
        for key in ("LOCAL_OPENAI_API_KEY", "LOCAL_OPENAI_BASE_URL"):
            self._saved[key] = os.environ.pop(key, None)
        patcher = patch("api.config.LOCAL_OPENAI_API_KEY", "not-needed")
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_remote_without_key_raises(self):
        with self.assertRaises(ValueError):
            self._resolve(self.REMOTE, None)

    def test_remote_explicit_placeholder_argument_allowed(self):
        from api.llm.client import NO_AUTH_PLACEHOLDER

        self.assertEqual(
            self._resolve(self.REMOTE, "not-needed"), NO_AUTH_PLACEHOLDER
        )

    def test_remote_env_placeholder_allowed(self):
        from api.llm.client import NO_AUTH_PLACEHOLDER

        os.environ["LOCAL_OPENAI_API_KEY"] = "not-needed"
        self.assertEqual(self._resolve(self.REMOTE, None), NO_AUTH_PLACEHOLDER)

    def test_remote_real_key_wins(self):
        self.assertEqual(self._resolve(self.REMOTE, self.UUID_KEY), self.UUID_KEY)

    def test_local_endpoint_keyless_allowed(self):
        from api.llm.client import NO_AUTH_PLACEHOLDER

        self.assertEqual(self._resolve(self.LOCAL, None), NO_AUTH_PLACEHOLDER)

    def test_llm_client_local_endpoint_detection(self):
        self.assertTrue(is_local_endpoint("http://localhost:8080/v1"))
        self.assertTrue(is_local_endpoint("http://host.docker.internal:8080/v1"))
        self.assertFalse(is_local_endpoint("https://api.openai.com/v1"))
        self.assertFalse(is_local_endpoint(None))
        self.assertFalse(is_local_endpoint(""))


class TestDropAuthHeaderHook(unittest.TestCase):
    """The Authorization-dropping request hook must be an async callable.

    httpx (0.28) awaits every event hook on ``AsyncClient``
    (``await hook(request)``). A sync hook returns ``None`` and every request
    raises ``TypeError: object NoneType can't be used in 'await' expression``,
    surfacing as a misleading openai ``APIConnectionError`` ("Connection
    error") on EVERY LLM call to a no-auth endpoint — the default LM Studio
    configuration. Regression tests for the live smoke finding.
    """

    def test_hook_is_async_and_removes_header(self):
        import asyncio

        import httpx

        from api.llm.client import _drop_auth_header, build_http_async_client

        # The hook must be awaitable when called with a request.
        request = httpx.Request(
            "POST", "http://localhost:1234/v1/chat/completions",
            headers={"Authorization": "Bearer not-needed"},
        )
        asyncio.run(_drop_auth_header(request))  # must not raise TypeError
        self.assertNotIn("authorization", request.headers)

        client = build_http_async_client(strip_auth=True)
        self.assertIn(_drop_auth_header, client.event_hooks["request"])

    def test_hook_survives_a_real_request_cycle(self):
        """End-to-end over the request hook: a request through a client with
        the hook reaches the transport (connection refused here, NOT a
        TypeError) and the header is stripped before send."""
        import asyncio

        import httpx

        from api.llm.client import _drop_auth_header

        seen: dict = {}

        async def _capture_hook(req: httpx.Request) -> None:
            seen["auth"] = req.headers.get("authorization")

        async def _run():
            client = httpx.AsyncClient(
                event_hooks={"request": [_drop_auth_header, _capture_hook]},
                timeout=5.0,
            )
            try:
                await client.post(
                    "http://localhost:9/v1/chat/completions", json={"x": 1}
                )
            except httpx.ConnectError:
                pass  # expected: nothing listens on port 9
            finally:
                await client.aclose()

        asyncio.run(_run())
        # The capture hook ran after the drop hook: header already gone
        # (None), and the request pipeline did not raise TypeError.
        self.assertIsNone(seen.get("auth"))

    def test_build_chat_model_stream_surfaces_connection_error_not_typeerror(self):
        """Streaming the built model against a dead port must raise a
        connection-type error (server absent), never the hook TypeError."""
        import asyncio

        from api.llm.client import build_chat_model

        async def _run():
            model = build_chat_model(
                model="qwen/qwen3.6-27b",
                base_url="http://localhost:9/v1",
                api_key="not-needed",
            )
            from langchain_core.messages import HumanMessage

            parts = []
            async for chunk in model.astream([HumanMessage(content="hi")]):
                parts.append(chunk)
            return parts

        with self.assertRaises(Exception) as ctx:
            asyncio.run(_run())
        # The failure must be a connection error, not the NoneType TypeError.
        self.assertNotIsInstance(ctx.exception, TypeError)
        self.assertIn(
            type(ctx.exception).__name__,
            {"APIConnectionError", "ConnectError", "ConnectionError"},
        )


class TestSyncRuntimeSettings(unittest.TestCase):
    @patch("api.config.settings.get_model_for_task")
    def test_exports_custom_keys(self, mock_get_model):
        custom_key = "custom-uuid-key-1234"
        custom_url = "https://ai-proxy.company.com/v1"

        mock_get_model.return_value = {
            "provider": "openai_compatible",
            "model": "qwen3.5:35b",
            "base_url": custom_url,
            "api_key": custom_key,
        }

        sync_runtime_settings()

        self.assertEqual(os.environ.get("LOCAL_OPENAI_BASE_URL"), custom_url)
        self.assertEqual(os.environ.get("LOCAL_OPENAI_API_KEY"), custom_key)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), custom_key)


class TestLongRunningTimeouts(unittest.TestCase):
    """The generation path can run 20-30 min on a local model. These
    env-driven helpers raise the SDK default ceilings so long calls are not
    prematurely aborted."""

    def setUp(self):
        # Snapshot + clear the relevant env vars so each test is deterministic.
        # NOTE: these env vars are the FALLBACK layer. The admin settings store
        # (timeouts.<key>) has higher precedence; these tests assume the store
        # is empty / unset, which is the case in a clean test DB.
        self._saved = {}
        for k in (
            "LLM_REQUEST_TIMEOUT_SECONDS",
            "LLM_RETRY_MAX_TIME_SECONDS",
            "DOCGEN_INDEXING_DRAIN_SECONDS",
        ):
            self._saved[k] = os.environ.pop(k, None)
        # Ensure the admin store does not leak a stored override into these
        # default-value assertions.
        from api.config.settings import list_settings, set_setting
        self._stored_timeouts = [r["key"] for r in list_settings(prefix="timeouts.")]
        for key in self._stored_timeouts:
            set_setting(key, "", encrypt=False)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Restore the admin-store overrides we cleared (best-effort).
        from api.config.settings import set_setting
        for key in self._stored_timeouts:
            set_setting(key, "", encrypt=False)

    def test_request_timeout_default_and_override(self):
        from api.config.timeout import resolve_llm_request_timeout

        # Default is 3600s (1 h) to accommodate long generation on large repos.
        self.assertEqual(resolve_llm_request_timeout(), 3600.0)
        os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "4800"
        self.assertEqual(resolve_llm_request_timeout(), 4800.0)
        # Below the 60s floor is clamped, so a typo can't make calls fail
        # instantly on a legitimately slow model.
        os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "5"
        self.assertEqual(resolve_llm_request_timeout(), 60.0)

    def test_retry_max_time_default_and_override(self):
        from api.config.timeout import resolve_llm_retry_max_time

        # Default 900s lets a rate-limited embedding call actually complete
        # its backoff instead of aborting after 5s.
        self.assertEqual(resolve_llm_retry_max_time(), 900.0)
        os.environ["LLM_RETRY_MAX_TIME_SECONDS"] = "1200"
        self.assertEqual(resolve_llm_retry_max_time(), 1200.0)
        os.environ["LLM_RETRY_MAX_TIME_SECONDS"] = "1"
        self.assertEqual(resolve_llm_retry_max_time(), 30.0)

    def test_docgen_drain_env_override(self):
        from api.config.timeout import resolve_docgen_indexing_drain_seconds

        # Explicit DOCGEN_INDEXING_DRAIN_SECONDS override wins over default.
        os.environ["DOCGEN_INDEXING_DRAIN_SECONDS"] = "60"
        self.assertEqual(resolve_docgen_indexing_drain_seconds(), 60.0)


class TestAllPlaceholderDetection(unittest.TestCase):
    """A total generation failure (every section is the unavailable
    placeholder) must raise so the job is marked failed, not committed as a
    placeholder-filled "success"."""

    def test_all_placeholder_raises(self):
        import api.docgen.codebase as adg

        placeholder = adg._SECTION_UNAVAILABLE_PLACEHOLDER
        with self.assertRaises(ValueError):
            adg._raise_if_all_sections_unavailable({"a": placeholder, "b": placeholder})

    def test_mixed_content_does_not_raise(self):
        import api.docgen.codebase as adg

        placeholder = adg._SECTION_UNAVAILABLE_PLACEHOLDER
        # At least one real section -> NOT a total failure; must not raise.
        adg._raise_if_all_sections_unavailable(
            {"a": placeholder, "b": "# Real content\n\n..."}
        )

    def test_empty_does_not_raise(self):
        import api.docgen.codebase as adg

        # No sections (e.g. a non-codebase artifact path) -> nothing to flag.
        adg._raise_if_all_sections_unavailable({})


if __name__ == "__main__":
    unittest.main()
