import os
import unittest
from unittest.mock import patch, MagicMock

from api.config.settings import _sanitize_api_key
from api.clients.openai_client import OpenAIClient
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

    def test_openai_client_resolve_custom_key(self):
        uuid_key = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        # Check resolution
        resolved = OpenAIClient._resolve_api_key(uuid_key)
        self.assertEqual(resolved, uuid_key)

        # Check placeholder flag
        self.assertFalse(OpenAIClient._is_no_auth_placeholder(uuid_key))
        self.assertTrue(OpenAIClient._is_no_auth_placeholder("not-needed"))
        self.assertTrue(OpenAIClient._is_no_auth_placeholder(None))

    @patch("api.clients.openai_client.OpenAI")
    def test_openai_client_initialization_with_custom_key(self, mock_openai):
        uuid_key = "12345678-abcd-1234-abcd-123456789abc"
        base_url = "http://my-gateway.example.com/v1"

        client = OpenAIClient(api_key=uuid_key, base_url=base_url)

        self.assertTrue(client._real_api_key)
        self.assertEqual(client._api_key, uuid_key)
        self.assertEqual(client.base_url, base_url)

        # Verify OpenAI SDK was initialized with the custom key and base_url
        mock_openai.assert_called_once()
        _, kwargs = mock_openai.call_args
        self.assertEqual(kwargs["api_key"], uuid_key)
        self.assertEqual(kwargs["base_url"], base_url)

    @patch("api.config.settings.get_model_for_task")
    def test_sync_runtime_settings_exports_custom_keys(self, mock_get_model):
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

    def test_cognee_rate_limiter_per_loop_primitives(self):
        import asyncio
        from api.cognee import _cognee_rate_limiter

        async def _test():
            sem, lock = _cognee_rate_limiter._get_loop_primitives(2)
            self.assertIsNotNone(sem)
            self.assertIsNotNone(lock)
            self.assertEqual(sem._value, 2)

        asyncio.run(_test())


class TestLongRunningTimeouts(unittest.TestCase):
    """The generation + cognify path can run 20-30 min on a local model.
    These env-driven helpers raise the SDK default ceilings so long calls
    are not prematurely aborted."""

    def setUp(self):
        # Snapshot + clear the relevant env vars so each test is deterministic.
        # NOTE: these env vars are the FALLBACK layer. The admin settings store
        # (timeouts.<key>) has higher precedence; these tests assume the store
        # is empty / unset, which is the case in a clean test DB.
        self._saved = {}
        for k in (
            "LLM_REQUEST_TIMEOUT_SECONDS",
            "LLM_RETRY_MAX_TIME_SECONDS",
            "COGNEE_GRAPH_EXTRACTION_TIMEOUT",
            "COGNEE_COGNIFY_TIMEOUT",
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
        from api.clients.openai_client import _resolve_request_timeout

        # Default is 3600s (1 h) to accommodate long generation/cognify on
        # large repos.
        self.assertEqual(_resolve_request_timeout(), 3600.0)
        os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "4800"
        self.assertEqual(_resolve_request_timeout(), 4800.0)
        # Below the 60s floor is clamped, so a typo can't make calls fail
        # instantly on a legitimately slow model.
        os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "5"
        self.assertEqual(_resolve_request_timeout(), 60.0)

    def test_retry_max_time_default_and_override(self):
        from api.clients.openai_client import _resolve_retry_max_time

        # Default 900s lets a rate-limited cognee/embedding call actually
        # complete its backoff instead of aborting after 5s.
        self.assertEqual(_resolve_retry_max_time(), 900.0)
        os.environ["LLM_RETRY_MAX_TIME_SECONDS"] = "1200"
        self.assertEqual(_resolve_retry_max_time(), 1200.0)
        os.environ["LLM_RETRY_MAX_TIME_SECONDS"] = "1"
        self.assertEqual(_resolve_retry_max_time(), 30.0)

    def test_graph_extraction_timeout_helpers(self):
        from api.cognee import (
            _resolve_graph_extraction_timeout,
            _resolve_cognify_timeout,
        )

        # Graph extraction default 1800s (30 min) per chunk so a slow corporate
        # gateway doing markdown_json_mode extraction is not aborted per chunk;
        # cognify default 7200s overall.
        self.assertEqual(_resolve_graph_extraction_timeout(), 1800.0)
        self.assertEqual(_resolve_cognify_timeout(), 7200.0)

        os.environ["COGNEE_GRAPH_EXTRACTION_TIMEOUT"] = "2400"
        self.assertEqual(_resolve_graph_extraction_timeout(), 2400.0)
        os.environ["COGNEE_COGNIFY_TIMEOUT"] = "14400"
        self.assertEqual(_resolve_cognify_timeout(), 14400.0)

        # Invalid values fall back to the defaults, not crash.
        os.environ["COGNEE_GRAPH_EXTRACTION_TIMEOUT"] = "not-a-number"
        self.assertEqual(_resolve_graph_extraction_timeout(), 1800.0)
        os.environ["COGNEE_COGNIFY_TIMEOUT"] = ""
        self.assertEqual(_resolve_cognify_timeout(), 7200.0)

    def test_docgen_drain_derives_from_cognify(self):
        # The docgen indexing-drain timeout defaults to the resolved cognify
        # timeout so a leftover cognify task gets the full cognify budget
        # instead of being cancelled at a fixed 30s (which previously dropped
        # the connection mid-graph-build).
        from api.cognee import _resolve_cognify_timeout
        from api.config.timeout import resolve_docgen_indexing_drain_seconds

        self.assertEqual(resolve_docgen_indexing_drain_seconds(), _resolve_cognify_timeout())
        os.environ["COGNEE_COGNIFY_TIMEOUT"] = "10800"
        self.assertEqual(resolve_docgen_indexing_drain_seconds(), 10800.0)
        # An explicit DOCGEN_INDEXING_DRAIN_SECONDS override still wins.
        os.environ["DOCGEN_INDEXING_DRAIN_SECONDS"] = "60"
        self.assertEqual(resolve_docgen_indexing_drain_seconds(), 60.0)

    @patch("api.clients.openai_client.OpenAI")
    def test_client_uses_configured_timeout(self, mock_openai):
        import httpx

        os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "1234"
        OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        mock_openai.assert_called_once()
        _, kwargs = mock_openai.call_args
        http_client = kwargs["http_client"]
        self.assertIsInstance(http_client, httpx.Client)
        # httpx stores the configured timeout as a Timeout object.
        self.assertEqual(http_client.timeout.read, 1234.0)


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
