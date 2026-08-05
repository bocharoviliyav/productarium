import os
import unittest
from unittest.mock import patch, MagicMock

from api.settings_store import _sanitize_api_key
from api.openai_client import OpenAIClient
from api.config_abstraction import sync_runtime_settings


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

    @patch("api.openai_client.OpenAI")
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

    @patch("api.settings_store.get_model_for_task")
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
        from api.cognee_manager import _cognee_rate_limiter

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
        self._saved = {}
        for k in (
            "LLM_REQUEST_TIMEOUT_SECONDS",
            "LLM_RETRY_MAX_TIME_SECONDS",
            "COGNEE_GRAPH_EXTRACTION_TIMEOUT",
            "COGNEE_COGNIFY_TIMEOUT",
        ):
            self._saved[k] = os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_request_timeout_default_and_override(self):
        from api.openai_client import _resolve_request_timeout

        # Default is 1800s (30 min) to accommodate long generation/cognify.
        self.assertEqual(_resolve_request_timeout(), 1800.0)
        os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "2400"
        self.assertEqual(_resolve_request_timeout(), 2400.0)
        # Below the 60s floor is clamped, so a typo can't make calls fail
        # instantly on a legitimately slow model.
        os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "5"
        self.assertEqual(_resolve_request_timeout(), 60.0)

    def test_retry_max_time_default_and_override(self):
        from api.openai_client import _resolve_retry_max_time

        # Default 600s (10 min) lets a rate-limited cognee/embedding call
        # actually complete its backoff instead of aborting after 5s.
        self.assertEqual(_resolve_retry_max_time(), 600.0)
        os.environ["LLM_RETRY_MAX_TIME_SECONDS"] = "900"
        self.assertEqual(_resolve_retry_max_time(), 900.0)
        os.environ["LLM_RETRY_MAX_TIME_SECONDS"] = "1"
        self.assertEqual(_resolve_retry_max_time(), 30.0)

    def test_graph_extraction_timeout_helpers(self):
        from api.cognee_manager import (
            _resolve_graph_extraction_timeout,
            _resolve_cognify_timeout,
        )

        # Graph extraction default 180s per chunk; cognify default 1800s overall.
        self.assertEqual(_resolve_graph_extraction_timeout(), 180.0)
        self.assertEqual(_resolve_cognify_timeout(), 1800.0)

        os.environ["COGNEE_GRAPH_EXTRACTION_TIMEOUT"] = "600"
        self.assertEqual(_resolve_graph_extraction_timeout(), 600.0)
        os.environ["COGNEE_COGNIFY_TIMEOUT"] = "3600"
        self.assertEqual(_resolve_cognify_timeout(), 3600.0)

        # Invalid values fall back to the defaults, not crash.
        os.environ["COGNEE_GRAPH_EXTRACTION_TIMEOUT"] = "not-a-number"
        self.assertEqual(_resolve_graph_extraction_timeout(), 180.0)
        os.environ["COGNEE_COGNIFY_TIMEOUT"] = ""
        self.assertEqual(_resolve_cognify_timeout(), 1800.0)

    @patch("api.openai_client.OpenAI")
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


if __name__ == "__main__":
    unittest.main()
