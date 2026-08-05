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


if __name__ == "__main__":
    unittest.main()
