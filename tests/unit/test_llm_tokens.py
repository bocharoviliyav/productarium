"""Unit tests for ``api.utils.llm_tokens`` (token / context-window math).

Covers:
- ``get_model_context_window``:
  * Env var override (``RLM_MODEL_CONTEXT_WINDOW``).
  * Admin setting override (``models.<task>.max_prompt_tokens``).
  * Cache hit (within TTL).
  * Live API query (OpenAI-compatible ``/v1/models`` with ``max_model_len``).
  * Live API query with various context keys (``context_window``, ``max_tokens``, etc.).
  * Fallback default (8192) on exception / non-200 / no matching model.
  * URL normalization (appends ``/v1``).
  * Authorization header when api_key is set.
- ``_count_tokens``:
  * Empty / None text.
  * Real tiktoken encoding.
  * Fallback path (len // 4) when tiktoken fails.
"""

from __future__ import annotations

import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.utils import llm_tokens
from api.utils.llm_tokens import (
    _CACHE_TTL_SECONDS,
    _MODEL_CTX_CACHE,
    _count_tokens,
    get_model_context_window,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the model context cache before each test."""
    _MODEL_CTX_CACHE.clear()
    yield
    _MODEL_CTX_CACHE.clear()


# ---------------------------------------------------------------------------
# get_model_context_window — env var overrides
# ---------------------------------------------------------------------------

class TestEnvVarOverride:
    def test_rlm_model_context_window_env(self, monkeypatch):
        monkeypatch.setenv("RLM_MODEL_CONTEXT_WINDOW", "32768")
        assert get_model_context_window() == 32768

    def test_invalid_env_ignored(self, monkeypatch):
        monkeypatch.setenv("RLM_MODEL_CONTEXT_WINDOW", "not-a-number")
        # Falls through to live API query / default
        # With no live API (requests not mocked), returns default 8192
        monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("x")))
        assert get_model_context_window() == 8192

    def test_zero_env_ignored(self, monkeypatch):
        monkeypatch.setenv("RLM_MODEL_CONTEXT_WINDOW", "0")
        monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("x")))
        assert get_model_context_window() == 8192

    def test_negative_env_ignored(self, monkeypatch):
        monkeypatch.setenv("RLM_MODEL_CONTEXT_WINDOW", "-100")
        monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("x")))
        assert get_model_context_window() == 8192


# ---------------------------------------------------------------------------
# get_model_context_window — admin setting override
# ---------------------------------------------------------------------------

class TestAdminSettingOverride:
    def test_admin_max_prompt_tokens(self, monkeypatch, isolated_db):
        from api.config import settings

        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        settings.set_setting("models.docgen.max_prompt_tokens", "16384", encrypt=False)
        assert get_model_context_window(task="docgen") == 16384

    def test_env_takes_precedence_over_admin_setting(self, monkeypatch, isolated_db):
        from api.config import settings

        monkeypatch.setenv("RLM_MODEL_CONTEXT_WINDOW", "99999")
        settings.set_setting("models.expert.max_prompt_tokens", "4096", encrypt=False)
        # Env var wins over admin setting
        assert get_model_context_window(task="expert") == 99999

    def test_invalid_admin_setting_ignored(self, monkeypatch, isolated_db):
        from api.config import settings

        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        settings.set_setting("models.docgen.max_prompt_tokens", "garbage", encrypt=False)
        monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("x")))
        # Falls through to default
        assert get_model_context_window(task="docgen") == 8192

    def test_zero_admin_setting_ignored(self, monkeypatch, isolated_db):
        from api.config import settings

        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        settings.set_setting("models.docgen.max_prompt_tokens", "0", encrypt=False)
        monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("x")))
        assert get_model_context_window(task="docgen") == 8192


# ---------------------------------------------------------------------------
# get_model_context_window — cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_hit(self, monkeypatch):
        # Pre-populate cache
        url = "http://localhost:8080/v1"
        model = "test-model"
        cache_key = (url.rstrip("/"), model)
        _MODEL_CTX_CACHE[cache_key] = (time.time(), 99999)
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        assert get_model_context_window(base_url=url, model_name=model) == 99999

    def test_cache_expired(self, monkeypatch):
        url = "http://localhost:8080/v1"
        model = "test-model"
        cache_key = (url.rstrip("/"), model)
        # Set cache entry with old timestamp
        _MODEL_CTX_CACHE[cache_key] = (time.time() - _CACHE_TTL_SECONDS - 1, 99999)
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        # Expired -> falls through to live API (mocked to fail) -> default
        monkeypatch.setattr("requests.get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("x")))
        assert get_model_context_window(base_url=url, model_name=model) == 8192


# ---------------------------------------------------------------------------
# get_model_context_window — live API query
# ---------------------------------------------------------------------------

class TestLiveApiQuery:
    def test_max_model_len_key(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "my-model", "max_model_len": 131072},
                ]
            },
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 131072

    def test_context_window_key(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "my-model", "context_window": 32768},
                ]
            },
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 32768

    def test_max_tokens_key(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "my-model", "max_tokens": 8192},
                ]
            },
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 8192

    def test_max_context_length_key(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "my-model", "max_context_length": 65536},
                ]
            },
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 65536

    def test_n_ctx_key(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "my-model", "n_ctx": 4096},
                ]
            },
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 4096

    def test_name_field_match(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"name": "my-model", "max_model_len": 16384},
                ]
            },
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 16384

    def test_non_200_returns_default(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(status_code=500, json=lambda: {})
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 8192

    def test_no_matching_model_returns_default(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {"data": [{"id": "other-model", "max_model_len": 9999}]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 8192

    def test_connection_error_returns_default(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        def boom(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("requests.get", boom)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 8192

    def test_result_cached_after_live_query(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        call_count = {"n": 0}

        def fake_get(*a, **kw):
            call_count["n"] += 1
            return types.SimpleNamespace(
                status_code=200,
                json=lambda: {"data": [{"id": "my-model", "max_model_len": 32768}]},
            )

        monkeypatch.setattr("requests.get", fake_get)
        url = "http://localhost:8080/v1"
        model = "my-model"
        result1 = get_model_context_window(base_url=url, model_name=model)
        assert result1 == 32768
        assert call_count["n"] == 1
        # Second call should hit cache
        result2 = get_model_context_window(base_url=url, model_name=model)
        assert result2 == 32768
        assert call_count["n"] == 1

    def test_url_normalization_appends_v1(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        captured = {}

        def fake_get(url, **kw):
            captured["url"] = url
            return types.SimpleNamespace(status_code=200, json=lambda: {"data": []})

        monkeypatch.setattr("requests.get", fake_get)
        get_model_context_window(base_url="http://localhost:8080", model_name="m")
        assert captured["url"] == "http://localhost:8080/v1/models"

    def test_authorization_header_sent(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        captured = {}

        def fake_get(url, **kw):
            captured["headers"] = kw.get("headers", {})
            return types.SimpleNamespace(status_code=200, json=lambda: {"data": []})

        monkeypatch.setattr("requests.get", fake_get)
        get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="m",
            api_key="sk-test-key",
        )
        assert captured["headers"]["Authorization"] == "Bearer sk-test-key"

    def test_no_auth_header_for_not_needed(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        captured = {}

        def fake_get(url, **kw):
            captured["headers"] = kw.get("headers", {})
            return types.SimpleNamespace(status_code=200, json=lambda: {"data": []})

        monkeypatch.setattr("requests.get", fake_get)
        get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="m",
            api_key="not-needed",
        )
        assert "Authorization" not in captured["headers"]

    def test_float_context_value(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)

        fake_resp = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "my-model", "max_model_len": 131072.0},
                ]
            },
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)
        result = get_model_context_window(
            base_url="http://localhost:8080/v1",
            model_name="my-model",
        )
        assert result == 131072

    def test_default_model_name(self, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_CONTEXT_WINDOW", raising=False)
        captured = {}

        def fake_get(url, **kw):
            captured["url"] = url
            return types.SimpleNamespace(status_code=200, json=lambda: {"data": []})

        monkeypatch.setattr("requests.get", fake_get)
        # No model_name -> should use default
        result = get_model_context_window()
        assert result == 8192


# ---------------------------------------------------------------------------
# _count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self):
        assert _count_tokens("") == 0

    def test_none(self):
        assert _count_tokens(None) == 0

    def test_simple_text(self):
        # tiktoken is available in the test env
        count = _count_tokens("hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        count = _count_tokens(text)
        assert count > 10

    def test_fallback_when_tiktoken_fails(self, monkeypatch):
        # Force tiktoken import to fail
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Fallback: len(text) // 4, minimum 1
        text = "hello world test"  # 16 chars -> 4 tokens
        count = _count_tokens(text)
        assert count == 4

    def test_fallback_minimum_one(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Very short text -> max(1, len // 4)
        count = _count_tokens("ab")  # 2 chars -> 0 -> max(1, 0) = 1
        assert count == 1
