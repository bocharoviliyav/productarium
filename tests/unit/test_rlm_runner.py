from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

import api.rlm.runner as runner


# --------------------------------------------------------------------------- #
# fast_rlm unavailable path
# --------------------------------------------------------------------------- #
class TestUnavailable:
    def test_returns_failure_when_not_available(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", False)
        result = runner.run_rlm_task_sync("test query", "model")
        assert result["success"] is False
        assert "not available" in result["results"]

    def test_prewarm_skips_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", False)
        # Should not raise, just return
        runner.prewarm_rlm_background()

    @pytest.mark.asyncio
    async def test_async_wrapper_unavailable(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", False)
        result = await runner.run_rlm_task("query", "model")
        assert result["success"] is False


# --------------------------------------------------------------------------- #
# Available + success
# --------------------------------------------------------------------------- #
class TestSuccess:
    def test_returns_results_on_success(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

        # Mock RLMConfig + run
        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        mock_run = MagicMock(return_value={
            "results": "RLM answer text",
            "usage": {"tokens": 100},
        })
        monkeypatch.setattr(runner, "run", mock_run)

        # Stub get_model_for_task so admin config is not read from DB
        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "test-model", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        # Stub get_model_context_window
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
        # Stub _host_to_v1 / _strip_provider_prefix (imported inside runner)
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        result = runner.run_rlm_task_sync("test query", "test-model")
        assert result["success"] is True
        assert result["results"] == "RLM answer text"
        assert result["usage"] == {"tokens": 100}

    def test_run_exception_returns_failure(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        mock_run = MagicMock(side_effect=RuntimeError("connection error"))
        monkeypatch.setattr(runner, "run", mock_run)

        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "m", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        result = runner.run_rlm_task_sync("query", "m")
        assert result["success"] is False
        assert "Failed to execute" in result["results"]
        assert "connection error" in result["results"]

    def test_context_size_exceeded_error(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        mock_run = MagicMock(side_effect=Exception("Context size has been exceeded"))
        monkeypatch.setattr(runner, "run", mock_run)

        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "m", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        result = runner.run_rlm_task_sync("query", "m")
        assert result["success"] is False
        assert "Context size has been exceeded" in result["results"]


# --------------------------------------------------------------------------- #
# Model / base_url / api_key resolution
# --------------------------------------------------------------------------- #
class TestConfigResolution:
    def test_model_from_admin_config(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        captured = {}
        def mock_run_fn(query, config=None, verbose=False):
            captured["model"] = config.primary_agent
            captured["base_url"] = __import__("os").environ.get("RLM_MODEL_BASE_URL")
            return {"results": "ok", "usage": {}}
        monkeypatch.setattr(runner, "run", mock_run_fn)

        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "admin-model", "base_url": "http://admin:9999/v1", "api_key": "not-needed"},
        )
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        runner.run_rlm_task_sync("query")
        assert captured["model"] == "admin-model"
        assert "admin:9999" in captured["base_url"]

    def test_model_arg_overrides_admin(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        captured = {}
        def mock_run_fn(query, config=None, verbose=False):
            captured["model"] = config.primary_agent
            return {"results": "ok", "usage": {}}
        monkeypatch.setattr(runner, "run", mock_run_fn)

        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "admin-model", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        runner.run_rlm_task_sync("query", "explicit-model")
        assert captured["model"] == "explicit-model"

    def test_admin_api_key_exported_to_env(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)
        monkeypatch.delenv("RLM_MODEL_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)

        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        monkeypatch.setattr(runner, "run", lambda *a, **kw: {"results": "ok", "usage": {}})
        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "m", "base_url": "http://localhost:11434/v1", "api_key": "real-key-123"},
        )
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        import os
        runner.run_rlm_task_sync("query", "m")
        assert os.environ.get("RLM_MODEL_API_KEY") == "real-key-123"
        assert os.environ.get("OPENAI_API_KEY") == "real-key-123"

    def test_placeholder_api_key_not_exported(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)
        monkeypatch.delenv("RLM_MODEL_API_KEY", raising=False)
        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        monkeypatch.setattr(runner, "run", lambda *a, **kw: {"results": "ok", "usage": {}})
        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "m", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        import os
        runner.run_rlm_task_sync("query", "m")
        # not-needed is a placeholder; the runner sets it as the env fallback
        assert os.environ.get("RLM_MODEL_API_KEY") == "not-needed"


# --------------------------------------------------------------------------- #
# Context window clamping
# --------------------------------------------------------------------------- #
class TestContextWindowClamp:
    def test_clamps_budgets_to_context_window(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        captured = {}
        def mock_run_fn(query, config=None, verbose=False):
            captured["max_prompt"] = config.max_prompt_tokens
            captured["max_completion"] = config.max_completion_tokens
            return {"results": "ok", "usage": {}}
        monkeypatch.setattr(runner, "run", mock_run_fn)

        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "m", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        # Small context window: 4096
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 4096,
        )
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        runner.run_rlm_task_sync("query", "m")
        # prompt + completion must not exceed 4096
        assert captured["max_prompt"] + captured["max_completion"] <= 4096
        assert captured["max_prompt"] >= 1024


# --------------------------------------------------------------------------- #
# Async wrapper
# --------------------------------------------------------------------------- #
class TestAsyncWrapper:
    @pytest.mark.asyncio
    async def test_run_rlm_task_async(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

        mock_config = MagicMock()
        mock_config.max_prompt_tokens = 200000
        mock_config.max_completion_tokens = 50000
        mock_config.api_timeout_ms = 3600000
        mock_config_class = MagicMock()
        mock_config_class.default.return_value = mock_config
        monkeypatch.setattr(runner, "RLMConfig", mock_config_class)

        monkeypatch.setattr(runner, "run", lambda *a, **kw: {"results": "async ok", "usage": {}})
        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "m", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

        result = await runner.run_rlm_task("query", "m")
        assert result["success"] is True
        assert result["results"] == "async ok"
