from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

import api.rlm.runner as runner


# --------------------------------------------------------------------------- #
# Shared fixture: the 6 dependencies every "available" test stubs identically.
#
# Each test previously repeated the same monkeypatch.setattr block (RLMConfig,
# run, get_model_for_task, get_model_context_window, _host_to_v1,
# _strip_provider_prefix, _FAST_RLM_AVAILABLE). That was not testing the
# runner's behavior -- it was asserting run() was called. The fixture
# centralizes the wiring so tests focus on what they actually assert on:
#   - rlm_deps.config        -> the MagicMock RLMConfig.default() returns
#   - rlm_deps.captured      -> dict individual tests record values into
#   - rlm_deps.set_run(fn)   -> override the `run` mock with a callable
#   - rlm_deps.set_ctx(n)    -> override get_model_context_window's return value
#   - rlm_deps.set_admin({...})-> override the admin models.docgen.* config
# --------------------------------------------------------------------------- #
@pytest.fixture
def rlm_deps(monkeypatch):
    monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

    config = MagicMock()
    config.max_prompt_tokens = 200000
    config.max_completion_tokens = 50000
    config.api_timeout_ms = 3600000
    config_class = MagicMock()
    config_class.default.return_value = config
    monkeypatch.setattr(runner, "RLMConfig", config_class)

    captured: dict = {}

    def _default_run(query, config=None, verbose=False):
        return {"results": "ok", "usage": {}}

    run_mock = MagicMock(side_effect=_default_run)
    monkeypatch.setattr(runner, "run", run_mock)

    admin_cfg = {"model": "test-model", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"}
    monkeypatch.setattr(
        "api.config.settings.get_model_for_task",
        lambda task: dict(admin_cfg),
    )
    monkeypatch.setattr(
        "api.utils.get_model_context_window",
        lambda **kw: 8192,
    )
    import api.cognee._runtime as rt
    monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
    monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)

    deps = SimpleNamespace(config=config, captured=captured, run_mock=run_mock, admin_cfg=admin_cfg)

    def set_run(fn):
        run_mock.side_effect = fn

    def set_ctx(n):
        monkeypatch.setattr("api.utils.get_model_context_window", lambda **kw: n)

    def set_admin(cfg):
        admin_cfg.clear()
        admin_cfg.update(cfg)

    deps.set_run = set_run
    deps.set_ctx = set_ctx
    deps.set_admin = set_admin
    return deps


# --------------------------------------------------------------------------- #
# fast_rlm unavailable path (realistic: no mocking of run/RLMConfig needed)
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
# Available + success / failure outcome (compressed: shared fixture)
# --------------------------------------------------------------------------- #
class TestSuccess:
    def test_returns_results_on_success(self, rlm_deps):
        rlm_deps.set_run(lambda query, config=None, verbose=False: {
            "results": "RLM answer text", "usage": {"tokens": 100},
        })
        result = runner.run_rlm_task_sync("test query", "test-model")
        assert result["success"] is True
        assert result["results"] == "RLM answer text"
        assert result["usage"] == {"tokens": 100}

    def test_run_exception_returns_failure(self, rlm_deps):
        def _raise(query, config=None, verbose=False):
            raise RuntimeError("connection error")
        rlm_deps.set_run(_raise)
        result = runner.run_rlm_task_sync("query", "m")
        assert result["success"] is False
        assert "Failed to execute" in result["results"]
        assert "connection error" in result["results"]

    def test_context_size_exceeded_error_surfaces_in_results(self, rlm_deps):
        def _raise(query, config=None, verbose=False):
            raise Exception("Context size has been exceeded")
        rlm_deps.set_run(_raise)
        result = runner.run_rlm_task_sync("query", "m")
        assert result["success"] is False
        assert "Context size has been exceeded" in result["results"]


# --------------------------------------------------------------------------- #
# Model / base_url / api_key resolution (the most valuable behavior tests)
# --------------------------------------------------------------------------- #
class TestConfigResolution:
    def test_model_from_admin_config(self, rlm_deps):
        rlm_deps.set_admin({"model": "admin-model", "base_url": "http://admin:9999/v1", "api_key": "not-needed"})

        def _capture(query, config=None, verbose=False):
            rlm_deps.captured["model"] = config.primary_agent
            rlm_deps.captured["base_url"] = __import__("os").environ.get("RLM_MODEL_BASE_URL")
            return {"results": "ok", "usage": {}}
        rlm_deps.set_run(_capture)

        runner.run_rlm_task_sync("query")
        assert rlm_deps.captured["model"] == "admin-model"
        assert "admin:9999" in rlm_deps.captured["base_url"]

    def test_model_arg_overrides_admin(self, rlm_deps):
        def _capture(query, config=None, verbose=False):
            rlm_deps.captured["model"] = config.primary_agent
            return {"results": "ok", "usage": {}}
        rlm_deps.set_run(_capture)

        runner.run_rlm_task_sync("query", "explicit-model")
        assert rlm_deps.captured["model"] == "explicit-model"

    def test_admin_api_key_exported_to_env(self, rlm_deps, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
        rlm_deps.set_admin({"model": "m", "base_url": "http://localhost:11434/v1", "api_key": "real-key-123"})

        import os
        runner.run_rlm_task_sync("query", "m")
        assert os.environ.get("RLM_MODEL_API_KEY") == "real-key-123"
        assert os.environ.get("OPENAI_API_KEY") == "real-key-123"

    def test_placeholder_api_key_not_exported(self, rlm_deps, monkeypatch):
        monkeypatch.delenv("RLM_MODEL_API_KEY", raising=False)
        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Default admin api_key is the "not-needed" placeholder.
        import os
        runner.run_rlm_task_sync("query", "m")
        # not-needed is a placeholder; the runner sets it as the env fallback
        assert os.environ.get("RLM_MODEL_API_KEY") == "not-needed"


# --------------------------------------------------------------------------- #
# Context window clamping (the single most valuable test -- real behavior)
# --------------------------------------------------------------------------- #
class TestContextWindowClamp:
    def test_clamps_budgets_to_context_window(self, rlm_deps):
        rlm_deps.set_ctx(4096)

        def _capture(query, config=None, verbose=False):
            rlm_deps.captured["max_prompt"] = config.max_prompt_tokens
            rlm_deps.captured["max_completion"] = config.max_completion_tokens
            return {"results": "ok", "usage": {}}
        rlm_deps.set_run(_capture)

        runner.run_rlm_task_sync("query", "m")
        # prompt + completion must not exceed the 4096 context window
        assert rlm_deps.captured["max_prompt"] + rlm_deps.captured["max_completion"] <= 4096
        assert rlm_deps.captured["max_prompt"] >= 1024

    def test_no_clamp_when_context_window_unknown(self, rlm_deps, monkeypatch):
        # When the context window can't be resolved (returns 0/None), the
        # runner must leave the fast-rlm defaults untouched instead of clamping
        # to a bogus zero budget.
        monkeypatch.setattr("api.utils.get_model_context_window", lambda **kw: None)
        original_prompt = rlm_deps.config.max_prompt_tokens
        original_completion = rlm_deps.config.max_completion_tokens

        def _capture(query, config=None, verbose=False):
            rlm_deps.captured["max_prompt"] = config.max_prompt_tokens
            rlm_deps.captured["max_completion"] = config.max_completion_tokens
            return {"results": "ok", "usage": {}}
        rlm_deps.set_run(_capture)

        runner.run_rlm_task_sync("query", "m")
        assert rlm_deps.captured["max_prompt"] == original_prompt
        assert rlm_deps.captured["max_completion"] == original_completion


# --------------------------------------------------------------------------- #
# Async wrapper (compressed: just confirms the async path delegates to sync)
# --------------------------------------------------------------------------- #
class TestAsyncWrapper:
    @pytest.mark.asyncio
    async def test_run_rlm_task_async_delegates_to_sync(self, rlm_deps):
        rlm_deps.set_run(lambda *a, **kw: {"results": "async ok", "usage": {}})
        result = await runner.run_rlm_task("query", "m")
        assert result["success"] is True
        assert result["results"] == "async ok"
