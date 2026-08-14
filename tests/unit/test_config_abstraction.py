"""Unit tests for ``api.config.abstraction`` (central config abstraction layer).

Covers:
- ``get_task_config`` (present + env fallback + hardcoded defaults).
- ``sync_runtime_settings`` (env export, cognee sync skip, timeout sync, cache clear).
- ``bootstrap_config`` (secret key bootstrap + runtime sync).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.config import abstraction as abst


# ---------------------------------------------------------------------------
# get_task_config
# ---------------------------------------------------------------------------

class TestGetTaskConfig:
    def test_returns_model_for_task(self, isolated_db, monkeypatch):
        monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://test:1234/v1")
        cfg = abst.get_task_config("docgen")
        assert "model" in cfg
        assert "base_url" in cfg
        assert "api_key" in cfg
        assert cfg["base_url"] == "http://test:1234/v1"

    def test_store_overrides_env(self, isolated_db, monkeypatch):
        from api.config import settings

        monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://env:1234/v1")
        settings.set_setting("models.expert.base_url", "http://store:1234/v1", encrypt=False)
        cfg = abst.get_task_config("expert")
        assert cfg["base_url"] == "http://store:1234/v1"

    def test_all_tasks_supported(self, isolated_db):
        for task in ("docgen", "expert", "summary", "cognee", "embedder"):
            cfg = abst.get_task_config(task)
            assert isinstance(cfg, dict)
            assert "model" in cfg

    def test_fallback_when_settings_raises(self, monkeypatch, isolated_db):
        # Force get_model_for_task to raise -> fallback path
        import api.config.settings as settings_mod
        original = settings_mod.get_model_for_task
        monkeypatch.setattr(
            settings_mod, "get_model_for_task",
            lambda task: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # The fallback path reads individual settings; should still return a dict
        try:
            cfg = abst.get_task_config("docgen")
            assert isinstance(cfg, dict)
            assert "model" in cfg
            assert "base_url" in cfg
            assert "api_key" in cfg
        finally:
            settings_mod.get_model_for_task = original

    def test_fallback_uses_individual_settings(self, monkeypatch, isolated_db):
        import api.config.settings as settings_mod
        from api.config import settings

        monkeypatch.setattr(
            settings_mod, "get_model_for_task",
            lambda task: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        settings.set_setting("models.docgen.model", "custom-model", encrypt=False)
        cfg = abst.get_task_config("docgen")
        assert cfg["model"] == "custom-model"


# ---------------------------------------------------------------------------
# sync_runtime_settings
# ---------------------------------------------------------------------------

class TestSyncRuntimeSettings:
    def test_exports_env_vars(self, isolated_db, monkeypatch):
        from api.config import settings

        monkeypatch.delenv("LOCAL_OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings.set_setting("models.docgen.base_url", "http://synced:1234/v1", encrypt=False)
        settings.set_setting("models.docgen.api_key", "synced-key", encrypt=True)
        abst.sync_runtime_settings()
        assert os.environ.get("LOCAL_OPENAI_BASE_URL") == "http://synced:1234/v1"
        assert os.environ.get("LOCAL_OPENAI_API_KEY") == "synced-key"
        assert os.environ.get("OPENAI_API_KEY") == "synced-key"

    def test_skips_not_needed_api_key(self, isolated_db, monkeypatch):
        from api.config import settings

        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings.set_setting("models.expert.api_key", "not-needed", encrypt=False)
        abst.sync_runtime_settings()
        # "not-needed" should NOT be exported
        assert os.environ.get("LOCAL_OPENAI_API_KEY") != "not-needed"

    def test_skips_not_needed_underscore(self, isolated_db, monkeypatch):
        from api.config import settings

        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings.set_setting("models.summary.api_key", "not_needed", encrypt=False)
        abst.sync_runtime_settings()
        assert os.environ.get("LOCAL_OPENAI_API_KEY") != "not_needed"

    def test_clears_model_ctx_cache(self, isolated_db):
        from api.utils import _MODEL_CTX_CACHE

        _MODEL_CTX_CACHE[("http://test", "model")] = (0.0, 999)
        abst.sync_runtime_settings()
        assert len(_MODEL_CTX_CACHE) == 0

    def test_does_not_raise_on_failures(self, monkeypatch):
        # Force all sub-steps to fail; sync should not raise
        import api.utils
        monkeypatch.setattr(api.utils, "_MODEL_CTX_CACHE", None)
        # Should not raise
        abst.sync_runtime_settings()

    def test_cognee_sync_failure_handled(self, monkeypatch, isolated_db):
        import api.cognee

        monkeypatch.setattr(
            "api.cognee.apply_cognee_runtime_config",
            lambda: (_ for _ in ()).throw(RuntimeError("cognee boom")),
        )
        # Should not raise
        abst.sync_runtime_settings()

    def test_timeout_sync_failure_handled(self, monkeypatch, isolated_db):
        import api.config.timeout as timeout_mod

        monkeypatch.setattr(
            timeout_mod, "sync_timeout_env",
            lambda: (_ for _ in ()).throw(RuntimeError("timeout boom")),
        )
        # Should not raise
        abst.sync_runtime_settings()


# ---------------------------------------------------------------------------
# bootstrap_config
# ---------------------------------------------------------------------------

class TestBootstrapConfig:
    def test_runs_without_error(self, isolated_db):
        abst.bootstrap_config()

    def test_secret_key_bootstrap_failure_handled(self, monkeypatch):
        from api.config import settings

        monkeypatch.setattr(
            settings, "bootstrap_secret_key",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # Should not raise despite bootstrap failure
        abst.bootstrap_config()

    def test_calls_sync_runtime(self, isolated_db, monkeypatch):
        called = {"sync": False}
        original = abst.sync_runtime_settings

        def fake_sync():
            called["sync"] = True

        monkeypatch.setattr(abst, "sync_runtime_settings", fake_sync)
        abst.bootstrap_config()
        assert called["sync"] is True
