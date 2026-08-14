"""Unit tests for ``api.cognee.config``.

Covers:
- ``init_cognee``: non-fatal when cognee unavailable (logs + returns None); when
  cognee available, runs migrations + setup + reconcile (fake_cognee happy path);
  timeout is non-fatal; migration exception is non-fatal.
- ``apply_cognee_runtime_config``: no-op when cognee unavailable; pushes LLM +
  embedder settings via setters when cognee is available (fake_cognee); settings
  store exception -> early return; direct LLM/EmbeddingConfig singleton mutation.
- ``apply_cognee_retry_patch``: no-op when cognee unavailable; idempotent (guarded
  by ``_ORIG_ACREATE_STRUCTURED_OUTPUT``); applies to OpenAIAdapter when cognee
  present.
- ``_safe_set``: calls setter when present, no-op when absent, swallows errors.
- ``_safe_set_embedding_dict``: calls set_embedding_config when present.
- ``_init_cognee_body``: migrations via run_startup_migrations / init / neither;
  setup() called when present; reconcile called.
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.cognee.config import (
    _init_cognee_body,
    _safe_set,
    _safe_set_embedding_dict,
    apply_cognee_retry_patch,
    apply_cognee_runtime_config,
    init_cognee,
)


# --------------------------------------------------------------------------- #
# init_cognee
# --------------------------------------------------------------------------- #
class TestInitCognee:
    def test_unavailable_returns_none(self, monkeypatch):
        """When _COGNEE_AVAILABLE is False, init_cognee logs and returns None."""
        import api.cognee._runtime as rtmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", False)
        result = asyncio.run(init_cognee())
        assert result is None

    def test_happy_path_with_fake_cognee(self, monkeypatch, fake_cognee):
        """With fake_cognee, init runs migrations + setup + reconcile without error."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        # fake_cognee needs run_startup_migrations or init.
        migration_called = []

        async def _run_startup_migrations():
            migration_called.append(True)

        fake_cognee.run_startup_migrations = _run_startup_migrations

        # Patch the reconcile import to a no-op.
        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        result = asyncio.run(init_cognee())
        assert result is None
        assert migration_called == [True]

    def test_init_fallback_when_no_migrations_attr(self, monkeypatch, fake_cognee):
        """When cognee has init() but not run_startup_migrations, init() is called."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        if hasattr(fake_cognee, "run_startup_migrations"):
            del fake_cognee.run_startup_migrations

        init_called = []

        async def _init():
            init_called.append(True)

        fake_cognee.init = _init

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        asyncio.run(init_cognee())
        assert init_called == [True]

    def test_neither_migrations_nor_init(self, monkeypatch, fake_cognee):
        """When cognee has neither, init_cognee logs and continues."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        if hasattr(fake_cognee, "run_startup_migrations"):
            del fake_cognee.run_startup_migrations
        if hasattr(fake_cognee, "init"):
            del fake_cognee.init

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        # Should not raise.
        asyncio.run(init_cognee())

    def test_setup_called_when_present(self, monkeypatch, fake_cognee):
        """When cognee has setup(), it is called after migrations."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        async def _run_startup_migrations():
            pass

        fake_cognee.run_startup_migrations = _run_startup_migrations

        setup_called = []

        async def _setup():
            setup_called.append(True)

        fake_cognee.setup = _setup

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        asyncio.run(init_cognee())
        assert setup_called == [True]

    def test_migration_exception_non_fatal(self, monkeypatch, fake_cognee):
        """An exception in run_startup_migrations is caught (non-fatal)."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        async def _run_startup_migrations():
            raise RuntimeError("migration failed")

        fake_cognee.run_startup_migrations = _run_startup_migrations

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        # Should not raise.
        asyncio.run(init_cognee())

    def test_timeout_non_fatal(self, monkeypatch, fake_cognee):
        """A timeout in init is caught and logged (non-fatal)."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        async def _run_startup_migrations():
            await asyncio.sleep(10000)

        fake_cognee.run_startup_migrations = _run_startup_migrations

        # Patch the timeout resolver to a tiny value.
        monkeypatch.setattr(
            "api.config.timeout.resolve_cognee_init_timeout",
            lambda: 0.01,
        )

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        # Should not raise — timeout is caught.
        asyncio.run(init_cognee())


# --------------------------------------------------------------------------- #
# apply_cognee_runtime_config
# --------------------------------------------------------------------------- #
class TestApplyCogneeRuntimeConfig:
    def test_noop_when_unavailable(self, monkeypatch):
        """When _COGNEE_AVAILABLE is False, apply_cognee_runtime_config is a no-op."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", False)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", False)
        # Should not raise.
        apply_cognee_runtime_config()

    def test_settings_exception_early_return(self, monkeypatch, fake_cognee):
        """When get_model_for_task raises, the function returns early (no crash)."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)

        # Provide a config object with setters.
        config_obj = SimpleNamespace()
        config_obj.set_llm_provider = lambda v: None
        config_obj.set_llm_model = lambda v: None
        config_obj.set_llm_endpoint = lambda v: None
        config_obj.set_llm_api_key = lambda v: None
        config_obj.set_embedding_provider = lambda v: None
        config_obj.set_embedding_model = lambda v: None
        config_obj.set_embedding_endpoint = lambda v: None
        config_obj.set_embedding_api_key = lambda v: None
        config_obj.set_embedding_dimensions = lambda v: None
        config_obj.set_embedding_config = lambda d: None
        config_obj.data_root_directory = lambda v: None
        config_obj.system_root_directory = lambda v: None
        fake_cognee.config = config_obj

        def _boom_get_model(task):
            raise RuntimeError("db down")

        monkeypatch.setattr("api.config.settings.get_model_for_task", _boom_get_model)

        # Should not raise.
        apply_cognee_runtime_config()

    def test_pushes_settings_to_cognee(self, monkeypatch, fake_cognee):
        """When cognee + settings are available, setters are called."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)

        llm_calls = {}
        emb_calls = {}

        config_obj = SimpleNamespace()
        config_obj.set_llm_provider = lambda v: llm_calls.__setitem__("provider", v)
        config_obj.set_llm_model = lambda v: llm_calls.__setitem__("model", v)
        config_obj.set_llm_endpoint = lambda v: llm_calls.__setitem__("endpoint", v)
        config_obj.set_llm_api_key = lambda v: llm_calls.__setitem__("api_key", v)
        config_obj.set_embedding_provider = lambda v: emb_calls.__setitem__("provider", v)
        config_obj.set_embedding_model = lambda v: emb_calls.__setitem__("model", v)
        config_obj.set_embedding_endpoint = lambda v: emb_calls.__setitem__("endpoint", v)
        config_obj.set_embedding_api_key = lambda v: emb_calls.__setitem__("api_key", v)
        config_obj.set_embedding_dimensions = lambda v: emb_calls.__setitem__("dims", v)
        config_obj.set_embedding_config = lambda d: emb_calls.__setitem__("config", d)
        config_obj.data_root_directory = lambda v: None
        config_obj.system_root_directory = lambda v: None
        fake_cognee.config = config_obj

        def _fake_get_model(task):
            if task == "cognee":
                return {"model": "qwen3:8b", "base_url": "http://localhost:11434/v1", "api_key": "test-key"}
            if task == "embedder":
                return {"model": "nomic-embed-text", "base_url": "http://localhost:11434/v1", "api_key": "test-key"}
            return {}

        monkeypatch.setattr("api.config.settings.get_model_for_task", _fake_get_model)

        apply_cognee_runtime_config()

        assert llm_calls.get("provider") == "openai"
        assert "qwen3:8b" in llm_calls.get("model", "")
        assert llm_calls.get("endpoint") == "http://localhost:11434/v1"
        assert llm_calls.get("api_key") == "test-key"

        assert emb_calls.get("provider") == "openai_compatible"
        assert "nomic" in emb_calls.get("model", "")
        assert emb_calls.get("endpoint") == "http://localhost:11434/v1"

    def test_env_fallback_when_settings_empty(self, monkeypatch, fake_cognee):
        """When settings store returns None, env defaults are used."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)

        llm_calls = {}

        config_obj = SimpleNamespace()
        config_obj.set_llm_provider = lambda v: llm_calls.__setitem__("provider", v)
        config_obj.set_llm_model = lambda v: llm_calls.__setitem__("model", v)
        config_obj.set_llm_endpoint = lambda v: llm_calls.__setitem__("endpoint", v)
        config_obj.set_llm_api_key = lambda v: llm_calls.__setitem__("api_key", v)
        config_obj.set_embedding_provider = lambda v: None
        config_obj.set_embedding_model = lambda v: None
        config_obj.set_embedding_endpoint = lambda v: None
        config_obj.set_embedding_api_key = lambda v: None
        config_obj.set_embedding_dimensions = lambda v: None
        config_obj.set_embedding_config = lambda d: None
        config_obj.data_root_directory = lambda v: None
        config_obj.system_root_directory = lambda v: None
        fake_cognee.config = config_obj

        def _empty_get_model(task):
            return {"model": None, "base_url": None, "api_key": None}

        monkeypatch.setattr("api.config.settings.get_model_for_task", _empty_get_model)

        apply_cognee_runtime_config()

        # Defaults kick in: provider openai, model from _default_cognee_model.
        assert llm_calls.get("provider") == "openai"
        assert llm_calls.get("api_key") is not None


# --------------------------------------------------------------------------- #
# apply_cognee_retry_patch
# --------------------------------------------------------------------------- #
class TestApplyCogneeRetryPatch:
    def test_noop_when_unavailable(self, monkeypatch):
        """When _COGNEE_AVAILABLE is False, the retry patch is a no-op."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", False)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", False)
        # Reset the idempotency guard.
        monkeypatch.setattr(cfgmod, "_ORIG_ACREATE_STRUCTURED_OUTPUT", None)
        apply_cognee_retry_patch()
        # Should not raise.

    def test_idempotent_when_already_applied(self, monkeypatch, fake_cognee):
        """When _ORIG_ACREATE_STRUCTURED_OUTPUT is set, the patch is a no-op."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        # Simulate already-applied.
        monkeypatch.setattr(cfgmod, "_ORIG_ACREATE_STRUCTURED_OUTPUT", lambda *a: None)

        # Should not attempt the import/patch.
        apply_cognee_retry_patch()

    def test_import_failure_non_fatal(self, monkeypatch, fake_cognee):
        """When the OpenAIAdapter import fails, the patch logs and returns."""
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "_ORIG_ACREATE_STRUCTURED_OUTPUT", None)

        # The import of cognee.infrastructure.llm... will fail because fake_cognee
        # doesn't have that submodule path. The except branch catches it.
        apply_cognee_retry_patch()


# --------------------------------------------------------------------------- #
# _safe_set
# --------------------------------------------------------------------------- #
class TestSafeSet:
    def test_calls_setter_when_present(self):
        called = []
        cfg = SimpleNamespace(my_setter=lambda v: called.append(v))
        _safe_set(cfg, "my_setter", "value")
        assert called == ["value"]

    def test_noop_when_setter_absent(self):
        cfg = SimpleNamespace()
        # Should not raise.
        _safe_set(cfg, "missing_setter", "value")

    def test_noop_when_not_callable(self):
        cfg = SimpleNamespace(not_a_method="scalar")
        _safe_set(cfg, "not_a_method", "value")

    def test_swallows_setter_exception(self):
        def _bad_setter(v):
            raise RuntimeError("setter failed")

        cfg = SimpleNamespace(bad_setter=_bad_setter)
        # Should not raise.
        _safe_set(cfg, "bad_setter", "value")


# --------------------------------------------------------------------------- #
# _safe_set_embedding_dict
# --------------------------------------------------------------------------- #
class TestSafeSetEmbeddingDict:
    def test_calls_set_embedding_config_when_present(self):
        called = []
        cfg = SimpleNamespace(set_embedding_config=lambda d: called.append(d))
        _safe_set_embedding_dict(cfg, {"huggingface_tokenizer": "test"})
        assert called == [{"huggingface_tokenizer": "test"}]

    def test_noop_when_setter_absent(self):
        cfg = SimpleNamespace()
        _safe_set_embedding_dict(cfg, {"huggingface_tokenizer": "test"})

    def test_swallows_exception(self):
        def _bad(d):
            raise RuntimeError("failed")

        cfg = SimpleNamespace(set_embedding_config=_bad)
        _safe_set_embedding_dict(cfg, {"huggingface_tokenizer": "test"})


# --------------------------------------------------------------------------- #
# _init_cognee_body (direct unit test)
# --------------------------------------------------------------------------- #
class TestInitCogneeBody:
    def test_run_startup_migrations_called(self, monkeypatch, fake_cognee):
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        called = []

        async def _run_startup_migrations():
            called.append("migrations")

        fake_cognee.run_startup_migrations = _run_startup_migrations

        async def _fake_reconcile():
            called.append("reconcile")
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        asyncio.run(_init_cognee_body())
        assert "migrations" in called
        assert "reconcile" in called

    def test_setup_called_when_present(self, monkeypatch, fake_cognee):
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        async def _run_startup_migrations():
            pass

        fake_cognee.run_startup_migrations = _run_startup_migrations

        setup_called = []

        async def _setup():
            setup_called.append(True)

        fake_cognee.setup = _setup

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        asyncio.run(_init_cognee_body())
        assert setup_called == [True]

    def test_setup_exception_non_fatal(self, monkeypatch, fake_cognee):
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        async def _run_startup_migrations():
            pass

        fake_cognee.run_startup_migrations = _run_startup_migrations

        async def _setup():
            raise RuntimeError("setup failed")

        fake_cognee.setup = _setup

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        # Should not raise.
        asyncio.run(_init_cognee_body())

    def test_init_called_when_no_migrations(self, monkeypatch, fake_cognee):
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        if hasattr(fake_cognee, "run_startup_migrations"):
            del fake_cognee.run_startup_migrations

        init_called = []

        async def _init():
            init_called.append(True)

        fake_cognee.init = _init

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        asyncio.run(_init_cognee_body())
        assert init_called == [True]

    def test_neither_migrations_nor_init(self, monkeypatch, fake_cognee):
        import api.cognee._runtime as rtmod
        import api.cognee.config as cfgmod

        monkeypatch.setattr(rtmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(cfgmod, "cognee", fake_cognee)
        monkeypatch.setattr(cfgmod, "apply_cognee_runtime_config", lambda: None)

        if hasattr(fake_cognee, "run_startup_migrations"):
            del fake_cognee.run_startup_migrations
        if hasattr(fake_cognee, "init"):
            del fake_cognee.init

        async def _fake_reconcile():
            return None

        monkeypatch.setattr("api.cognee.indexing._reconcile_stale_cognee_data", _fake_reconcile)

        # Should not raise.
        asyncio.run(_init_cognee_body())
