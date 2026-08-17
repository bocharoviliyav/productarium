"""Unit tests for ``api.config.__init__`` (config loader).

Covers:
- ``replace_env_placeholders`` (dict / list / str / non-str / missing env var).
- ``load_json_config`` (DEEPWIKI_CONFIG_DIR override, missing file, bad JSON).
- ``load_generator_config`` / ``load_embedder_config`` (model_client injection).
- ``load_repo_config`` / ``load_lang_config`` (malformed lang -> default).
- ``get_model_config`` (default model, explicit model, missing providers).
- ``get_embedder_config``.
- ``configs`` dict populated at import time.
- ``fetch_openai_local_models`` (success, non-200, exception, URL normalization).
- ``get_available_models``.
- ``get_section_title``-like helpers are in test_prompts; here we focus on
  the loader + resolver surface.
"""

from __future__ import annotations

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.config as config_mod
from api.config import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    configs,
    get_available_models,
    get_embedder_config,
    get_model_config,
    load_embedder_config,
    load_generator_config,
    load_json_config,
    load_lang_config,
    load_repo_config,
    replace_env_placeholders,
)
from api.clients.openai_client import OpenAIClient


# ---------------------------------------------------------------------------
# replace_env_placeholders
# ---------------------------------------------------------------------------

class TestReplaceEnvPlaceholders:
    def test_resolves_string_placeholder(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello-world")
        assert replace_env_placeholders("v=${MY_TEST_VAR}") == "v=hello-world"

    def test_missing_placeholder_left_as_is(self, monkeypatch):
        monkeypatch.delenv("MISSING_PLACEHOLDER_X", raising=False)
        result = replace_env_placeholders("${MISSING_PLACEHOLDER_X}")
        assert result == "${MISSING_PLACEHOLDER_X}"

    def test_dict_recursive(self, monkeypatch):
        monkeypatch.setenv("FOO", "bar")
        d = {"a": "${FOO}", "b": {"c": "${FOO}"}, "d": 123}
        out = replace_env_placeholders(d)
        assert out == {"a": "bar", "b": {"c": "bar"}, "d": 123}

    def test_list_recursive(self, monkeypatch):
        monkeypatch.setenv("FOO", "bar")
        lst = ["${FOO}", "plain", 42, {"nested": "${FOO}"}]
        out = replace_env_placeholders(lst)
        assert out == ["bar", "plain", 42, {"nested": "bar"}]

    def test_non_str_passthrough(self):
        assert replace_env_placeholders(42) == 42
        assert replace_env_placeholders(True) is True
        assert replace_env_placeholders(None) is None
        assert replace_env_placeholders(3.14) == 3.14

    def test_multiple_placeholders_in_one_string(self, monkeypatch):
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert replace_env_placeholders("${A}/${B}") == "1/2"

    def test_placeholder_within_larger_text(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        assert replace_env_placeholders("http://${HOST}:8080") == "http://localhost:8080"


# ---------------------------------------------------------------------------
# load_json_config
# ---------------------------------------------------------------------------

class TestLoadJsonConfig:
    def test_loads_from_config_dir(self, monkeypatch, tmp_path):
        cfg_file = tmp_path / "custom.json"
        cfg_file.write_text(json.dumps({"key": "val"}), encoding="utf-8")
        # load_json_config reads the module-global CONFIG_DIR at call time, so
        # patch it directly instead of reloading the module to pick up the env.
        monkeypatch.setattr(config_mod, "CONFIG_DIR", str(tmp_path))
        result = config_mod.load_json_config("custom.json")
        assert result == {"key": "val"}

    def test_missing_file_returns_empty_dict(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config_mod, "CONFIG_DIR", str(tmp_path))
        assert config_mod.load_json_config("does_not_exist.json") == {}

    def test_bad_json_returns_empty_dict(self, monkeypatch, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(config_mod, "CONFIG_DIR", str(tmp_path))
        assert config_mod.load_json_config("bad.json") == {}

    def test_env_placeholders_resolved_in_loaded_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_EMBED_MODEL", "text-embed-test")
        cfg_file = tmp_path / "testcfg.json"
        cfg_file.write_text(json.dumps({"model": "${TEST_EMBED_MODEL}"}), encoding="utf-8")
        monkeypatch.setattr(config_mod, "CONFIG_DIR", str(tmp_path))
        result = config_mod.load_json_config("testcfg.json")
        assert result == {"model": "text-embed-test"}


# ---------------------------------------------------------------------------
# load_generator_config / load_embedder_config
# ---------------------------------------------------------------------------

class TestLoadGeneratorConfig:
    def test_injects_model_client(self):
        cfg = load_generator_config()
        assert "providers" in cfg
        for provider_cfg in cfg["providers"].values():
            assert provider_cfg["model_client"] is OpenAIClient

    def test_default_provider_present(self):
        cfg = load_generator_config()
        assert cfg.get("default_provider") == "openai_local"


class TestLoadEmbedderConfig:
    def test_injects_model_client(self):
        cfg = load_embedder_config()
        assert "embedder_openai_local" in cfg
        assert cfg["embedder_openai_local"]["model_client"] is OpenAIClient

    def test_retriever_and_splitter_present(self):
        cfg = load_embedder_config()
        assert "retriever" in cfg
        assert "text_splitter" in cfg


class TestLoadRepoConfig:
    def test_has_file_filters_and_repository(self):
        cfg = load_repo_config()
        assert "file_filters" in cfg
        assert "repository" in cfg
        assert "excluded_dirs" in cfg["file_filters"]


class TestLoadLangConfig:
    def test_valid_config(self):
        cfg = load_lang_config()
        assert "supported_languages" in cfg
        assert "default" in cfg

    def test_malformed_returns_default(self, monkeypatch, tmp_path):
        lang_file = tmp_path / "lang.json"
        lang_file.write_text(json.dumps({"only_partial": True}), encoding="utf-8")
        monkeypatch.setattr(config_mod, "CONFIG_DIR", str(tmp_path))
        cfg = config_mod.load_lang_config()
        assert cfg["default"] == "ru"
        assert "supported_languages" in cfg

    def test_empty_file_returns_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config_mod, "CONFIG_DIR", str(tmp_path))
        cfg = config_mod.load_lang_config()
        assert cfg["default"] == "ru"


# ---------------------------------------------------------------------------
# configs dict (populated at import time)
# ---------------------------------------------------------------------------

class TestConfigsDict:
    def test_configs_has_expected_keys(self):
        assert "default_provider" in configs
        assert "providers" in configs
        assert "embedder_openai_local" in configs
        assert "lang_config" in configs
        assert "file_filters" in configs

    def test_default_lists_non_empty(self):
        assert len(DEFAULT_EXCLUDED_DIRS) > 0
        assert len(DEFAULT_EXCLUDED_FILES) > 0
        assert "./.git/" in DEFAULT_EXCLUDED_DIRS
        assert ".env" in DEFAULT_EXCLUDED_FILES


# ---------------------------------------------------------------------------
# get_model_config
# ---------------------------------------------------------------------------

class TestGetModelConfig:
    def test_default_model(self):
        result = get_model_config()
        assert result["model_client"] is OpenAIClient
        assert "model" in result["model_kwargs"]
        # default model from generator.json
        assert result["model_kwargs"]["model"] == "qwen/qwen3.6-27b"

    def test_explicit_model_with_params(self):
        result = get_model_config("gemma3:12b")
        assert result["model_client"] is OpenAIClient
        assert result["model_kwargs"]["model"] == "gemma3:12b"
        assert result["model_kwargs"]["temperature"] == 0.1

    def test_explicit_model_without_known_params(self):
        result = get_model_config("unknown-model")
        assert result["model_kwargs"]["model"] == "unknown-model"
        # falls back to default model's params
        assert "temperature" in result["model_kwargs"]

    def test_raises_when_no_providers(self, monkeypatch):
        # Mutate the live ``configs`` dict (no reload in this module: the
        # top-level import and ``config_mod.configs`` are the same object) and
        # call through the module so the mutation is visible.
        monkeypatch.delitem(config_mod.configs, "providers", raising=False)
        with pytest.raises(ValueError, match="Provider configuration not loaded"):
            config_mod.get_model_config()

    def test_raises_when_openai_local_missing(self, monkeypatch):
        monkeypatch.setitem(config_mod.configs, "providers", {"other": {}})
        with pytest.raises(ValueError, match="not found"):
            config_mod.get_model_config()


# ---------------------------------------------------------------------------
# get_embedder_config
# ---------------------------------------------------------------------------

class TestGetEmbedderConfig:
    def test_returns_embedder_config(self):
        cfg = get_embedder_config()
        # The configs dict stores the raw embedder config (without model_client,
        # which is injected by load_embedder_config into a separate copy).
        assert "model_kwargs" in cfg or "batch_size" in cfg

    def test_returns_empty_when_missing(self, monkeypatch):
        monkeypatch.delitem(config_mod.configs, "embedder_openai_local", raising=False)
        assert config_mod.get_embedder_config() == {}


# ---------------------------------------------------------------------------
# fetch_openai_local_models / get_available_models
# ---------------------------------------------------------------------------

class TestFetchOpenaiLocalModels:
    def test_success(self, monkeypatch):
        from api.config import fetch_openai_local_models

        fake_response = types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "model-a", "object": "model", "created": 123},
                    {"id": "model-b", "object": "model", "created": 456},
                ]
            },
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_response)
        result = fetch_openai_local_models("http://localhost:1234/v1")
        assert len(result) == 2
        assert result[0]["id"] == "model-a"
        assert result[1]["id"] == "model-b"

    def test_non_200_returns_empty(self, monkeypatch):
        from api.config import fetch_openai_local_models

        fake_response = types.SimpleNamespace(status_code=500, json=lambda: {})
        monkeypatch.setattr("requests.get", lambda *a, **kw: fake_response)
        result = fetch_openai_local_models("http://localhost:1234/v1")
        assert result == []

    def test_exception_returns_empty(self, monkeypatch):
        from api.config import fetch_openai_local_models

        def boom(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("requests.get", boom)
        result = fetch_openai_local_models("http://localhost:1234/v1")
        assert result == []

    def test_url_normalization_appends_v1(self, monkeypatch):
        from api.config import fetch_openai_local_models

        captured = {}

        def fake_get(url, **kw):
            captured["url"] = url
            return types.SimpleNamespace(status_code=200, json=lambda: {"data": []})

        monkeypatch.setattr("requests.get", fake_get)
        fetch_openai_local_models("http://localhost:1234")
        assert captured["url"] == "http://localhost:1234/v1/models"

    def test_url_already_has_v1(self, monkeypatch):
        from api.config import fetch_openai_local_models

        captured = {}

        def fake_get(url, **kw):
            captured["url"] = url
            return types.SimpleNamespace(status_code=200, json=lambda: {"data": []})

        monkeypatch.setattr("requests.get", fake_get)
        fetch_openai_local_models("http://localhost:1234/v1")
        assert captured["url"] == "http://localhost:1234/v1/models"

    def test_none_url_returns_empty(self, monkeypatch):
        from api.config import fetch_openai_local_models

        # When base_url is None and LOCAL_OPENAI_BASE_URL is also None,
        # fetch_openai_local_models returns [] without calling requests.
        monkeypatch.setattr(config_mod, "LOCAL_OPENAI_BASE_URL", None)
        assert fetch_openai_local_models(None) == []


class TestGetAvailableModels:
    def test_returns_dict(self, monkeypatch):
        from api.config import fetch_openai_local_models

        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: types.SimpleNamespace(
                status_code=200,
                json=lambda: {"data": [{"id": "m1", "object": "model", "created": 0}]},
            ),
        )
        result = get_available_models()
        assert isinstance(result, dict)
        assert "openai_local" in result
        assert result["openai_local"][0]["id"] == "m1"

    def test_empty_when_no_models(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: types.SimpleNamespace(status_code=500, json=lambda: {}),
        )
        result = get_available_models()
        assert result == {}
