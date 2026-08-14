"""Unit tests for ``api.config.settings`` (encrypted settings store).

Covers:
- Core CRUD: ``get_setting`` / ``set_setting`` / ``delete_setting`` / ``get_secret``.
- Encrypted roundtrip (encrypt=True -> Fernet ciphertext -> decrypt on read).
- ``list_settings`` (prefix filter, encrypted flag, no plaintext leak).
- ``_fernet`` (env > persisted > dev key precedence).
- ``bootstrap_secret_key`` (env present -> no-op; env absent -> persisted).
- Grouped getters: ``get_model_for_task`` / ``get_git_creds`` /
  ``get_confluence_creds`` / ``get_integration_config`` (read-through + env fallback).
- ``_sanitize_api_key`` (quotes, Bearer prefix, whitespace).
- ``_parse_int_setting`` (valid, empty, negative, non-numeric).
- ``get_rlm_mode`` / ``get_all_rlm_modes`` (fast-rlm unavailable -> "llm").
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.config import settings as settings_mod


# ---------------------------------------------------------------------------
# Core CRUD (plaintext)
# ---------------------------------------------------------------------------

class TestSettingCRUD:
    def test_set_and_get_plaintext(self, isolated_db):
        settings_mod.set_setting("test.key", "hello", encrypt=False)
        assert settings_mod.get_setting("test.key") == "hello"

    def test_get_default_when_missing(self, isolated_db):
        assert settings_mod.get_setting("nonexistent.key", default="fallback") == "fallback"

    def test_get_default_none_when_missing(self, isolated_db):
        assert settings_mod.get_setting("nonexistent.key") is None

    def test_delete_existing(self, isolated_db):
        settings_mod.set_setting("test.delete", "val", encrypt=False)
        assert settings_mod.delete_setting("test.delete") is True
        assert settings_mod.get_setting("test.delete") is None

    def test_delete_nonexistent_returns_false(self, isolated_db):
        assert settings_mod.delete_setting("nonexistent.key") is False

    def test_update_existing(self, isolated_db):
        settings_mod.set_setting("test.update", "v1", encrypt=False)
        settings_mod.set_setting("test.update", "v2", encrypt=False)
        assert settings_mod.get_setting("test.update") == "v2"

    def test_set_none_value(self, isolated_db):
        settings_mod.set_setting("test.none", None, encrypt=False)
        # stored as None
        assert settings_mod.get_setting("test.none") is None

    def test_get_secret_alias(self, isolated_db):
        settings_mod.set_setting("test.secret", "secret-val", encrypt=False)
        assert settings_mod.get_secret("test.secret") == "secret-val"
        assert settings_mod.get_secret("test.secret.missing", default="d") == "d"


# ---------------------------------------------------------------------------
# Encrypted roundtrip
# ---------------------------------------------------------------------------

class TestEncryptedSettings:
    def test_encrypted_roundtrip(self, isolated_db):
        settings_mod.set_setting("test.enc", "my-secret-value", encrypt=True)
        # The stored value should NOT be plaintext
        from api.models import SettingORM
        with isolated_db.SessionLocal() as db:
            row = db.get(SettingORM, "test.enc")
            assert row.encrypted is True
            assert row.value != "my-secret-value"
        # But get_setting decrypts it
        assert settings_mod.get_setting("test.enc") == "my-secret-value"

    def test_get_secret_decrypts_encrypted(self, isolated_db):
        settings_mod.set_setting("test.enc2", "secret-data", encrypt=True)
        assert settings_mod.get_secret("test.enc2") == "secret-data"

    def test_encrypted_empty_value(self, isolated_db):
        settings_mod.set_setting("test.enc_empty", "", encrypt=True)
        # encrypt=True with empty value -> stored as empty (not encrypted)
        assert settings_mod.get_setting("test.enc_empty") == ""

    def test_decrypt_failure_returns_default(self, isolated_db):
        # Write a bad ciphertext directly
        from api.models import SettingORM
        with isolated_db.SessionLocal() as db:
            row = SettingORM(key="test.bad", value="not-valid-fernet", encrypted=True)
            db.add(row)
            db.commit()
        assert settings_mod.get_setting("test.bad", default="safe") == "safe"


# ---------------------------------------------------------------------------
# list_settings
# ---------------------------------------------------------------------------

class TestListSettings:
    def test_list_all(self, isolated_db):
        settings_mod.set_setting("list.a", "1", encrypt=False)
        settings_mod.set_setting("list.b", "2", encrypt=False)
        result = settings_mod.list_settings()
        keys = [r["key"] for r in result]
        assert "list.a" in keys
        assert "list.b" in keys

    def test_list_with_prefix(self, isolated_db):
        settings_mod.set_setting("prefix.x", "1", encrypt=False)
        settings_mod.set_setting("other.y", "2", encrypt=False)
        result = settings_mod.list_settings(prefix="prefix.")
        keys = [r["key"] for r in result]
        assert "prefix.x" in keys
        assert "other.y" not in keys

    def test_list_encrypted_flag_no_plaintext(self, isolated_db):
        settings_mod.set_setting("list.enc", "secret-val", encrypt=True)
        result = settings_mod.list_settings(prefix="list.enc")
        assert len(result) == 1
        assert result[0]["encrypted"] is True
        # The value in the listing is the ciphertext, not the plaintext
        assert result[0]["value"] != "secret-val"

    def test_list_empty_when_db_down(self, monkeypatch):
        # Force SessionLocal to raise
        def boom():
            raise RuntimeError("DB down")

        monkeypatch.setattr("api.db.SessionLocal", boom)
        assert settings_mod.list_settings() == []


# ---------------------------------------------------------------------------
# _fernet / bootstrap_secret_key
# ---------------------------------------------------------------------------

class TestFernet:
    def test_fernet_returns_valid_instance(self):
        f = settings_mod._fernet()
        assert f is not None
        # Roundtrip
        token = f.encrypt(b"test")
        assert f.decrypt(token) == b"test"

    def test_fernet_with_env_key(self, monkeypatch):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("SETTINGS_SECRET_KEY", key)
        f = settings_mod._fernet()
        assert f is not None
        # Should use the env key
        assert f.decrypt(f.encrypt(b"x")) == b"x"

    def test_fnet_invalid_key_returns_none(self, monkeypatch):
        monkeypatch.setenv("SETTINGS_SECRET_KEY", "not-a-valid-key")
        assert settings_mod._fernet() is None

    def test_dev_fernet_key_cached(self):
        k1 = settings_mod._dev_fernet_key()
        k2 = settings_mod._dev_fernet_key()
        assert k1 == k2

    def test_persisted_key_path_with_config_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEEPWIKI_CONFIG_DIR", str(tmp_path))
        path = settings_mod._persisted_key_path()
        assert path == str(tmp_path / ".settings_secret_key")

    def test_persisted_key_path_default(self, monkeypatch):
        monkeypatch.delenv("DEEPWIKI_CONFIG_DIR", raising=False)
        path = settings_mod._persisted_key_path()
        assert path.endswith(".adalflow/.settings_secret_key")

    def test_bootstrap_secret_key_noop_when_env_set(self, monkeypatch):
        monkeypatch.setenv("SETTINGS_SECRET_KEY", "existing-key")
        settings_mod.bootstrap_secret_key()
        # Should remain unchanged
        assert os.environ.get("SETTINGS_SECRET_KEY") == "existing-key"

    def test_bootstrap_secret_key_creates_persisted(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SETTINGS_SECRET_KEY", raising=False)
        monkeypatch.setenv("DEEPWIKI_CONFIG_DIR", str(tmp_path))
        settings_mod.bootstrap_secret_key()
        assert os.environ.get("SETTINGS_SECRET_KEY") is not None

    def test_load_or_create_persisted_key_creates_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DEEPWIKI_CONFIG_DIR", str(tmp_path))
        key = settings_mod._load_or_create_persisted_key()
        assert key is not None
        # Second call should read the same key from file
        key2 = settings_mod._load_or_create_persisted_key()
        assert key == key2


# ---------------------------------------------------------------------------
# Grouped convenience getters
# ---------------------------------------------------------------------------

class TestGetModelForTask:
    def test_env_fallback(self, monkeypatch, isolated_db):
        monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://my-server:1234/v1")
        monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "my-key")
        cfg = settings_mod.get_model_for_task("docgen")
        assert cfg["base_url"] == "http://my-server:1234/v1"
        assert cfg["api_key"] == "my-key"

    def test_store_overrides_env(self, monkeypatch, isolated_db):
        monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://env:1234/v1")
        settings_mod.set_setting("models.docgen.base_url", "http://store:1234/v1", encrypt=False)
        cfg = settings_mod.get_model_for_task("docgen")
        assert cfg["base_url"] == "http://store:1234/v1"

    def test_returns_model_key(self, isolated_db):
        cfg = settings_mod.get_model_for_task("expert")
        assert "model" in cfg
        assert "base_url" in cfg
        assert "api_key" in cfg
        assert "max_prompt_tokens" in cfg
        assert "dimensions" in cfg

    def test_max_prompt_tokens_from_store(self, isolated_db):
        settings_mod.set_setting("models.docgen.max_prompt_tokens", "8192", encrypt=False)
        cfg = settings_mod.get_model_for_task("docgen")
        assert cfg["max_prompt_tokens"] == 8192

    def test_dimensions_from_store(self, isolated_db):
        settings_mod.set_setting("models.embedder.dimensions", "768", encrypt=False)
        cfg = settings_mod.get_model_for_task("embedder")
        assert cfg["dimensions"] == 768


class TestGetGitCreds:
    def test_env_fallback_github(self, monkeypatch, isolated_db):
        monkeypatch.setenv("GITHUB_ENTERPRISE_URL", "https://github.corp.com")
        creds = settings_mod.get_git_creds("github")
        assert creds["url"] == "https://github.corp.com"
        assert creds["token"] is None  # token has no env fallback

    def test_env_fallback_gitlab(self, monkeypatch, isolated_db):
        monkeypatch.setenv("GITLAB_SELF_HOSTED_URL", "https://gitlab.corp.com")
        creds = settings_mod.get_git_creds("gitlab")
        assert creds["url"] == "https://gitlab.corp.com"

    def test_store_overrides_env(self, monkeypatch, isolated_db):
        monkeypatch.setenv("GITHUB_ENTERPRISE_URL", "https://env.com")
        settings_mod.set_setting("git.github.url", "https://store.com", encrypt=False)
        settings_mod.set_setting("git.github.token", "tok", encrypt=True)
        creds = settings_mod.get_git_creds("github")
        assert creds["url"] == "https://store.com"
        assert creds["token"] == "tok"

    def test_unknown_host_no_env(self, isolated_db):
        creds = settings_mod.get_git_creds("bitbucket")
        assert creds["url"] == ""
        assert creds["token"] is None


class TestGetConfluenceCreds:
    def test_env_fallback(self, monkeypatch, isolated_db):
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://wiki.corp.com")
        monkeypatch.setenv("CONFLUENCE_TOKEN", "conf-tok")
        monkeypatch.setenv("CONFLUENCE_SPACE", "ENG")
        creds = settings_mod.get_confluence_creds()
        assert creds["base_url"] == "https://wiki.corp.com"
        assert creds["token"] == "conf-tok"
        assert creds["space"] == "ENG"
        assert creds["mode"] == "direct"

    def test_store_overrides(self, monkeypatch, isolated_db):
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://env.com")
        settings_mod.set_setting("confluence.base_url", "https://store.com", encrypt=False)
        settings_mod.set_setting("confluence.mode", "mcp", encrypt=False)
        creds = settings_mod.get_confluence_creds()
        assert creds["base_url"] == "https://store.com"
        assert creds["mode"] == "mcp"

    def test_mcp_defaults(self, isolated_db):
        creds = settings_mod.get_confluence_creds()
        # mcp_server defaults to "confluence"
        assert creds["mcp_server"] == "confluence"
        assert creds["mcp_tool"] is None


class TestGetIntegrationConfig:
    def test_missing_returns_empty(self, isolated_db):
        assert settings_mod.get_integration_config("nonexistent") == {}

    def test_valid_json(self, isolated_db):
        settings_mod.set_setting(
            "integrations.mcp",
            json.dumps({"servers": [{"name": "s1"}]}),
            encrypt=False,
        )
        cfg = settings_mod.get_integration_config("mcp")
        assert cfg == {"servers": [{"name": "s1"}]}

    def test_invalid_json_returns_empty(self, isolated_db):
        settings_mod.set_setting("integrations.bad", "not-json{", encrypt=False)
        assert settings_mod.get_integration_config("bad") == {}

    def test_non_dict_json_wrapped(self, isolated_db):
        settings_mod.set_setting("integrations.list", "[1,2,3]", encrypt=False)
        cfg = settings_mod.get_integration_config("list")
        assert cfg == {"value": [1, 2, 3]}


# ---------------------------------------------------------------------------
# _sanitize_api_key
# ---------------------------------------------------------------------------

class TestSanitizeApiKey:
    def test_plain_key(self):
        assert settings_mod._sanitize_api_key("sk-abc123") == "sk-abc123"

    def test_strips_whitespace(self):
        assert settings_mod._sanitize_api_key("  sk-abc123  ") == "sk-abc123"

    def test_strips_double_quotes(self):
        assert settings_mod._sanitize_api_key('"sk-abc123"') == "sk-abc123"

    def test_strips_single_quotes(self):
        assert settings_mod._sanitize_api_key("'sk-abc123'") == "sk-abc123"

    def test_strips_bearer_prefix(self):
        assert settings_mod._sanitize_api_key("Bearer sk-abc123") == "sk-abc123"

    def test_strips_bearer_prefix_case_insensitive(self):
        assert settings_mod._sanitize_api_key("bearer sk-abc123") == "sk-abc123"

    def test_empty_returns_empty(self):
        assert settings_mod._sanitize_api_key("") == ""

    def test_none_returns_none(self):
        assert settings_mod._sanitize_api_key(None) is None

    def test_strips_bearer_and_quotes(self):
        assert settings_mod._sanitize_api_key('"Bearer sk-abc"') == "sk-abc"


# ---------------------------------------------------------------------------
# _parse_int_setting
# ---------------------------------------------------------------------------

class TestParseIntSetting:
    def test_valid_int(self):
        assert settings_mod._parse_int_setting("42") == 42

    def test_none_returns_none(self):
        assert settings_mod._parse_int_setting(None) is None

    def test_empty_returns_none(self):
        assert settings_mod._parse_int_setting("") is None

    def test_whitespace_returns_none(self):
        assert settings_mod._parse_int_setting("   ") is None

    def test_non_numeric_returns_none(self):
        assert settings_mod._parse_int_setting("abc") is None

    def test_negative_returns_none(self):
        assert settings_mod._parse_int_setting("-5") is None

    def test_zero_valid(self):
        assert settings_mod._parse_int_setting("0") == 0


# ---------------------------------------------------------------------------
# get_rlm_mode / get_all_rlm_modes
# ---------------------------------------------------------------------------

class TestRlmMode:
    def test_returns_llm_when_fast_rlm_unavailable(self, monkeypatch, isolated_db):
        # fast_rlm is not installed in the test env, so _FAST_RLM_AVAILABLE is False
        creds = settings_mod.get_rlm_mode("docgen")
        assert creds == "llm"

    def test_get_all_rlm_modes(self, isolated_db):
        modes = settings_mod.get_all_rlm_modes()
        assert set(modes.keys()) == {"docgen", "expert", "summary"}
        # All should be "llm" since fast-rlm is unavailable
        for v in modes.values():
            assert v == "llm"

    def test_get_rlm_mode_env_default_when_fast_rlm_available(self, monkeypatch, isolated_db):
        # Simulate fast-rlm being available
        import api.rlm.runner as runner_mod
        monkeypatch.setattr(runner_mod, "_FAST_RLM_AVAILABLE", True)
        monkeypatch.setenv("RLM_DEFAULT_MODE", "auto")
        # Clear any stored setting
        settings_mod.set_setting("rlm.docgen.mode", "", encrypt=False)
        assert settings_mod.get_rlm_mode("docgen") == "auto"

    def test_get_rlm_mode_store_overrides_env(self, monkeypatch, isolated_db):
        import api.rlm.runner as runner_mod
        monkeypatch.setattr(runner_mod, "_FAST_RLM_AVAILABLE", True)
        monkeypatch.setenv("RLM_DEFAULT_MODE", "auto")
        settings_mod.set_setting("rlm.expert.mode", "llm", encrypt=False)
        assert settings_mod.get_rlm_mode("expert") == "llm"

    def test_get_rlm_mode_invalid_store_falls_back(self, monkeypatch, isolated_db):
        import api.rlm.runner as runner_mod
        monkeypatch.setattr(runner_mod, "_FAST_RLM_AVAILABLE", True)
        monkeypatch.setenv("RLM_DEFAULT_MODE", "auto")
        settings_mod.set_setting("rlm.summary.mode", "invalid_mode", encrypt=False)
        assert settings_mod.get_rlm_mode("summary") == "auto"

    def test_get_rlm_mode_invalid_env_falls_back_to_auto(self, monkeypatch, isolated_db):
        import api.rlm.runner as runner_mod
        monkeypatch.setattr(runner_mod, "_FAST_RLM_AVAILABLE", True)
        monkeypatch.setenv("RLM_DEFAULT_MODE", "invalid")
        assert settings_mod.get_rlm_mode("docgen") == "auto"
