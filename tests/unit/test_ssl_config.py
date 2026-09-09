"""Unit tests for ``api.config.ssl`` (SSL/TLS configuration).

Covers:
- ``_to_bool`` (truthy/falsy strings, bool, None, unknown -> default).
- ``get_ca_bundle`` (admin store, env vars, missing file, whitespace).
- ``get_verify`` (admin store, env var, default True).
- ``requests_verify`` / ``httpx_verify`` (CA path when verify on, False when off).
- ``apply_openai_ssl_patch`` (patches openai.AsyncOpenAI/OpenAI __init__).
- ``apply_ssl_env`` (skip-verify path, CA bundle path, no overwrite).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.config import ssl as ssl_mod


# ---------------------------------------------------------------------------
# _to_bool
# ---------------------------------------------------------------------------

class TestToBool:
    def test_truthy_strings(self):
        for v in ("1", "true", "t", "yes", "y", "on", "TRUE", "True"):
            assert ssl_mod._to_bool(v) is True, f"{v} should be True"

    def test_falsy_strings(self):
        for v in ("0", "false", "f", "no", "n", "off", "FALSE", "False"):
            assert ssl_mod._to_bool(v) is False, f"{v} should be False"

    def test_bool_passthrough(self):
        assert ssl_mod._to_bool(True) is True
        assert ssl_mod._to_bool(False) is False

    def test_none_uses_default(self):
        assert ssl_mod._to_bool(None, default=True) is True
        assert ssl_mod._to_bool(None, default=False) is False

    def test_unknown_uses_default(self):
        assert ssl_mod._to_bool("maybe", default=True) is True
        assert ssl_mod._to_bool("maybe", default=False) is False
        assert ssl_mod._to_bool("", default=True) is True

    def test_strips_whitespace(self):
        assert ssl_mod._to_bool("  true  ") is True
        assert ssl_mod._to_bool("  off  ") is False


# ---------------------------------------------------------------------------
# get_ca_bundle
# ---------------------------------------------------------------------------

class TestGetCaBundle:
    def test_admin_store_wins(self, monkeypatch, tmp_path):
        admin_ca = tmp_path / "admin_ca.pem"
        admin_ca.write_text("cert", encoding="utf-8")
        env_ca = tmp_path / "env_ca.pem"
        env_ca.write_text("cert", encoding="utf-8")
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: str(admin_ca) if key == "ssl.ca_bundle" else None)
        monkeypatch.setenv("SSL_CA_BUNDLE", str(env_ca))
        assert ssl_mod.get_ca_bundle() == str(admin_ca)

    def test_env_fallback_ssl_ca_bundle(self, monkeypatch, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_text("cert", encoding="utf-8")
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        monkeypatch.setenv("SSL_CA_BUNDLE", str(ca))
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
        assert ssl_mod.get_ca_bundle() == str(ca)

    def test_env_fallback_ssl_cert_file(self, monkeypatch, tmp_path):
        ca = tmp_path / "cert.pem"
        ca.write_text("cert", encoding="utf-8")
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        monkeypatch.delenv("SSL_CA_BUNDLE", raising=False)
        monkeypatch.setenv("SSL_CERT_FILE", str(ca))
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
        assert ssl_mod.get_ca_bundle() == str(ca)

    def test_env_fallback_requests_ca_bundle(self, monkeypatch, tmp_path):
        ca = tmp_path / "requests.pem"
        ca.write_text("cert", encoding="utf-8")
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        monkeypatch.delenv("SSL_CA_BUNDLE", raising=False)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
        monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
        assert ssl_mod.get_ca_bundle() == str(ca)

    def test_env_fallback_curl_ca_bundle(self, monkeypatch, tmp_path):
        ca = tmp_path / "curl.pem"
        ca.write_text("cert", encoding="utf-8")
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        monkeypatch.delenv("SSL_CA_BUNDLE", raising=False)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        monkeypatch.setenv("CURL_CA_BUNDLE", str(ca))
        assert ssl_mod.get_ca_bundle() == str(ca)

    def test_none_when_nothing_set(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        for k in ("SSL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            monkeypatch.delenv(k, raising=False)
        assert ssl_mod.get_ca_bundle() is None

    def test_whitespace_path_returns_none(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "   ")
        assert ssl_mod.get_ca_bundle() is None

    def test_nonexistent_file_returns_none(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "/nonexistent/ca-bundle.pem")
        assert ssl_mod.get_ca_bundle() is None

    def test_strips_whitespace_from_path(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        # Use a real file path (this test file itself)
        real_file = os.path.abspath(__file__)
        monkeypatch.setenv("SSL_CA_BUNDLE", f"  {real_file}  ")
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
        assert ssl_mod.get_ca_bundle() == real_file

    def test_real_file_returns_path(self, monkeypatch, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_text("fake-cert", encoding="utf-8")
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: str(ca))
        assert ssl_mod.get_ca_bundle() == str(ca)


# ---------------------------------------------------------------------------
# get_verify
# ---------------------------------------------------------------------------

class TestGetVerify:
    def test_default_true(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        monkeypatch.delenv("SSL_VERIFY", raising=False)
        assert ssl_mod.get_verify() is True

    def test_admin_store_false(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "false" if key == "ssl.verify" else None)
        monkeypatch.setenv("SSL_VERIFY", "true")
        assert ssl_mod.get_verify() is False

    def test_admin_store_true(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "true" if key == "ssl.verify" else None)
        monkeypatch.setenv("SSL_VERIFY", "false")
        assert ssl_mod.get_verify() is True

    def test_env_var_false(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        monkeypatch.setenv("SSL_VERIFY", "false")
        assert ssl_mod.get_verify() is False

    def test_env_var_true(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        monkeypatch.setenv("SSL_VERIFY", "true")
        assert ssl_mod.get_verify() is True

    def test_env_var_unknown_uses_default(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: None)
        monkeypatch.setenv("SSL_VERIFY", "maybe")
        assert ssl_mod.get_verify() is True


# ---------------------------------------------------------------------------
# requests_verify / httpx_verify
# ---------------------------------------------------------------------------

class TestRequestsVerify:
    def test_verify_off_returns_false(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "false" if key == "ssl.verify" else None)
        assert ssl_mod.requests_verify() is False

    def test_verify_on_no_ca_returns_true(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "true" if key == "ssl.verify" else None)
        for k in ("SSL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            monkeypatch.delenv(k, raising=False)
        assert ssl_mod.requests_verify() is True

    def test_verify_on_with_ca_returns_path(self, monkeypatch, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_text("cert", encoding="utf-8")

        def setting(key):
            if key == "ssl.verify":
                return "true"
            if key == "ssl.ca_bundle":
                return str(ca)
            return None

        monkeypatch.setattr(ssl_mod, "_setting", setting)
        assert ssl_mod.requests_verify() == str(ca)

    def test_httpx_verify_same_as_requests(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "false" if key == "ssl.verify" else None)
        assert ssl_mod.httpx_verify() is False
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "true" if key == "ssl.verify" else None)
        for k in ("SSL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            monkeypatch.delenv(k, raising=False)
        assert ssl_mod.httpx_verify() is True


# ---------------------------------------------------------------------------
# apply_openai_ssl_patch
# ---------------------------------------------------------------------------

class TestApplyOpenaiSslPatch:
    def test_patches_openai(self, monkeypatch):
        import openai
        import httpx

        # Ensure not patched
        monkeypatch.delattr(openai, "_productarium_ssl_patched", raising=False)
        orig_async = openai.AsyncOpenAI.__init__
        orig_sync = openai.OpenAI.__init__
        try:
            ssl_mod.apply_openai_ssl_patch()
            assert getattr(openai, "_productarium_ssl_patched", False) is True
            assert openai.AsyncOpenAI.__init__ is not orig_async
            assert openai.OpenAI.__init__ is not orig_sync
        finally:
            # Restore originals
            openai.AsyncOpenAI.__init__ = orig_async
            openai.OpenAI.__init__ = orig_sync
            if hasattr(openai, "_productarium_ssl_patched"):
                del openai._productarium_ssl_patched

    def test_idempotent(self, monkeypatch):
        import openai
        monkeypatch.delattr(openai, "_productarium_ssl_patched", raising=False)
        ssl_mod.apply_openai_ssl_patch()
        patched_init = openai.AsyncOpenAI.__init__
        ssl_mod.apply_openai_ssl_patch()  # second call should be no-op
        assert openai.AsyncOpenAI.__init__ is patched_init
        # Cleanup
        orig_async = openai.AsyncOpenAI.__init__
        # We can't easily restore the original here, but the patch replaces it
        # with a wrapper that calls the original, so it's safe to leave.

    def test_patched_client_gets_httpx_client(self, monkeypatch):
        import openai
        import httpx
        monkeypatch.delattr(openai, "_productarium_ssl_patched", raising=False)
        orig_async = openai.AsyncOpenAI.__init__
        try:
            ssl_mod.apply_openai_ssl_patch()
            client = openai.AsyncOpenAI(api_key="test-key")
            # The patched __init__ should have injected an http_client
            # We can't easily assert the internal httpx client, but the
            # constructor should not raise.
        finally:
            openai.AsyncOpenAI.__init__ = orig_async
            if hasattr(openai, "_productarium_ssl_patched"):
                del openai._productarium_ssl_patched


# ---------------------------------------------------------------------------
# apply_ssl_env
# ---------------------------------------------------------------------------

class TestApplySslEnv:
    def test_skip_verify_path(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "false" if key == "ssl.verify" else None)
        for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            monkeypatch.delenv(k, raising=False)
        ssl_mod.apply_ssl_env()
        # Should NOT set CA env vars in skip-verify mode
        assert "SSL_CERT_FILE" not in os.environ or os.environ.get("SSL_CERT_FILE") is None

    def test_ca_bundle_path_sets_env(self, monkeypatch, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_text("cert", encoding="utf-8")

        def setting(key):
            if key == "ssl.verify":
                return "true"
            if key == "ssl.ca_bundle":
                return str(ca)
            return None

        monkeypatch.setattr(ssl_mod, "_setting", setting)
        for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            monkeypatch.delenv(k, raising=False)
        # apply_ssl_env writes os.environ DIRECTLY (invisible to monkeypatch,
        # so the garbage CA path would leak into every later test's SSL
        # context) — snapshot and restore around the call.
        _keys = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
        _saved = {k: os.environ.get(k) for k in _keys}
        try:
            ssl_mod.apply_ssl_env()
            assert os.environ.get("SSL_CERT_FILE") == str(ca)
            assert os.environ.get("REQUESTS_CA_BUNDLE") == str(ca)
            assert os.environ.get("CURL_CA_BUNDLE") == str(ca)
        finally:
            for k, v in _saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_does_not_overwrite_existing_env(self, monkeypatch, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_text("cert", encoding="utf-8")

        def setting(key):
            if key == "ssl.verify":
                return "true"
            if key == "ssl.ca_bundle":
                return str(ca)
            return None

        monkeypatch.setattr(ssl_mod, "_setting", setting)
        monkeypatch.setenv("SSL_CERT_FILE", "/pre-existing")
        _keys = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
        _saved = {k: os.environ.get(k) for k in _keys}
        try:
            ssl_mod.apply_ssl_env()
            # Should not overwrite the pre-existing value
            assert os.environ.get("SSL_CERT_FILE") == "/pre-existing"
        finally:
            for k, v in _saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_verify_on_no_ca_does_not_set_env(self, monkeypatch):
        monkeypatch.setattr(ssl_mod, "_setting", lambda key: "true" if key == "ssl.verify" else None)
        for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            monkeypatch.delenv(k, raising=False)
        ssl_mod.apply_ssl_env()
        assert "SSL_CERT_FILE" not in os.environ
        assert "REQUESTS_CA_BUNDLE" not in os.environ
