from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from api.auth import keycloak


class TestIsConfigured:
    def test_false_when_no_authlib(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", False)
        monkeypatch.setenv("KEYCLOAK_URL", "http://localhost:8080")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "test-client")
        assert keycloak.is_configured() is False

    def test_false_when_no_url(self, monkeypatch):
        # _cfg() defaults KEYCLOAK_URL to "http://localhost:8080", so deleting
        # the env var is NOT enough; patch _cfg to return an empty url.
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setattr(keycloak, "_cfg", lambda: {
            "url": "",
            "client_id": "test-client",
            "client_secret": "",
            "realm": "productarium",
        })
        assert keycloak.is_configured() is False

    def test_false_when_no_client_id(self, monkeypatch):
        # _cfg() defaults KEYCLOAK_CLIENT_ID to "productarium-frontend", so
        # patch _cfg to return an empty client_id.
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setattr(keycloak, "_cfg", lambda: {
            "url": "http://localhost:8080",
            "client_id": "",
            "client_secret": "",
            "realm": "productarium",
        })
        assert keycloak.is_configured() is False

    def test_true_with_url_and_client_id(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setenv("KEYCLOAK_URL", "http://localhost:8080")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "test-client")
        assert keycloak.is_configured() is True

    def test_true_without_secret(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setenv("KEYCLOAK_URL", "http://localhost:8080")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "test-client")
        monkeypatch.delenv("KEYCLOAK_CLIENT_SECRET", raising=False)
        assert keycloak.is_configured() is True


class TestCfg:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("KEYCLOAK_URL", raising=False)
        monkeypatch.delenv("KEYCLOAK_CLIENT_ID", raising=False)
        monkeypatch.delenv("KEYCLOAK_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("KEYCLOAK_REALM", raising=False)
        c = keycloak._cfg()
        assert c["url"] == "http://localhost:8080"
        assert c["client_id"] == "productarium-frontend"
        assert c["client_secret"] == ""
        assert c["realm"] == "productarium"

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com/")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "myapp")
        monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "secret")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")
        c = keycloak._cfg()
        assert c["url"] == "http://kc.example.com"
        assert c["client_id"] == "myapp"
        assert c["client_secret"] == "secret"
        assert c["realm"] == "myrealm"


class TestRealmUrl:
    def test_realm_url(self, monkeypatch):
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")
        assert keycloak._realm_url() == "http://kc.example.com/realms/myrealm"


class TestPkceHelpers:
    def test_new_code_verifier_length(self):
        v = keycloak.new_code_verifier()
        assert isinstance(v, str)
        assert 43 <= len(v) <= 128

    def test_new_code_verifier_unique(self):
        assert keycloak.new_code_verifier() != keycloak.new_code_verifier()

    def test_code_challenge_s256(self):
        import base64
        import hashlib

        verifier = "test-verifier-1234567890"
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        assert keycloak._code_challenge(verifier) == expected

    def test_new_state(self):
        s = keycloak.new_state()
        assert isinstance(s, str) and len(s) > 5

    def test_new_state_unique(self):
        assert keycloak.new_state() != keycloak.new_state()


class TestGetAuthorizeUrl:
    def test_builds_url_with_pkce(self, monkeypatch):
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "myapp")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")
        url = keycloak.get_authorize_url(
            "http://localhost:3000/callback", "mystate", "myverifier"
        )
        assert url.startswith("http://kc.example.com/realms/myrealm/protocol/openid-connect/auth?")
        assert "client_id=myapp" in url
        assert "response_type=code" in url
        assert "state=mystate" in url
        assert "code_challenge_method=S256" in url
        assert "redirect_uri=http" in url
        assert "code_challenge=" in url


class TestExchangeCode:
    def test_returns_none_when_no_authlib(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", False)
        assert keycloak.exchange_code("code", "http://callback") is None

    def test_returns_token_dict_on_success(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "myapp")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")

        mock_session = MagicMock()
        mock_session.fetch_token.return_value = {"access_token": "tok123", "token_type": "Bearer"}
        with patch.object(keycloak, "OAuth2Session", return_value=mock_session):
            result = keycloak.exchange_code("code", "http://callback", code_verifier="verifier")
        assert result == {"access_token": "tok123", "token_type": "Bearer"}
        mock_session.fetch_token.assert_called_once()

    def test_returns_none_on_exception(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "myapp")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")

        mock_session = MagicMock()
        mock_session.fetch_token.side_effect = Exception("network error")
        with patch.object(keycloak, "OAuth2Session", return_value=mock_session):
            assert keycloak.exchange_code("code", "http://callback") is None

    def test_passes_client_secret_for_confidential_client(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "myapp")
        monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "mysecret")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")

        mock_session = MagicMock()
        mock_session.fetch_token.return_value = {"access_token": "tok"}
        with patch.object(keycloak, "OAuth2Session", return_value=mock_session) as mock_oa2:
            keycloak.exchange_code("code", "http://callback")
        # OAuth2Session is called as OAuth2Session(client_id, client_secret, scope=...)
        # so client_secret is the 2nd positional arg (index 1) of args.
        args, kwargs = mock_oa2.call_args
        assert args[1] == "mysecret"


class TestFetchUserinfo:
    def test_returns_none_when_no_authlib(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", False)
        assert keycloak.fetch_userinfo("token") is None

    def test_returns_none_when_no_token(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        assert keycloak.fetch_userinfo("") is None

    def test_returns_userinfo_on_success(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "myapp")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sub": "user123", "preferred_username": "alice"}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        with patch.object(keycloak, "OAuth2Session", return_value=mock_session):
            result = keycloak.fetch_userinfo("access_tok")
        assert result == {"sub": "user123", "preferred_username": "alice"}

    def test_returns_none_on_non_200(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "myapp")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        with patch.object(keycloak, "OAuth2Session", return_value=mock_session):
            assert keycloak.fetch_userinfo("bad_tok") is None

    def test_returns_none_on_exception(self, monkeypatch):
        monkeypatch.setattr(keycloak, "AUTHLIB_AVAILABLE", True)
        monkeypatch.setenv("KEYCLOAK_URL", "http://kc.example.com")
        monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "myapp")
        monkeypatch.setenv("KEYCLOAK_REALM", "myrealm")

        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("connection refused")
        with patch.object(keycloak, "OAuth2Session", return_value=mock_session):
            assert keycloak.fetch_userinfo("tok") is None
