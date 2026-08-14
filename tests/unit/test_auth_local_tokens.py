from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from api.auth.local import (
    RESET_TOKEN_TTL_SECONDS,
    generate_reset_token,
    hash_password,
    hash_token,
    is_available,
    verify_password,
)
from api.auth.tokens import (
    SESSION_COOKIE_NAME,
    SESSION_TOKEN_TTL,
    create_session_token,
    verify_session_token,
)


# --------------------------------------------------------------------------- #
# api.auth.local
# --------------------------------------------------------------------------- #
class TestLocalAvailability:
    def test_is_available(self):
        assert is_available() is True


class TestHashPassword:
    def test_returns_bcrypt_hash(self):
        h = hash_password("secret123")
        assert h.startswith("$2") and len(h) > 20

    def test_empty_password_returns_empty(self):
        assert hash_password("") == ""

    def test_none_password_returns_empty(self):
        assert hash_password(None) == ""  # type: ignore[arg-type]


class TestVerifyPassword:
    def test_correct_password(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_wrong_password(self):
        h = hash_password("mypassword")
        assert verify_password("wrong", h) is False

    def test_empty_hash(self):
        assert verify_password("x", "") is False

    def test_empty_password(self):
        h = hash_password("x")
        assert verify_password("", h) is False

    def test_none_inputs(self):
        assert verify_password(None, None) is False  # type: ignore[arg-type]

    def test_garbage_hash(self):
        assert verify_password("x", "not-a-real-hash") is False


class TestResetTokens:
    def test_generate_returns_urlsafe_string(self):
        t = generate_reset_token()
        assert isinstance(t, str) and len(t) > 10

    def test_generate_unique(self):
        assert generate_reset_token() != generate_reset_token()

    def test_hash_token_is_sha256_hex(self):
        h = hash_token("abc123")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_token_deterministic(self):
        assert hash_token("xyz") == hash_token("xyz")

    def test_hash_token_different_inputs(self):
        assert hash_token("a") != hash_token("b")

    def test_ttl_is_7_days(self):
        assert RESET_TOKEN_TTL_SECONDS == 60 * 60 * 24 * 7


# --------------------------------------------------------------------------- #
# api.auth.tokens
# --------------------------------------------------------------------------- #
class _FakeUser:
    def __init__(self, uid="user_1", username="alice", role="admin"):
        self.id = uid
        self.username = username
        self.role = role


class TestSessionTokenConstants:
    def test_cookie_name(self):
        assert SESSION_COOKIE_NAME == "productarium_session"

    def test_token_ttl(self):
        assert SESSION_TOKEN_TTL == 60 * 60 * 24 * 7


class TestCreateSessionToken:
    def test_returns_non_empty_string(self):
        t = create_session_token(_FakeUser())
        assert isinstance(t, str) and len(t) > 10

    def test_contains_user_claims(self):
        import jwt as _jwt

        t = create_session_token(_FakeUser("user_42", "bob", "user"))
        claims = _jwt.decode(t, _secret(), algorithms=["HS256"])
        assert claims["sub"] == "user_42"
        assert claims["username"] == "bob"
        assert claims["role"] == "user"


class TestVerifySessionToken:
    def test_valid_token_roundtrip(self):
        t = create_session_token(_FakeUser("user_2", "bob", "user"))
        claims = verify_session_token(t)
        assert claims is not None
        assert claims["sub"] == "user_2"
        assert claims["username"] == "bob"
        assert claims["role"] == "user"
        assert "exp" in claims and "iat" in claims

    def test_empty_token(self):
        assert verify_session_token("") is None

    def test_none_token(self):
        assert verify_session_token(None) is None  # type: ignore[arg-type]

    def test_invalid_token(self):
        assert verify_session_token("not.a.jwt") is None

    def test_expired_token(self):
        import jwt as _jwt

        now = int(time.time())
        payload = {
            "sub": "u",
            "username": "u",
            "role": "user",
            "iat": now - 200,
            "exp": now - 100,
        }
        expired = _jwt.encode(payload, _secret(), algorithm="HS256")
        assert verify_session_token(expired) is None

    def test_wrong_secret(self):
        import jwt as _jwt

        now = int(time.time())
        payload = {
            "sub": "u",
            "username": "u",
            "role": "user",
            "iat": now,
            "exp": now + 3600,
        }
        token = _jwt.encode(payload, "wrong-secret", algorithm="HS256")
        assert verify_session_token(token) is None


def _secret() -> str:
    from api.auth.tokens import _secret as s

    return s()
