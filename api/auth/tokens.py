"""Issue/verify Productarium session JWTs (httpOnly cookie ``productarium_session``).

The JWT is HS256-signed with ``JWT_SECRET_KEY`` (or ``SETTINGS_SECRET_KEY``, or
an ephemeral dev secret if both are unset — logged as a warning). Claims:
``sub`` (user id), ``username``, ``role``, ``iat``, ``exp``.
"""

from __future__ import annotations

import logging
import os
import secrets as _secrets
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import jwt as _jwt  # type: ignore
    _PYJWT_AVAILABLE = True
except Exception as _e:  # pragma: no cover - dep missing
    _jwt = None  # type: ignore
    _PYJWT_AVAILABLE = False
    logger.warning("PyJWT not available; session tokens disabled: %s", _e)

SESSION_COOKIE_NAME = "productarium_session"
SESSION_TOKEN_TTL = int(os.environ.get("SESSION_TOKEN_TTL", str(60 * 60 * 24 * 7)))  # 7d

# Ephemeral fallback secret (stable per process) used when no env secret is set.
_FALLBACK_SECRET = _secrets.token_urlsafe(32)


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET_KEY") or os.environ.get("SETTINGS_SECRET_KEY")
    if not secret:
        logger.warning(
            "JWT_SECRET_KEY/SETTINGS_SECRET_KEY unset; using an ephemeral dev "
            "session secret (sessions will not survive restart)."
        )
        return _FALLBACK_SECRET
    return secret


def create_session_token(user) -> str:
    """Issue a signed session JWT for a UserORM (or duck-typed object)."""
    if not _PYJWT_AVAILABLE:
        return ""
    now = int(time.time())
    payload = {
        "sub": getattr(user, "id", ""),
        "username": getattr(user, "username", ""),
        "role": getattr(user, "role", "user"),
        "iat": now,
        "exp": now + SESSION_TOKEN_TTL,
    }
    return _jwt.encode(payload, _secret(), algorithm="HS256")


def verify_session_token(token: str) -> Optional[dict]:
    """Verify a session JWT; return the claims dict or None on any failure."""
    if not _PYJWT_AVAILABLE or not token:
        return None
    try:
        return _jwt.decode(token, _secret(), algorithms=["HS256"])
    except Exception as e:
        logger.debug("session token verify failed: %s", e)
        return None
