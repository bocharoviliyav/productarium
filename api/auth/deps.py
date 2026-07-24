"""FastAPI auth dependencies (contract J).

- ``get_current_user``   — reads the ``productarium_session`` cookie, returns a
  ``UserORM`` (or 401). When ``AUTH_PROVIDER=none`` returns a bootstrap/system
  admin so the API stays usable without auth (dev/bootstrap).
- ``require_admin``      — 403 unless the current user is an admin.
- ``require_api_token``  — validates a ``Bearer`` API token (public API),
  updates ``last_used_at``.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.auth import AUTH_PROVIDER
from api.auth.tokens import SESSION_COOKIE_NAME, verify_session_token
from api.db import get_db
from api.models import ApiTokenORM, UserORM

logger = logging.getLogger(__name__)

# A stable bootstrap/system user returned when AUTH_PROVIDER=none so endpoints
# that depend on get_current_user still work without auth.
_SYSTEM_USER: Optional[UserORM] = None


def _system_user() -> UserORM:
    global _SYSTEM_USER
    if _SYSTEM_USER is None:
        _SYSTEM_USER = UserORM(
            id="system",
            username="system",
            role="admin",
            provider="local",
            created_at=datetime.utcnow(),
        )
    return _SYSTEM_USER


def get_current_user(request: Request, db: Session = Depends(get_db)) -> UserORM:
    """Resolve the current user from the session cookie.

    With ``AUTH_PROVIDER=none`` returns a bootstrap/system admin user. Otherwise
    requires a valid ``productarium_session`` cookie -> 401 if missing/invalid.
    """
    if AUTH_PROVIDER == "none":
        return _system_user()
    token = request.cookies.get(SESSION_COOKIE_NAME)
    payload = verify_session_token(token) if token else None
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    user_id = payload.get("sub")
    user = db.get(UserORM, user_id) if user_id else None
    if user is None:
        # Fall back to a transient user from the token claims (e.g. a Keycloak
        # user not yet persisted). Role is taken from the token.
        user = UserORM(
            id=user_id or "unknown",
            username=payload.get("username", "unknown"),
            role=payload.get("role", "user"),
            provider="local",
            created_at=datetime.utcnow(),
        )
    return user


def require_admin(user: UserORM = Depends(get_current_user)) -> UserORM:
    """403 unless the current user is an admin."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def require_api_token(request: Request, db: Session = Depends(get_db)) -> ApiTokenORM:
    """Validate a Bearer API token (public API). Updates last_used_at."""
    auth = request.headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    raw = auth.split(" ", 1)[1].strip()
    if not raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API token")
    token_hash = _hash_token(raw)
    tok = db.query(ApiTokenORM).filter(ApiTokenORM.token_hash == token_hash).first()
    if tok is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")
    tok.last_used_at = datetime.utcnow()
    try:
        db.commit()
    except Exception:
        db.rollback()
    return tok
