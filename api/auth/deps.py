"""FastAPI auth dependencies (contract J).

- ``get_current_user``   — reads the ``productarium_session`` cookie, returns a
  ``UserORM`` (or 401). When ``AUTH_PROVIDER=none`` returns a bootstrap/system
  admin so the API stays usable without auth (dev/bootstrap). Unknown user ids
  in a valid token no longer fall back to a transient user (P0-9): the account
  must exist in the DB, otherwise 401.
- ``require_admin``      — 403 unless the current user is an admin.
- ``resolve_product_access`` — effective access of a user to a product
  ('rw' | 'ro' | None) under the P0-2 role/grant model.
- ``require_product_access`` — dependency factory enforcing that level on the
  ``product_id`` path param (404 invisible, 403 insufficient).
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
from api.models import ApiTokenORM, ProductGrantORM, ProductORM, UserORM

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
        # P0-9: a stale session token for a since-deleted (or never persisted)
        # user must NOT silently become a transient in-memory user with
        # attacker-controllable token claims — reject with 401.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
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


def resolve_product_access(
    user: UserORM,
    product: ProductORM,
    grant: Optional[ProductGrantORM] = None,
) -> Optional[str]:
    """Effective access level of ``user`` to ``product``: 'rw' | 'ro' | None.

    Role model (P0-2):

    - ``admin``         — 'rw' on every product.
    - ``viewer_global`` — 'ro' on every product.
    - ``manager``       — 'rw' on own products; grant level otherwise; 'ro'
      baseline on non-owned products (managers see everything read-only).
    - ``user``          — 'rw' on own products; grant level otherwise; no access
      without a grant.
    """
    if user.role == "admin":
        return "rw"
    if user.role == "viewer_global":
        return "ro"
    grant_level = grant.level if grant is not None else None
    if user.role == "manager":
        if product.owner_id == user.id:
            return "rw"
        if grant_level == "rw":
            return "rw"
        return "ro"
    # plain 'user' (and any unknown legacy role value — safest default)
    if product.owner_id == user.id:
        return "rw"
    return grant_level


def get_product_grant(db: Session, product_id: str, user_id: str) -> Optional[ProductGrantORM]:
    """Load the per-product grant row for a user (composite PK lookup)."""
    return db.get(ProductGrantORM, {"product_id": product_id, "user_id": user_id})


def require_product_access(level: str = "ro"):
    """Dependency factory enforcing product visibility/access on endpoints.

    Usage: ``product: ProductORM = Depends(require_product_access("rw"))`` —
    the endpoint must declare a ``product_id`` path parameter. Behaviour:

    - 401 when not authenticated;
    - 404 when the product does not exist OR is invisible to the caller
      (indistinguishable on purpose — no existence leak);
    - 403 when visible but the caller lacks ``level`` ('ro' < 'rw').
    """

    def _dependency(
        product_id: str,
        user: UserORM = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> ProductORM:
        product = db.get(ProductORM, product_id)
        grant = get_product_grant(db, product_id, user.id)
        effective = resolve_product_access(user, product, grant) if product else None
        if product is None or effective is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found",
            )
        if level == "rw" and effective != "rw":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Write access required for this product",
            )
        return product

    return _dependency


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
