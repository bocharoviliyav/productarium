"""Auth router (contract J): local login, me, logout; Keycloak login/callback.

Endpoints (prefix ``/api/auth``):
- ``POST /login``             — local username/password login, sets session cookie
- ``GET  /me``                — current user (requires session)
- ``POST /logout``            — clears the session cookie
- ``GET  /keycloak/login``    — redirects to Keycloak authorize URL (501 if
  Keycloak/authlib not configured)
- ``GET  /keycloak/callback`` — OIDC code -> session cookie (501 if unconfigured)

Local login + me + logout are fully implemented. Keycloak endpoints are
functional when ``authlib`` is installed and ``KEYCLOAK_*`` env is set, else
they return 501 with a clear message (Keycloak is configured separately).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.auth import AUTH_PROVIDER
from api.auth.deps import get_current_user
from api.auth.keycloak import (
    exchange_code,
    fetch_userinfo,
    get_authorize_url,
    is_configured as keycloak_is_configured,
    new_code_verifier,
    new_state,
)
from api.auth.local import (
    RESET_TOKEN_TTL_SECONDS,
    generate_reset_token,
    hash_password,
    hash_token,
    verify_password,
)
from api.auth.tokens import (
    SESSION_COOKIE_NAME,
    SESSION_TOKEN_TTL,
    create_session_token,
)
from api.db import SessionLocal, get_db
from api.models import UserORM
from api.schemas import (
    ChangePasswordRequest,
    LoginRequest,
    ResetPasswordRequest,
    SetupRequest,
    SetupStatus,
    UserOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Cookies are httpOnly + SameSite=Lax. Set secure=True behind HTTPS in prod via
# an env flag if needed.
_COOKIE_KWARGS = {"httponly": True, "samesite": "lax", "secure": False, "path": "/"}


def _user_out(user: UserORM) -> UserOut:
    return UserOut(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        provider=user.provider,
        created_at=user.created_at,
        must_change_password=getattr(user, "must_change_password", False),
    )


@router.post("/login", response_model=UserOut)
def login(body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    """Local username/password login. Sets the ``productarium_session`` cookie."""
    if AUTH_PROVIDER == "none":
        return UserOut(id="system", username="system", role="admin", provider="local")
    user = db.query(UserORM).filter(UserORM.username == body.username).first()
    if user is None or not verify_password(body.password, user.password_hash or ""):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_session_token(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME, value=token, max_age=SESSION_TOKEN_TTL, **_COOKIE_KWARGS
    )
    return _user_out(user)


@router.get("/me", response_model=UserOut)
def me(user: UserORM = Depends(get_current_user)):
    """Return the current authenticated user."""
    return _user_out(user)


@router.post("/logout")
def logout(response: Response):
    """Clear the session cookie."""
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"message": "Logged out"}


# --- First-run admin setup ---------------------------------------------------
def _local_user_count(db: Session) -> int:
    """Count local users in productarium_users (0 on a fresh install)."""
    try:
        return db.query(UserORM).filter(UserORM.provider == "local").count()
    except Exception as e:  # pragma: no cover - defensive (table missing etc.)
        logger.warning("setup-status: could not count local users: %s", e)
        return 0


@router.get("/setup-status", response_model=SetupStatus)
def setup_status(db: Session = Depends(get_db)):
    """Tell the UI whether the first-run admin setup flow is needed.

    ``setup_required`` is True only when local auth is enabled
    (``AUTH_PROVIDER`` in ``local``/``both``) AND no local users exist yet.
    """
    if AUTH_PROVIDER == "none":
        return SetupStatus(setup_required=False, auth_provider=AUTH_PROVIDER)
    if AUTH_PROVIDER not in ("local", "both"):
        return SetupStatus(setup_required=False, auth_provider=AUTH_PROVIDER)
    return SetupStatus(
        setup_required=_local_user_count(db) == 0,
        auth_provider=AUTH_PROVIDER,
    )


@router.post("/setup", response_model=UserOut)
def setup(body: SetupRequest, response: Response, db: Session = Depends(get_db)):
    """Create the first admin user (only allowed when no local users exist).

    Sets the session cookie so the caller is signed in as the new admin.
    """
    if AUTH_PROVIDER == "none":
        raise HTTPException(status_code=400, detail="Auth disabled (AUTH_PROVIDER=none).")
    if AUTH_PROVIDER not in ("local", "both"):
        raise HTTPException(
            status_code=400,
            detail=f"Local setup unavailable with AUTH_PROVIDER={AUTH_PROVIDER}.",
        )
    if _local_user_count(db) > 0:
        raise HTTPException(
            status_code=409,
            detail="Setup already complete: an admin exists. Use login instead.",
        )
    if not body.username.strip() or not body.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    existing = db.query(UserORM).filter(UserORM.username == body.username.strip()).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username already taken.")
    user = UserORM(
        id=f"user_{uuid.uuid4().hex[:24]}",
        username=body.username.strip(),
        email=body.email or None,
        password_hash=hash_password(body.password),
        role="admin",
        provider="local",
        must_change_password=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.warning("Setup: created first admin user %r.", user.username)
    token = create_session_token(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME, value=token, max_age=SESSION_TOKEN_TTL, **_COOKIE_KWARGS
    )
    return _user_out(user)


# --- Password change / reset -------------------------------------------------
@router.post("/change-password")
def change_password(
    body: ChangePasswordRequest,
    user: UserORM = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Authenticated password change (old -> new). Clears ``must_change_password``."""
    if user.provider != "local" or not user.id or user.id == "system":
        raise HTTPException(status_code=400, detail="Only local users can change a password here.")
    # ``user`` from get_current_user may be a transient (token-only) object; load
    # the persisted row so we can mutate + commit it.
    db_user = db.get(UserORM, user.id) if user.id else None
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if not verify_password(body.old_password, db_user.password_hash or ""):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    if not body.new_password:
        raise HTTPException(status_code=400, detail="New password is required.")
    db_user.password_hash = hash_password(body.new_password)
    db_user.must_change_password = False
    # Invalidate any outstanding reset token when the password changes.
    db_user.reset_token_hash = None
    db_user.reset_token_expires = None
    db.commit()
    return {"message": "Password changed."}


@router.post("/reset-password")
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Public password reset via a one-time reset token.

    Validates the token (sha256 hash match + not expired), sets the new
    password, and clears the reset token + ``must_change_password``.
    """
    if not body.token or not body.new_password:
        raise HTTPException(status_code=400, detail="Token and new password are required.")
    token_hash = hash_token(body.token)
    user = (
        db.query(UserORM)
        .filter(UserORM.reset_token_hash == token_hash)
        .first()
    )
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or unknown reset token.")
    if user.reset_token_expires is not None and user.reset_token_expires < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Reset token has expired.")
    user.password_hash = hash_password(body.new_password)
    user.reset_token_hash = None
    user.reset_token_expires = None
    user.must_change_password = False
    db.commit()
    return {"message": "Password reset. You can now sign in."}


@router.get("/keycloak/login")
def keycloak_login(request: Request):
    """Redirect to the Keycloak OIDC authorize endpoint. 501 if unconfigured.

    Generates a PKCE ``code_verifier`` and ``state`` and stores both in
    short-lived httpOnly cookies so the callback can complete the exchange.
    Works for public clients (no client_secret) and confidential clients.
    """
    if not keycloak_is_configured():
        raise HTTPException(
            status_code=501,
            detail="Keycloak not configured (set KEYCLOAK_URL + KEYCLOAK_CLIENT_ID "
            "and install authlib; KEYCLOAK_CLIENT_SECRET is optional for public "
            "PKCE clients).",
        )
    redirect_uri = str(request.url_for("keycloak_callback"))
    state = new_state()
    code_verifier = new_code_verifier()
    url = get_authorize_url(redirect_uri, state, code_verifier)
    resp = RedirectResponse(url)
    # Carry the state + PKCE verifier in short-lived httpOnly cookies.
    resp.set_cookie(
        "productarium_oauth_state", state, max_age=600, **_COOKIE_KWARGS
    )
    resp.set_cookie(
        "productarium_pkce_verifier", code_verifier, max_age=600, **_COOKIE_KWARGS
    )
    return resp


@router.get("/keycloak/callback", name="keycloak_callback")
def keycloak_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    """OIDC callback: exchange code (with PKCE verifier) -> userinfo -> session.

    Reads the PKCE ``code_verifier`` and ``state`` from the cookies set by
    ``keycloak_login``. 501 if Keycloak is unconfigured.
    """
    if not keycloak_is_configured():
        raise HTTPException(status_code=501, detail="Keycloak not configured.")
    if error:
        raise HTTPException(status_code=400, detail=f"Keycloak error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
    # Validate state round-trip (defensive; mismatch -> 400).
    cookie_state = request.cookies.get("productarium_oauth_state")
    if cookie_state and state and cookie_state != state:
        raise HTTPException(status_code=400, detail="OAuth state mismatch")
    code_verifier = request.cookies.get("productarium_pkce_verifier")
    redirect_uri = str(request.url_for("keycloak_callback"))
    tokens = exchange_code(code, redirect_uri, code_verifier=code_verifier)
    if not tokens:
        raise HTTPException(status_code=400, detail="Keycloak code exchange failed")
    access_token = tokens.get("access_token")
    userinfo = fetch_userinfo(access_token) if access_token else None
    if not userinfo:
        raise HTTPException(status_code=400, detail="Keycloak userinfo failed")
    sub = userinfo.get("sub") or userinfo.get("id")
    username = userinfo.get("preferred_username") or userinfo.get("username") or sub
    email = userinfo.get("email")
    with SessionLocal() as db:
        user = (
            db.query(UserORM).filter(UserORM.provider_subject == sub).first()
            if sub
            else None
        )
        if user is None:
            user = UserORM(
                id=f"user_{uuid.uuid4().hex[:24]}",
                username=username or sub or "keycloak_user",
                email=email,
                role="user",
                provider="keycloak",
                provider_subject=sub,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        token = create_session_token(user)
    resp = RedirectResponse("/")
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TOKEN_TTL,
        **_COOKIE_KWARGS,
    )
    # Clean up the PKCE + state cookies now that the flow is complete.
    resp.delete_cookie("productarium_pkce_verifier", path="/")
    resp.delete_cookie("productarium_oauth_state", path="/")
    return resp
