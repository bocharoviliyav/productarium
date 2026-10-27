"""Keycloak OIDC integration via authlib (PKCE-aware for public clients).

Implements: authorize URL building (with PKCE S256), authorization-code ->
token exchange, and userinfo retrieval. Configuration is read from env
(KEYCLOAK_URL, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET, KEYCLOAK_REALM).

Two modes:
- **Public client** (default, ``KEYCLOAK_CLIENT_ID=productarium-frontend``):
  uses PKCE (code_verifier / code_challenge S256); no client_secret required.
- **Confidential client**: when ``KEYCLOAK_CLIENT_SECRET`` is set, the secret
  is sent in the token exchange (legacy / server-side client behaviour).

If ``authlib`` is not installed, ``AUTHLIB_AVAILABLE`` is False and the auth
router returns 501 for the Keycloak endpoints.

Admin REST API (SPEC-001): a separate confidential service-account client
(``KEYCLOAK_ADMIN_CLIENT_ID`` + ``KEYCLOAK_ADMIN_CLIENT_SECRET``) lists realm
groups for the admin roles↔groups mapping reference; it never handles user
sessions. Unconfigured -> the admin endpoint reports 501.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from typing import List, Optional

import requests

from api.config.timeout import resolve_keycloak_admin_timeout

logger = logging.getLogger(__name__)

try:
    from authlib.integrations.requests_client import OAuth2Session  # type: ignore
    AUTHLIB_AVAILABLE = True
except Exception as _e:  # pragma: no cover - dep missing
    AUTHLIB_AVAILABLE = False
    OAuth2Session = None  # type: ignore
    logger.info("authlib not available; Keycloak login will return 501: %s", _e)


def _cfg() -> dict:
    """Read Keycloak config from env with productarium defaults.

    Defaults target the bundled docker-compose Keycloak (realm=productarium,
    public client ``productarium-frontend``) so that ``is_configured()`` is
    True out-of-the-box when KEYCLOAK_URL is set.
    """
    return {
        "url": os.environ.get("KEYCLOAK_URL", "http://localhost:8080").rstrip("/"),
        "client_id": os.environ.get("KEYCLOAK_CLIENT_ID", "productarium-frontend"),
        "client_secret": os.environ.get("KEYCLOAK_CLIENT_SECRET", ""),
        "realm": os.environ.get("KEYCLOAK_REALM", "productarium"),
    }


def is_configured() -> bool:
    """True if authlib is available and KEYCLOAK_URL + KEYCLOAK_CLIENT_ID are set.

    For a public client (PKCE) the client_secret is NOT required. A non-empty
    secret enables confidential-client mode.
    """
    c = _cfg()
    return bool(AUTHLIB_AVAILABLE and c["url"] and c["client_id"])


def _realm_url() -> str:
    c = _cfg()
    return f"{c['url']}/realms/{c['realm']}"


# --- PKCE helpers ------------------------------------------------------------
def new_code_verifier() -> str:
    """Generate a high-entropy PKCE code_verifier (43-128 chars, url-safe)."""
    return secrets.token_urlsafe(64)


def _code_challenge(verifier: str) -> str:
    """S256 code_challenge = base64url(sha256(verifier)) without padding."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def get_authorize_url(redirect_uri: str, state: str, code_verifier: str) -> str:
    """Build the OIDC authorization endpoint URL with PKCE S256 challenge."""
    c = _cfg()
    authorize = f"{_realm_url()}/protocol/openid-connect/auth"
    from urllib.parse import urlencode
    params = {
        "response_type": "code",
        "client_id": c["client_id"],
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": "openid profile email",
        "code_challenge": _code_challenge(code_verifier),
        "code_challenge_method": "S256",
    }
    return f"{authorize}?{urlencode(params)}"


def exchange_code(
    code: str, redirect_uri: str, code_verifier: Optional[str] = None
) -> Optional[dict]:
    """Exchange an authorization code for tokens.

    For a public client (no ``client_secret``) PKCE is mandatory: pass the
    ``code_verifier`` that was used to build the challenge in
    :func:`get_authorize_url`. For a confidential client the ``client_secret``
    is sent instead and ``code_verifier`` is optional.
    """
    if not AUTHLIB_AVAILABLE:
        return None
    c = _cfg()
    token_url = f"{_realm_url()}/protocol/openid-connect/token"
    # authlib OAuth2Session accepts client_id, client_secret (None for public).
    sess = OAuth2Session(
        c["client_id"],
        c["client_secret"] or None,
        scope="openid profile email",
    )
    try:
        kwargs = {
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        if code_verifier:
            kwargs["code_verifier"] = code_verifier
        token = sess.fetch_token(token_url, **kwargs)
        return dict(token) if token else None
    except Exception as e:
        logger.warning("Keycloak token exchange error: %s", e)
        return None


def fetch_userinfo(access_token: str) -> Optional[dict]:
    """Fetch OIDC userinfo for an access_token. Returns dict or None."""
    if not AUTHLIB_AVAILABLE or not access_token:
        return None
    userinfo_url = f"{_realm_url()}/protocol/openid-connect/userinfo"
    sess = OAuth2Session(
        _cfg()["client_id"], token={"access_token": access_token, "token_type": "Bearer"}
    )
    try:
        r = sess.get(userinfo_url)
        if r.status_code == 200:
            return r.json()
        logger.warning("Keycloak userinfo failed: %s %s", r.status_code, r.text)
        return None
    except Exception as e:
        logger.warning("Keycloak userinfo error: %s", e)
        return None


def new_state() -> str:
    """Generate a random OAuth state token."""
    return secrets.token_urlsafe(16)


# --- Admin REST API (SPEC-001 groups reference) -----------------------------
def _admin_cfg() -> dict:
    """Service-account client config (client credentials grant)."""
    return {
        "url": os.environ.get("KEYCLOAK_URL", "http://localhost:8080").rstrip("/"),
        "client_id": os.environ.get("KEYCLOAK_ADMIN_CLIENT_ID", ""),
        "client_secret": os.environ.get("KEYCLOAK_ADMIN_CLIENT_SECRET", ""),
        "realm": os.environ.get("KEYCLOAK_REALM", "productarium"),
    }


def admin_configured() -> bool:
    """True when the Admin API service-account client id + secret are set."""
    c = _admin_cfg()
    return bool(c["client_id"] and c["client_secret"])


def _service_account_token() -> Optional[str]:
    """Client-credentials access token for the Admin API (or None)."""
    c = _admin_cfg()
    try:
        r = requests.post(
            f"{c['url']}/realms/{c['realm']}/protocol/openid-connect/token",
            data={
                "grant_type": "client_credentials",
                "client_id": c["client_id"],
                "client_secret": c["client_secret"],
            },
            timeout=resolve_keycloak_admin_timeout(),
        )
        if r.status_code == 200:
            return r.json().get("access_token")
        logger.warning("Keycloak admin token fetch failed: HTTP %s", r.status_code)
    except Exception as e:
        logger.warning("Keycloak admin token fetch error: %s", e)
    return None


def list_groups() -> Optional[List[dict]]:
    """Top-level realm groups ``[{id, name, path}]`` for the mapping reference.

    None = unconfigured/upstream failure (the endpoint degrades to 502);
    names are simple (no leading slash), matching the ``groups`` claim mapper
    with ``full.path=false`` that role mapping compares against.
    """
    token = _service_account_token()
    if token is None:
        return None
    c = _admin_cfg()
    try:
        r = requests.get(
            f"{c['url']}/admin/realms/{c['realm']}/groups",
            headers={"Authorization": f"Bearer {token}"},
            params={"briefRepresentation": "true"},
            timeout=resolve_keycloak_admin_timeout(),
        )
        if r.status_code == 200:
            return [
                {"id": g.get("id"), "name": g.get("name"), "path": g.get("path")}
                for g in r.json()
                if isinstance(g, dict) and g.get("name")
            ]
        logger.warning("Keycloak admin groups fetch failed: HTTP %s", r.status_code)
    except Exception as e:
        logger.warning("Keycloak admin groups fetch error: %s", e)
    return None
