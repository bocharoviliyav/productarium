"""Local password hashing/verification (bcrypt).

Uses the ``bcrypt`` library directly to produce $2b$ hashes. ``passlib[bcrypt]``
is kept as a declared dependency (it brings bcrypt), but passlib 1.7.4 is
incompatible with bcrypt >= 4.1 on Python 3.12 (bcrypt removed ``__about__``
and changed ``hashpw`` behavior, raising ``ValueError`` even for short
passwords), so we call bcrypt directly for robustness. Behavior is equivalent:
bcrypt $2b$ hashes that passlib-verified systems can still read.
"""

from __future__ import annotations

import hashlib
import logging
import secrets as _secrets

logger = logging.getLogger(__name__)

try:
    import bcrypt as _bcrypt  # type: ignore
    _BCRYPT_AVAILABLE = True
except Exception as _e:  # pragma: no cover - dep missing
    _bcrypt = None  # type: ignore
    _BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not available; local password auth disabled: %s", _e)


def is_available() -> bool:
    return _BCRYPT_AVAILABLE


def hash_password(password: str) -> str:
    """Return a bcrypt $2b$ hash, or "" if bcrypt is unavailable / password empty."""
    if not _BCRYPT_AVAILABLE or not password:
        return ""
    try:
        return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")
    except Exception as e:
        logger.error("bcrypt hash failed: %s", e)
        return ""


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time verify a password against a bcrypt hash. False on any error."""
    if not _BCRYPT_AVAILABLE or not password_hash or not password:
        return False
    try:
        return _bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# --- Reset / temp-password tokens -------------------------------------------
# Reset tokens are opaque url-safe strings. We store only their sha256 hash in
# the DB (never the raw token) so a DB leak doesn't expose live reset tokens.
# The raw token is shown once to the admin (or the first-run setup flow) and is
# later supplied by the user to POST /api/auth/reset-password.
RESET_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


def generate_reset_token() -> str:
    """Return a fresh opaque reset token (url-safe, 32 bytes)."""
    return _secrets.token_urlsafe(32)


def hash_token(raw: str) -> str:
    """sha256 hash of a reset/temp token for safe DB storage."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
