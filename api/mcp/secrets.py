"""Secrets handling for MCP server configs (Wave C).

``McpServerORM.headers`` (http auth headers) and ``McpServerORM.env`` (stdio
environment variables) hold credentials, so they are:

- **encrypted at rest** with the shared Fernet key (``SETTINGS_SECRET_KEY``,
  same key + precedence as ``api.config.settings`` — env > persisted key file
  > ephemeral dev key). The stored value is the Fernet ciphertext of the JSON
  dict (stored as a plain JSON string in the JSON column).
- **never returned to API clients**: responses expose a masked key-only view
  (``{"Authorization": "***"}``) via :func:`mask_secret_dict`.

If ``cryptography`` is unavailable (or the Fernet key cannot be built), the
values are stored as plain JSON with a logged warning — mirroring the
``set_setting(encrypt=True)`` degradation in ``api.config.settings``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Masked placeholder shown instead of every secret VALUE (keys stay visible).
MASK = "***"


def _fernet():
    """The shared Fernet instance from the settings store (or None)."""
    try:
        from api.config.settings import _fernet as settings_fernet

        return settings_fernet()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("could not build Fernet for MCP secrets: %s", e)
        return None


def encrypt_secret_dict(data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Encrypt a secrets dict to a Fernet ciphertext string (JSON payload).

    ``None`` stays ``None`` (no secret configured). Empty dicts encrypt to an
    empty payload so they round-trip as ``{}``.
    """
    if data is None:
        return None
    f = _fernet()
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True)
    if f is None:
        logger.warning(
            "Fernet unavailable; storing MCP secrets dict UNENCRYPTED "
            "(set SETTINGS_SECRET_KEY in production)."
        )
        return "json:" + payload
    return f.encrypt(payload.encode("utf-8")).decode("utf-8")


def decrypt_secret_dict(stored: Optional[str]) -> Dict[str, str]:
    """Decrypt a stored secrets ciphertext back to a dict.

    Never raises: any failure (missing key, corrupt ciphertext) logs a warning
    and returns ``{}`` so a rotated/lost key degrades to "no credentials"
    instead of crashing connections.
    """
    if not stored:
        return {}
    if isinstance(stored, dict):  # legacy/plain shape — accept as-is
        return {str(k): str(v) for k, v in stored.items()}
    text = str(stored)
    if text.startswith("json:"):  # plaintext fallback written by encrypt path
        try:
            data = json.loads(text[len("json:"):])
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except Exception:
            return {}
    f = _fernet()
    if f is None:
        return {}
    try:
        payload = f.decrypt(text.encode("utf-8")).decode("utf-8")
        data = json.loads(payload)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("could not decrypt MCP secrets (%s); ignoring.", e)
        return {}


def mask_secret_dict(data: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Masked key-only view of a secrets dict for API responses.

    ``{"Authorization": "Bearer secret"}`` -> ``{"Authorization": "***"}`` —
    clients learn WHICH headers/env vars are configured, never their values.
    """
    if not data:
        return {}
    return {str(k): MASK for k in data.keys()}


__all__ = [
    "MASK",
    "decrypt_secret_dict",
    "encrypt_secret_dict",
    "mask_secret_dict",
]
