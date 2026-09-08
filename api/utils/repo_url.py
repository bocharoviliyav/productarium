"""Shared ``repo_url`` validation (P0-1, adversarial review).

Git clone sources must be plain ``http://`` / ``https://`` URLs with a valid
host. Everything else — ``ext::`` transport helpers (which execute arbitrary
shell commands), ``file://``, ``ssh://``, ``git://``, SCP-like ``user@host:``
syntax, bare local paths and empty values — is rejected *before* any ``git``
invocation.

The only exception: a non-URL *path* that lives inside the managed state dir
(``~/.adalflow``, the same directory the clone pipeline itself uses) is allowed
when ``PRODUCTARIUM_ALLOW_LOCAL_CLONES`` is set to a truthy value. This keeps
"managed clones" (repos the app itself cloned earlier) usable in local/dev
setups without re-opening arbitrary filesystem reads.

Every entry point that can trigger a clone must call :func:`validate_repo_url`
(CRUD create, docgen generate, integration pull, ``download_repo`` itself as
defense in depth).
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Truthy values for PRODUCTARIUM_ALLOW_LOCAL_CLONES (mirrors FastAPI-style env
# flags elsewhere in the project).
_TRUTHY = {"1", "true", "yes", "on"}

_DEFAULT_MANAGED_ROOT = os.path.join(os.path.expanduser("~"), ".adalflow")


def local_clones_allowed() -> bool:
    """True when PRODUCTARIUM_ALLOW_LOCAL_CLONES explicitly enables local paths."""
    return (os.environ.get("PRODUCTARIUM_ALLOW_LOCAL_CLONES", "") or "").strip().lower() in _TRUTHY


def _managed_root() -> str:
    """The managed state dir (~/.adalflow by default, via adalflow's helper)."""
    try:
        from adalflow.utils import get_adalflow_default_root_path

        return os.path.realpath(get_adalflow_default_root_path())
    except Exception:
        return os.path.realpath(_DEFAULT_MANAGED_ROOT)


def _is_within(child: str, parent: str) -> bool:
    """True when ``child`` is ``parent`` itself or nested inside it."""
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:
        # Different drives / mixed absolute-relative on Windows.
        return False


def is_http_repo_url(repo_url: str) -> bool:
    """True for a syntactically valid ``http://`` / ``https://`` repo URL."""
    text = (repo_url or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    # A real host is required;userinfo tricks like https://token@host are left
    # to the existing token-injection helpers, but an empty netloc is invalid.
    return bool(parsed.hostname)


def validate_repo_url(repo_url: str) -> str:
    """Validate a clone source URL (or managed local path). Raises ``ValueError``.

    Accepts:
    - ``http://`` / ``https://`` URLs with a non-empty host;
    - a filesystem path *inside* the managed state dir when
      ``PRODUCTARIUM_ALLOW_LOCAL_CLONES`` is truthy.

    Rejects everything else (``ext::``, ``file://``, ``ssh://``, ``git://``,
    ``user@host:repo``, relative/absolute local paths outside the managed dir,
    empty values) with a descriptive error.
    """
    text = (repo_url or "").strip()
    if not text:
        raise ValueError("repo_url is empty; provide an http(s) git repository URL")

    if is_http_repo_url(text):
        return text

    # Non-http value: only a managed local clone path is acceptable, and only
    # when the operator explicitly opted in.
    if not local_clones_allowed():
        raise ValueError(
            "repo_url must be an http(s) URL (file://, ssh://, git::, ext:: and "
            "local paths are rejected). Set PRODUCTARIUM_ALLOW_LOCAL_CLONES=1 to "
            "allow paths inside the managed state dir (~/.adalflow)."
        )

    root = _managed_root()
    candidate = os.path.realpath(os.path.expanduser(text))
    if not _is_within(candidate, root):
        raise ValueError(
            "Local repo paths are only allowed inside the managed state dir "
            f"({root}); got {text!r}."
        )
    return text
