"""Resolve the active memory backend from the ``memory.backend`` admin setting.

The active backend (``pgvector`` default, ``cognee`` alternative) is read from
the settings store on every ``get_memory_backend()`` call, but the backend
instance is cached so a hot recall path does not re-instantiate it per call.
The cache is invalidated by ``reset_memory_backend_cache`` (called from
``sync_runtime_settings`` after an admin save) so a switch takes effect on the
next call without a process restart.
"""

from __future__ import annotations

import logging
from typing import Optional

from api.memory.base import MemoryBackend

logger = logging.getLogger(__name__)

_DEFAULT_BACKEND = "pgvector"
_VALID_BACKENDS = ("pgvector", "cognee")

# Cached backend instance + the name it was built for. Reset by
# reset_memory_backend_cache (called after an admin setting change).
_cache: dict = {"backend": None, "name": None}


def get_memory_backend_name() -> str:
    """Return the configured backend name (``pgvector`` | ``cognee``).

    Reads ``memory.backend`` from the settings store (admin > env), falling
    back to the default. An invalid value falls back to the default rather
    than raising, so a typo cannot brick the memory path.
    """
    try:
        from api.config.settings import get_setting

        raw = (get_setting("memory.backend") or _DEFAULT_BACKEND).strip().lower()
    except Exception:
        raw = _DEFAULT_BACKEND
    if raw not in _VALID_BACKENDS:
        logger.warning(
            "Invalid memory.backend %r; falling back to %r.", raw, _DEFAULT_BACKEND
        )
        return _DEFAULT_BACKEND
    return raw


def get_memory_backend() -> MemoryBackend:
    """Return the active ``MemoryBackend`` instance (cached per backend name).

    The cache is keyed by the resolved name so switching backends via the admin
    panel invalidates it naturally: a different name builds a new instance.
    """
    name = get_memory_backend_name()
    if _cache["backend"] is not None and _cache["name"] == name:
        return _cache["backend"]  # type: ignore[return-value]
    backend = _build_backend(name)
    _cache["backend"] = backend
    _cache["name"] = name
    return backend


def reset_memory_backend_cache() -> None:
    """Drop the cached backend instance so the next call rebuilds it.

    Called from ``sync_runtime_settings`` after an admin save so a backend
    switch takes effect immediately. Safe to call any time.
    """
    _cache["backend"] = None
    _cache["name"] = None


def _build_backend(name: str) -> MemoryBackend:
    """Instantiate a backend by name. Falls back to pgvector on import error."""
    if name == "cognee":
        try:
            from api.memory.cognee_backend import CogneeMemoryBackend

            return CogneeMemoryBackend()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Could not build cognee memory backend (%s); falling back to pgvector.", e
            )
            name = "pgvector"
    # default + fallback
    from api.memory.pgvector_backend import PgVectorMemoryBackend

    return PgVectorMemoryBackend()
