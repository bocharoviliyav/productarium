"""Disk cache for MCP database-introspection results (item 2.3b).

Port of the fork's atomic ``map_cache`` pattern (checkpointed, atomic writes)
applied to the most expensive stage of a database docgen run: the
introspection walk over an MCP surface. A big schema (hundreds of MB of DDL
with triggers and functions) can take the walk tens of minutes; the result is
reproducible while the MCP surface is unchanged, so reruns — a repair after an
interrupted enrichment, a model/language change — must NOT hammer the database
MCP server for the same payload again.

Design (mirrors the fork's invariants):

- **Atomic writes**: payload is streamed to a unique tmp file in the same
  directory, ``fsync``-ed, then ``os.replace``-d onto the final path — a
  reader never observes a torn file, and a crash mid-write leaves the previous
  entry intact.
- **Key = sha256(normalized payload)**: format version + the MCP surface
  identity (pinned server id + binding allowlist, or the product's enabled
  binding ids) + the walk budgets (a budget change reshapes the payload).
  Entry files are owner-only (``0600``) under an owner-only directory — the
  raw dump may carry stored-procedure bodies with secret-like strings.
- **Bounded lifetime**: entries expire via ``DB_INTROSPECTION_CACHE_TTL_SECONDS``
  (default 7 days; ``<= 0`` disables the cache entirely) so a schema that
  actually changed is eventually re-introspected. Disk growth is bounded by
  the operator (clear ``<state>/introspection_cache``); distinct surfaces get
  distinct keys, and a changed binding set naturally writes a new file.

Every entry point is best-effort: a cache that cannot be read or written is a
warning, never a failed generation run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from api.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

#: Payload format version — a bump invalidates every stored entry.
#: v4: the walk is multi-schema for Oracle (ALL_* bulk walk via the sql
#: tool, quoted-alias catalog queries), filters PostgreSQL system schemas
#: (pg_*/information_schema) and extension-owned objects (pg_depend
#: deptype 'e'), and the SQL packs are AUTHORITATIVE for their categories —
#: v3 payloads carry single-schema Oracle tables and extension-noisy
#: categories and must not be replayed.
#: v3: the DB-RE restructure reshaped the payload — the walk now also carries
#: ``fk_edges``, the category collections (views/triggers/routines/sequences/
#: types with sources), the sql-pack evidence merged into table meta, and
#: ``tools_used``/``unavailable`` — v2 payloads lack the FK graph and the
#: category evidence the renderer/provenance now require, so they must not
#: be replayed.
#: v2: ``_call_tool`` now unwraps MCP content-block/TextContent envelopes and
#: ``_parse_names``/``_render_search_full`` drill into inner result
#: collections — v1 payloads may hold envelope-shredded garbage and must not
#: be replayed.
CACHE_FORMAT_VERSION = 3

_CACHE_DIR_NAME = "introspection_cache"
#: Default TTL: long enough to cover reruns/repairs of a multi-hour job,
#: short enough that a genuinely evolved schema is re-introspected within
#: a week of operator inactivity.
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600.0


def _env_float(name: str, default: float) -> float:
    """Call-time env parsing that never crashes on garbage values."""
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw)
    except ValueError:
        if raw:
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def cache_ttl_seconds() -> float:
    """TTL for introspection cache entries (read at call time).

    ``<= 0`` disables the cache entirely: nothing is written, nothing is
    served, every run introspects fresh.
    """
    return _env_float("DB_INTROSPECTION_CACHE_TTL_SECONDS", _DEFAULT_TTL_SECONDS)


def _state_dir() -> str:
    """Managed state root (same resolution as api.db / api.agents.runtime)."""
    return os.environ.get("PRODUCTARIUM_STATE_DIR") or os.path.expanduser(
        "~/.productarium"
    )


def _cache_root() -> str:
    return os.path.join(_state_dir(), _CACHE_DIR_NAME)


def introspection_cache_key(
    *,
    mcp_server_id: Optional[str] = None,
    binding_ids: Optional[list] = None,
    allowlist: Optional[list] = None,
    budgets: Optional[Dict[str, Any]] = None,
) -> str:
    """Deterministic cache key: sha256 over the normalized payload.

    Only inputs that change the introspection RESULT participate: the MCP
    surface identity (pinned server + allowlist, or the enabled binding ids)
    and the walk budgets. The entity/product ids deliberately do NOT — two
    entities pinned to the same server share one cache entry instead of
    duplicating a multi-hundred-MB payload on disk.
    """
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "mcp_server_id": (mcp_server_id or "").strip(),
        "binding_ids": sorted(str(b) for b in (binding_ids or [])),
        "allowlist": sorted(str(a) for a in (allowlist or [])),
        "budgets": {str(k): v for k, v in sorted((budgets or {}).items())},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _entry_path(key: str) -> str:
    return os.path.join(_cache_root(), f"{key}.json")


def load_introspection_cache(key: str) -> Optional[Dict[str, Any]]:
    """Read a cached introspection payload; ``None`` on ANY miss.

    Misses: disabled cache, unreadable/corrupt file, key or version mismatch,
    expired entry (file mtime vs TTL). Never raises — cache bookkeeping must
    not fail a generation run.
    """
    if not key or cache_ttl_seconds() <= 0:
        return None
    path = _entry_path(key)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug("introspection cache miss for %s…: %s", key[:12], e)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_key") != key or payload.get("version") != CACHE_FORMAT_VERSION:
        logger.debug("introspection cache key/version mismatch for %s…", key[:12])
        return None
    ttl = cache_ttl_seconds()
    try:
        age = max(0.0, time.time() - os.path.getmtime(path))
    except OSError:  # pragma: no cover - file vanished between open and stat
        return None
    if age > ttl:
        logger.info(
            "introspection cache entry expired (age=%.0fs ttl=%.0fs); "
            "re-introspecting.",
            age, ttl,
        )
        return None
    info = payload.get("info")
    return info if isinstance(info, dict) else None


def store_introspection_cache(key: str, info: Dict[str, Any]) -> bool:
    """Atomically persist an introspection payload (best-effort, never raises).

    Streams the JSON to a per-thread tmp file, ``fsync``-s, then atomically
    replaces the entry — concurrent writers of the same key (two entities on
    one MCP surface) converge on one intact file instead of tearing it.
    """
    if not key or not isinstance(info, dict) or cache_ttl_seconds() <= 0:
        return False
    path = _entry_path(key)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        os.makedirs(_cache_root(), exist_ok=True)
        try:
            # Owner-only for a directory WE created (never chmod shared dirs).
            os.chmod(_cache_root(), 0o700)
        except OSError:  # pragma: no cover - FS dependent
            pass
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cache_key": key,
                    "version": CACHE_FORMAT_VERSION,
                    "stored_at": time.time(),
                    "info": info,
                },
                f,
                ensure_ascii=False,
            )
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
        return True
    except Exception as e:  # pragma: no cover - cache write must never break gen
        logger.warning("introspection cache write failed for %s…: %s", key[:12], e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False


__all__ = [
    "CACHE_FORMAT_VERSION",
    "cache_ttl_seconds",
    "introspection_cache_key",
    "load_introspection_cache",
    "store_introspection_cache",
]
