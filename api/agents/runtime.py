"""LangGraph checkpointer management for the expert agent.

Postgres is the ONLY runtime backend (explicit product decision): the
checkpointer is initialized eagerly at app startup and an unreachable
Postgres fails startup outright — there is no silent SQLite degradation.

``AsyncPostgresSaver.from_conn_string(..., pipeline=True)`` is deliberately
NOT used: its ``setup()`` runs the checkpoint-table migrations (which include
``CREATE INDEX CONCURRENTLY``) while the connection is already in psycopg
pipeline mode, and libpq wraps queued statements there in an implicit
transaction block — Postgres rejects ``CREATE INDEX CONCURRENTLY`` in a
transaction, the setup failed on every boot, and the process silently fell
back to a SQLite file. Instead we connect manually, run ``setup()`` on the
bare autocommit connection (no pipeline, no transaction), and only THEN enter
pipeline mode for the saver's lifetime.

``PRODUCTARIUM_CHECKPOINTER=memory`` is an explicit test-only override that
swaps in ``InMemorySaver`` (set by ``tests/conftest.py`` for the hermetic
suite). It is not a runtime fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_checkpointer: Optional[Any] = None
_checkpointer_lock = asyncio.Lock()


def _memory_override_enabled() -> bool:
    """True when the test-only memory checkpointer override is requested."""
    return (os.environ.get("PRODUCTARIUM_CHECKPOINTER") or "").strip().lower() == "memory"


def _postgres_conn_string() -> Optional[str]:
    """Build a psycopg connection string from the app's DB_* env config.

    Returns None when the provider is not Postgres (tests / SQLite setups).
    """
    from api import db as db_mod

    provider = (getattr(db_mod, "DB_PROVIDER", "") or "").lower()
    if provider not in ("postgres", "postgresql"):
        return None
    return (
        f"host={db_mod.DB_HOST} port={db_mod.DB_PORT} dbname={db_mod.DB_NAME} "
        f"user={db_mod.DB_USERNAME} password={db_mod.DB_PASSWORD}"
    )


async def _build_postgres_saver() -> Any:
    """Open an AsyncPostgresSaver: setup() FIRST, pipeline SECOND.

    Order matters: ``CREATE INDEX CONCURRENTLY`` (part of the checkpoint
    migrations) cannot run inside a transaction block, and pipeline mode
    implies one — so the migrations run on the plain autocommit connection
    before the pipeline is entered.
    """
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    conn_string = _postgres_conn_string()
    if not conn_string:
        raise RuntimeError(
            "DB_PROVIDER is not postgres: the agent checkpointer requires Postgres."
        )
    conn = await AsyncConnection.connect(
        conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
    )
    try:
        saver = AsyncPostgresSaver(conn=conn)
        await saver.setup()
    except BaseException:
        # Never leak the connection when construction or setup fails.
        try:
            await conn.close()
        except Exception:  # pragma: no cover - defensive
            pass
        raise
    pipe = conn.pipeline()
    await pipe.__aenter__()
    saver.pipe = pipe
    saver._productarium_conn = conn
    saver._productarium_pipe = pipe
    return saver


def _build_memory_saver() -> Any:
    """Create the in-memory saver (explicit test-only override)."""
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


async def _close_saver(saver: Any) -> None:
    """Close a saver's pipeline and psycopg connection (non-fatal)."""
    pipe = getattr(saver, "_productarium_pipe", None)
    if pipe is not None:
        try:
            await pipe.__aexit__(None, None, None)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("agent checkpointer pipeline close failed: %s", e)
    conn = getattr(saver, "_productarium_conn", None)
    if conn is not None:
        try:
            await conn.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("agent checkpointer connection close failed: %s", e)


async def get_checkpointer() -> Any:
    """Return the process-wide async checkpointer (creating it if needed).

    Postgres-only at runtime: any initialization failure RAISES so app
    startup fails loudly (no Postgres — no app). The memory backend is
    reachable only through the explicit ``PRODUCTARIUM_CHECKPOINTER=memory``
    test override.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    async with _checkpointer_lock:
        if _checkpointer is not None:
            return _checkpointer
        if _memory_override_enabled():
            logger.info(
                "agent checkpointer: using memory backend "
                "(PRODUCTARIUM_CHECKPOINTER=memory test-only override)."
            )
            _checkpointer = _build_memory_saver()
            return _checkpointer
        saver = await _build_postgres_saver()  # fatal on failure (by design)
        logger.info("agent checkpointer: using postgres backend (pipeline mode).")
        _checkpointer = saver
        return _checkpointer


async def close_checkpointer() -> None:
    """Close the active checkpointer (app shutdown). Non-fatal."""
    global _checkpointer
    saver = _checkpointer
    _checkpointer = None
    if saver is not None:
        await _close_saver(saver)


def reset_checkpointer_cache() -> None:
    """Drop the cached checkpointer so the next call rebuilds it (tests)."""
    global _checkpointer
    _checkpointer = None


__all__ = [
    "close_checkpointer",
    "get_checkpointer",
    "reset_checkpointer_cache",
]
