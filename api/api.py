"""FastAPI app entry point.

This module is intentionally thin: app creation, CORS, the startup lifecycle
(init DB, bootstrap config/admin, background memory init, inbound MCP session
manager), and dynamic router loading. All HTTP endpoints live in
``api/routers/*.py`` (auto-discovered) and DB access lives in
``api/repositories/``. ``main.py`` imports ``app`` from here.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth.bootstrap import bootstrap_admin
from api.auth.deps import get_current_user  # noqa: F401  (injection point)
from api.db import get_db, init_db  # noqa: F401  (re-exported for tests)
from api.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Strong references to fire-and-forget background tasks so the running event
# loop does not garbage-collect them before they finish. Currently holds the
# background memory init scheduled at startup (see ``lifespan``).
_memory_init_tasks: set = set()


async def _startup() -> None:
    # Capture the long-lived main event loop so the docgen worker threads (which
    # run their own short-lived loops) can hand off fire-and-forget memory
    # indexing via run_coroutine_threadsafe. This keeps a long-running index
    # running after the worker loop closes (display decoupled from indexing).
    try:
        from api.docgen import set_main_event_loop
        set_main_event_loop(asyncio.get_running_loop())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not capture main event loop for docgen indexing: %s", e)

    # init_db() is non-fatal: logs a warning and returns False if the DB is
    # unreachable, so app startup is never blocked. init_db also creates the
    # pgvector extension; the HNSW index is pinned + created lazily on the
    # first memory index run (dimensionless vector column until then).
    init_db()

    # Agent checkpointer (Postgres-only): eager, FATAL init — no Postgres, no
    # app (explicit product decision; replaced the old silent SQLite
    # fallback). An exception here propagates and fails app startup.
    from api.agents.runtime import get_checkpointer  # noqa: WPS433

    await get_checkpointer()
    logger.info("Agent checkpointer initialized (postgres).")

    # P0-2: one-shot migration of any legacy plaintext codebase tokens to
    # Fernet-encrypted form (non-fatal — per-row lazy migration also runs on
    # access, so a failure here never blocks startup).
    try:
        from api.repositories import product_repo
        product_repo.migrate_plaintext_tokens()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Plaintext token migration skipped (non-fatal): %s", e)

    # Bootstrap configuration abstraction layer (highest precedence to DB settings)
    try:
        from api.config.abstraction import bootstrap_config
        bootstrap_config()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("bootstrap_config failed (non-fatal): %s", e)

    # Initialize the active memory backend (pgvector) in the background so app
    # startup is never blocked. For pgvector this is a no-op (schema created in
    # init_db). Non-fatal on any failure.
    try:
        from api.memory import init_memory
        _memory_init_task = asyncio.create_task(init_memory())
        _memory_init_task.add_done_callback(
            lambda t: _memory_init_tasks.discard(t)
        )
        _memory_init_tasks.add(_memory_init_task)
        logger.info("Scheduled memory backend init in the background; app startup not blocked.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not schedule background memory init: %s", e)

    # One-shot bootstrap admin (non-fatal): creates an admin from
    # BOOTSTRAP_ADMIN_USERNAME/PASSWORD when no admin exists yet.
    try:
        bootstrap_admin()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("bootstrap_admin failed (non-fatal): %s", e)


async def _shutdown() -> None:
    # Close the process-wide agent checkpointer (its Postgres saver owns a
    # pipeline + connection that must not outlive the app). Non-fatal by design.
    try:
        from api.agents.runtime import close_checkpointer

        await close_checkpointer()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("close_checkpointer failed (non-fatal): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle: startup, the inbound MCP session manager, then shutdown.

    The inbound MCP server (mounted at ``/api/mcp``) needs its streamable-HTTP
    session manager running for the app's lifetime; Starlette does not
    propagate lifespans into mounted sub-apps, so it is entered here.
    """
    try:
        from api.mcp.inbound import inbound_session_manager
    except ImportError as e:
        # mcp dependency missing — degrade to plain startup/shutdown.
        logger.warning("Inbound MCP server disabled (dependency missing): %s", e)
        inbound_session_manager = None

    await _startup()
    if inbound_session_manager is not None:
        async with inbound_session_manager():
            yield
    else:
        yield
    await _shutdown()


app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions",
    lifespan=lifespan,
)

# CORS: explicit origin allowlist (comma-separated CORS_ORIGINS). The old
# "*" + credentials combination is invalid per the CORS spec (browsers
# refuse credentialed requests with a wildcard origin), so "*" now also
# disables allow_credentials. The Next.js frontend is same-origin via its
# /api proxy and does not need CORS at all; this only matters for direct
# browser-to-API calls.
_cors_raw = (os.environ.get("CORS_ORIGINS") or "").strip()
_cors_list = [o.strip() for o in _cors_raw.split(",") if o.strip()]
# "*" ANYWHERE in the list means allow-all: a mixed list like "*,http://foo"
# must not smuggle a literal "*" entry into allow_origins — Starlette treats
# any "*" member as allow-all and then reflects arbitrary origins even with
# credentials enabled, bypassing the rest of the allowlist (review #5).
_cors_allow_all = "*" in _cors_list
_cors_origins = (
    ["*"]
    if _cors_allow_all
    else (_cors_list or ["http://localhost:3000"])
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _cors_allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Dynamic router loader (foundation + Wave 2 routers) -------------------
# Discovers api/routers/*.py modules and includes their `router` APIRouters,
# plus the foundation auth router (api/auth/router.py). New routers just drop
# in api/routers/<name>.py without editing this file.
from api.routers import include_all_routers  # noqa: E402

_router_includes = include_all_routers(app)
if _router_includes:
    logger.info("Included routers via dynamic loader: %s", _router_includes)

# --- Inbound MCP server (Wave C): Productarium as an MCP server at /api/mcp --
# Bearer-API-token auth is enforced by the ASGI wrapper itself. Mounted AFTER
# the routers so no route can shadow the mount (and vice versa). Import failure
# (mcp dependency missing) degrades to a warning — the REST API still boots.
try:
    from api.mcp.inbound import get_inbound_mcp_app

    app.mount("/api/mcp", get_inbound_mcp_app())
    logger.info("Inbound MCP server mounted at /api/mcp")
except Exception as e:  # pragma: no cover - mcp dependency missing
    logger.warning("Inbound MCP server not mounted: %s", e)
