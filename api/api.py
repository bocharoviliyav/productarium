"""FastAPI app entry point.

This module is intentionally thin: app creation, CORS, the startup lifecycle
(init DB, bootstrap config/admin, pre-warm RLM), and dynamic router loading.
All HTTP endpoints live in ``api/routers/*.py`` (auto-discovered) and DB
access lives in ``api/repositories/``. ``main.py`` imports ``app`` from here.
"""

import asyncio
import logging

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
# background cognee init scheduled at startup (see ``startup_event``).
_cognee_init_tasks: set = set()

app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Capture the long-lived main event loop so the docgen worker threads (which
    # run their own short-lived loops) can hand off fire-and-forget cognee
    # indexing via run_coroutine_threadsafe. This keeps a 20-30 min cognify
    # running after the worker loop closes (display decoupled from the graph).
    try:
        from api.docgen import set_main_event_loop
        set_main_event_loop(asyncio.get_running_loop())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not capture main event loop for docgen indexing: %s", e)

    # init_db() is non-fatal: logs a warning and returns False if the DB is
    # unreachable, so app startup is never blocked.
    init_db()

    # Bootstrap configuration abstraction layer (highest precedence to DB settings)
    try:
        from api.config_abstraction import bootstrap_config
        bootstrap_config()
    except Exception as e:
        logger.warning("bootstrap_config failed (non-fatal): %s", e)

    # Cognee is a SECONDARY feature (a knowledge-graph index over generated
    # docs). The app must start and serve requests regardless of cognee's
    # availability: docgen writes generated docs to the product DB directly,
    # and the expert agent / summary fall back to artifact docs when cognee is
    # unavailable or still initializing. We therefore fire-and-forget
    # ``init_cognee()`` on the main loop instead of awaiting it: the app starts
    # immediately, and cognee finishes its (timeout-capped) init in the
    # background. The task reference is kept so the GC does not drop a
    # still-running init. Non-fatal: any failure is logged inside init_cognee.
    try:
        from api.cognee_manager import init_cognee
        _cognee_init_task = asyncio.create_task(init_cognee())
        # Prevent the loop from garbage-collecting the task before it finishes.
        _cognee_init_task.add_done_callback(
            lambda t: _cognee_init_tasks.discard(t)
        )
        _cognee_init_tasks.add(_cognee_init_task)
        logger.info("Scheduled cognee init in the background; app startup not blocked.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not schedule background cognee init: %s", e)

    # One-shot bootstrap admin (non-fatal): creates an admin from
    # BOOTSTRAP_ADMIN_USERNAME/PASSWORD when no admin exists yet.
    try:
        bootstrap_admin()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("bootstrap_admin failed (non-fatal): %s", e)

    # Pre-warm RLM in a background thread so the first-run fast-rlm
    # npm/pyodide download happens at boot, not inside the first generate
    # request. Non-fatal if fast-rlm is unavailable.
    try:
        from api.rlm.runner import prewarm_rlm_background
        prewarm_rlm_background()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("RLM prewarm could not start (non-fatal): %s", e)


# --- Dynamic router loader (foundation + Wave 2 routers) -------------------
# Discovers api/routers/*.py modules and includes their `router` APIRouters,
# plus the foundation auth router (api/auth/router.py). New routers just drop
# in api/routers/<name>.py without editing this file.
from api.routers import include_all_routers  # noqa: E402

_router_includes = include_all_routers(app)
if _router_includes:
    logger.info("Included routers via dynamic loader: %s", _router_includes)
