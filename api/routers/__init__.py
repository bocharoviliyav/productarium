"""Dynamic router loader for ``api/routers/*.py``.

Wave 2 agents drop a ``api/routers/<name>.py`` defining a module-level
``router = APIRouter(...)`` and it is auto-included into the FastAPI app — no
need to edit ``api/api.py``. The loader scans the package at startup, imports
each module, and includes every ``APIRouter`` it finds.

Foundation routers that live outside ``api/routers/`` (the auth router at
``api/auth/router.py``) are also wired here so ``api/api.py`` only needs a
single ``include_all_routers(app)`` call.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

from fastapi import APIRouter, FastAPI

logger = logging.getLogger(__name__)

_ROUTER_PACKAGE = "api.routers"
_ROUTER_DIR = Path(__file__).parent

# Foundation routers that live outside api/routers/ but should still be wired by
# this loader (kept here so api.api only calls include_all_routers once).
_EXTRA_ROUTER_MODULES: Tuple[str, ...] = ("api.auth.router",)


def discover_routers() -> List[Tuple[str, APIRouter]]:
    """Import every module in ``api/routers`` and collect module-level APIRouters."""
    found: List[Tuple[str, APIRouter]] = []
    for module_info in pkgutil.iter_modules([str(_ROUTER_DIR)]):
        name = module_info.name
        if name.startswith("_"):
            continue
        full = f"{_ROUTER_PACKAGE}.{name}"
        try:
            module: ModuleType = importlib.import_module(full)
        except Exception as e:  # pragma: no cover - bad router module
            logger.warning("Could not import router module %s: %s", full, e)
            continue
        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            found.append((name, router))
            logger.info(
                "Discovered router: %s (prefix=%s)", full, getattr(router, "prefix", "")
            )
        else:
            logger.debug("Module %s has no `router` APIRouter; skipped.", full)
    return found


def include_all_routers(app: FastAPI) -> List[str]:
    """Discover and include all ``api/routers/*.py`` routers + foundation routers."""
    included: List[str] = []
    for name, router in discover_routers():
        try:
            app.include_router(router)
            included.append(name)
        except Exception as e:  # pragma: no cover - bad router
            logger.warning("Could not include router %s: %s", name, e)

    # Foundation routers that live outside api/routers/ (auth).
    for mod_full in _EXTRA_ROUTER_MODULES:
        try:
            module = importlib.import_module(mod_full)
        except Exception as e:  # pragma: no cover - dep missing
            logger.warning("Could not import %s: %s", mod_full, e)
            continue
        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            try:
                app.include_router(router)
                included.append(mod_full)
            except Exception as e:  # pragma: no cover
                logger.warning("Could not include %s: %s", mod_full, e)
    return included
