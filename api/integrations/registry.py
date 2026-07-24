"""Connector registry with pkgutil auto-discovery (plan section G).

Connectors are auto-registered: :func:`_autodiscover` scans
``api/integrations/*.py`` via :mod:`pkgutil`, imports each module, and
collects every concrete :class:`api.integrations.base.IntegrationConnector`
subclass defined in it. Dropping a new connector file in the package is
therefore enough — no core code changes required.

Discovery is lazy (runs on the first ``get_connector`` / ``list_connectors``
call) so importing :mod:`api.integrations` stays cheap and side-effect free.

A ``register`` decorator is also provided for explicit registration (e.g. for
tests or third-party plugins loaded outside the package), but the built-in
connectors rely purely on auto-discovery.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Dict, List, Optional, Type

from api.integrations.base import IntegrationConnector

logger = logging.getLogger(__name__)

# name -> connector class
_REGISTRY: Dict[str, Type[IntegrationConnector]] = {}
_DISCOVERED = False

_PACKAGE = "api.integrations"
# Modules in the package that are not connectors and should be skipped during
# discovery (the base class + this registry module itself).
_NON_CONNECTOR_MODULES = {"base", "registry"}


def register(cls: Type[IntegrationConnector]) -> Type[IntegrationConnector]:
    """Explicitly register a connector class. Usable as a decorator.

    Auto-discovery already handles in-package connectors, so this is mainly
    for tests or plugins loaded from outside the package.
    """
    if not getattr(cls, "name", ""):
        raise ValueError(
            f"Connector {cls.__name__} has no `name` attribute; cannot register."
        )
    _REGISTRY[cls.name] = cls
    logger.debug("Registered integration connector: %s (%s)", cls.name, cls.__name__)
    return cls


def reset_registry() -> None:
    """Clear the registry and force re-discovery on next access (test helper)."""
    _REGISTRY.clear()
    global _DISCOVERED
    _DISCOVERED = False


def _autodiscover() -> None:
    """Import every connector module in the package and collect subclasses."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    try:
        pkg = importlib.import_module(_PACKAGE)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not import %s package: %s", _PACKAGE, e)
        return
    pkg_path = getattr(pkg, "__path__", None)
    if not pkg_path:
        logger.warning("Package %s has no __path__; auto-discovery skipped.", _PACKAGE)
        return
    for module_info in pkgutil.iter_modules(pkg_path):
        mod_name = module_info.name
        if mod_name.startswith("_") or mod_name in _NON_CONNECTOR_MODULES:
            continue
        full = f"{_PACKAGE}.{mod_name}"
        try:
            module = importlib.import_module(full)
        except Exception as e:  # pragma: no cover - bad connector module
            logger.warning("Could not import integration module %s: %s", full, e)
            continue
        for attr_name, attr in vars(module).items():
            if attr_name.startswith("_"):
                continue
            if not isinstance(attr, type):
                continue
            if not issubclass(attr, IntegrationConnector) or attr is IntegrationConnector:
                continue
            # Only register classes DEFINED in this module (not re-exported).
            if getattr(attr, "__module__", "") != module.__name__:
                continue
            if not getattr(attr, "name", ""):
                logger.warning(
                    "Connector %s in %s has no `name`; skipped.", attr.__name__, full
                )
                continue
            _REGISTRY.setdefault(attr.name, attr)
            logger.debug("Auto-discovered connector: %s (%s)", attr.name, attr.__name__)


def get_connector_class(name: str) -> Optional[Type[IntegrationConnector]]:
    """Return the registered connector class for ``name`` (or None)."""
    _autodiscover()
    return _REGISTRY.get(name)


def get_connector(name: str) -> Optional[IntegrationConnector]:
    """Instantiate the configured connector for ``name`` (or None if unknown).

    The connector's ``get_config()`` classmethod is used to load its config
    from the settings store, so the returned instance is ready to use. Returns
    ``None`` when no connector with that name is registered.
    """
    cls = get_connector_class(name)
    if cls is None:
        return None
    try:
        config = cls.get_config()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("get_config() for %s failed: %s", name, e)
        config = {}
    return cls(config)


def list_connectors() -> List[Dict[str, Any]]:
    """List all registered connectors with their configured status.

    Returns a list of dicts (sorted by name) with:
        - ``name``: connector slug
        - ``display_name``: human-readable name
        - ``description``: short description
        - ``kind``: connector category ("web" or "mcp")
        - ``requires_credentials``: bool
        - ``configured``: bool — whether the connector has config loaded
    """
    _autodiscover()
    out: List[Dict[str, Any]] = []
    for name in sorted(_REGISTRY.keys()):
        cls = _REGISTRY[name]
        try:
            config = cls.get_config()
        except Exception:
            config = {}
        try:
            configured = cls(config).is_configured()
        except Exception:
            configured = bool(config)
        out.append(
            {
                "name": cls.name,
                "display_name": cls.display_name or cls.name,
                "description": cls.description,
                "kind": getattr(cls, "kind", "web"),
                "requires_credentials": cls.requires_credentials,
                "configured": configured,
            }
        )
    return out
