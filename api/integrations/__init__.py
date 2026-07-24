"""Scalable integrations framework for Productarium (plan section G).

A connector is a thin adapter that knows how to talk to an external source of
knowledge (a Git host, Confluence, an MCP server, ...). Every connector lives
in its own module under ``api/integrations/`` and subclasses
:class:`api.integrations.base.IntegrationConnector`.

Connectors are auto-registered: dropping a new ``api/integrations/<name>.py``
that defines an ``IntegrationConnector`` subclass is enough —
:mod:`api.integrations.registry` discovers it via :mod:`pkgutil` at first use
and no core code needs to change.

Public API (re-exported here for convenience)::

    from api.integrations import IntegrationConnector, get_connector, list_connectors

The package ``__init__`` deliberately does NOT eagerly import the connector
modules (to avoid import cycles and heavy deps at import time). Discovery is
lazy and lives in :mod:`api.integrations.registry`.
"""

from __future__ import annotations

from api.integrations.base import IntegrationConnector
from api.integrations.registry import (
    get_connector,
    list_connectors,
    register,
    reset_registry,
)

__all__ = [
    "IntegrationConnector",
    "get_connector",
    "list_connectors",
    "register",
    "reset_registry",
]
