"""Abstract integration connector interface (plan section G).

Every connector implements this interface so the registry and the
``/api/integrations`` router can treat them uniformly. The contract is
intentionally small so new connectors (a new Git host, a wiki, an MCP server)
can be added by implementing three methods:

- ``test()``          -> ``{"success": bool, "message": str}`` — validate
  connectivity / credentials. Never raises; failures are reported via the
  returned dict so admin UIs can surface them.
- ``list_spaces()``   -> ``list[{"id", "title", "type", ...}]`` — enumerate the
  top-level sources the connector can pull from (GitHub repos, Confluence
  spaces, MCP knowledge sources, ...).
- ``pull(source_id, opts=None)`` -> ``{"title", "markdown", "attachments", ...}``
  — fetch a single space/repo/page and return its markdown representation.
  ``attachments`` is a list of ``{"filename", "markdown"}`` for converted
  binary attachments (handled via :mod:`api.formats.markitdown`).

Configuration is injected via the constructor (``config`` dict). Connectors
load their own config from :mod:`api.config.settings` through the classmethod
``get_config()`` so the registry can build a configured instance and report
whether a connector is ready to use (``is_configured()``).

All connectors must be import-safe with no live services/creds available —
network calls happen only inside ``test``/``list_spaces``/``pull``, and those
catch their own errors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IntegrationConnector(ABC):
    """Abstract base class for all integration connectors.

    Subclasses set the class attributes ``name`` (unique slug used in URLs and
    the registry), ``display_name`` and ``description`` (for admin UIs), and
    optionally ``requires_credentials``. They implement ``test``,
    ``list_spaces`` and ``pull``, and override ``get_config`` to read their
    credentials from the settings store.
    """

    # Unique connector slug (used in /api/integrations/{name} paths + registry).
    name: str = ""
    display_name: str = ""
    description: str = ""
    # Connector category: "web" for HTTP/git/wiki connectors, "mcp" for MCP servers.
    kind: str = "web"
    # Whether the connector needs admin-configured credentials to be useful.
    requires_credentials: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # The resolved config (e.g. git creds / confluence creds / mcp config).
        # Connectors should treat missing keys gracefully.
        self.config: Dict[str, Any] = dict(config or {})

    # -- Configuration -------------------------------------------------------
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Load this connector's config from the settings store.

        Default returns an empty dict (connector has no persisted config).
        Subclasses override to read e.g. ``get_git_creds`` / ``get_confluence_creds``
        / ``get_integration_config``. This must be import-safe (no network) and
        must not raise — the settings store already degrades gracefully when
        the DB is down.
        """
        return {}

    def is_configured(self) -> bool:
        """Whether the connector has enough config to attempt operations.

        Default: True when ``config`` is non-empty. Subclasses may override for
        stricter checks (e.g. require both ``url`` and ``token``).
        """
        return bool(self.config)

    # -- Operations ----------------------------------------------------------
    @abstractmethod
    def test(self) -> Dict[str, Any]:
        """Test connectivity / credentials.

        Returns a dict with at least ``{"success": bool, "message": str}``.
        Implementations MUST NOT raise — wrap all failures in the returned
        dict so the admin panel can surface them safely.
        """

    @abstractmethod
    def list_spaces(self) -> List[Dict[str, Any]]:
        """List the top-level pull sources (repos / spaces / knowledge sources).

        Returns a list of dicts with at least ``{"id", "title", "type"}``.
        Implementations MUST NOT raise — return ``[]`` (or an error entry) on
        failure.
        """

    @abstractmethod
    def pull(self, source_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Pull a single space/repo/page.

        Args:
            source_id: Connector-specific source identifier (repo URL, page
                id, MCP source name, ...).
            opts: Optional dict of per-pull options (e.g. ``{"recursive": True}``
                to pull child pages, ``{"branch": "main"}`` for git).

        Returns a dict with at least:
            - ``title``: str — human-readable title for the pulled content.
            - ``markdown``: str — the pulled content as markdown.
            - ``attachments``: list[{"filename": str, "markdown": str}] —
              binary attachments converted to markdown (may be empty).

        Implementations MAY add connector-specific keys (e.g. ``repo_url``,
        ``repo_type``, ``local_path`` for git connectors). Implementations
        SHOULD raise on hard failures (the router converts these to 4xx/5xx);
        they MUST NOT raise on missing optional data.
        """
