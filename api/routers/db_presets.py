"""Database preset catalog router — the ``GET /api/db-presets`` endpoint.

Serves the static preset registry (``api/mcp/presets.py``) to the frontend:
one entry per out-of-the-box database type (PostgreSQL/MySQL/MariaDB/SQL
Server/SQLite via dbhub, Oracle via oracle-mcp-server) with a DSN example,
the expected DSN shape and the MCP server's license attribution (both
servers are MIT; see NOTICE.md).

No secrets, no DSNs — the payload is a constant computed from the registry.
Authenticated like the products router family (no admin gate: every user
who can add a database needs the catalog for the type selector).
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from api.auth.deps import get_current_user
from api.mcp.presets import presets_public_view

router = APIRouter(
    prefix="/api/db-presets",
    tags=["db-presets"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", response_model=List[Dict[str, Any]])
async def list_db_presets() -> List[Dict[str, Any]]:
    """The static preset catalog for the add-database dialog."""
    return presets_public_view()


__all__ = ["router"]
