#!/usr/bin/env python3
"""SPEC-028 ``api.db._backfill_mcp_server_scopes`` over an isolated SQLite DB.

Preset rows (``preset_key`` set) become database-only; every other
pre-scopes row keeps its old visibility (``["expert", "database"]``); rows
with scopes already set are untouched — the UPDATE is idempotent across
restarts (``WHERE scopes IS NULL``).
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestMcpServerScopesBackfill:
    def test_backfill_assigns_scopes_and_is_idempotent(self, isolated_db):
        from api.models import McpServerORM, ProductORM

        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_1", name="P1"))
            s.add(McpServerORM(
                id="mcp_preset", name="preset", preset_key="postgresql",
                transport="stdio", command="dbhub", enabled=True,
            ))
            s.add(McpServerORM(
                id="mcp_reg", name="reg", transport="http",
                url="http://r/mcp", enabled=True,
            ))
            s.add(McpServerORM(
                id="mcp_kept", name="kept", transport="http",
                url="http://k/mcp", enabled=True, scopes=["codebase"],
            ))
            s.commit()

        isolated_db._backfill_mcp_server_scopes()

        with isolated_db.SessionLocal() as s:
            assert s.get(McpServerORM, "mcp_preset").scopes == ["database"]
            assert s.get(McpServerORM, "mcp_reg").scopes == ["expert", "database"]
            # Explicit scopes are never overwritten.
            assert s.get(McpServerORM, "mcp_kept").scopes == ["codebase"]

        # Second run is a no-op (WHERE scopes IS NULL).
        isolated_db._backfill_mcp_server_scopes()
        with isolated_db.SessionLocal() as s:
            assert s.get(McpServerORM, "mcp_preset").scopes == ["database"]
            assert s.get(McpServerORM, "mcp_reg").scopes == ["expert", "database"]
