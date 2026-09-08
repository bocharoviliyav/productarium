"""MCP platform package (Wave C).

Outbound: :mod:`api.mcp.manager` — connection/tool cache over
``langchain-mcp-adapters`` for MCP servers registered by admins
(``McpServerORM``) and bound to products (``ProductMcpServerORM``).

Inbound: :mod:`api.mcp.inbound` — FastMCP streamable-HTTP server at
``/api/mcp`` exposing Productarium knowledge to external MCP clients
(Bearer API-token authenticated).

Secrets: :mod:`api.mcp.secrets` — Fernet encryption of MCP server
``headers``/``env`` at rest + masked (key-only) views for API responses.
"""

from __future__ import annotations

__all__ = ["manager", "secrets"]
