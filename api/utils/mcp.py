"""MCP client facade (former ``api/mcp_client.py``).

High-level MCP client for Productarium internal services. Wraps
``McpConnector`` from ``api.integrations.mcp`` so callers (Expert Agent,
background tasks, the Confluence connector's MCP mode) can discover and invoke
tools from configured local and remote MCP servers without touching the
transport layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from api.integrations.mcp import McpConnector

logger = logging.getLogger(__name__)


class LocalMcpClient:
    """High-level MCP client for Productarium internal services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.connector = McpConnector(config=config)

    def is_configured(self) -> bool:
        """True when at least one MCP server is configured."""
        return self.connector.is_configured()

    def list_servers(self) -> List[Dict[str, Any]]:
        """List configured MCP servers and their transport status."""
        return self.connector._servers()

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools across all configured MCP servers."""
        try:
            return self.connector.list_spaces()
        except Exception as e:
            logger.warning("LocalMcpClient list_tools failed: %s", e)
            return []

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke a specific tool on an MCP server.

        Args:
            server_name: Name of the configured MCP server.
            tool_name: Name of the tool on that server.
            arguments: Dict of keyword arguments for the tool.

        Returns:
            Dict containing `title`, `markdown` (result text), `source`, `server`, `tool`.
        """
        source_id = f"{server_name}:{tool_name}" if tool_name else server_name
        return self.connector.pull(source_id, opts=arguments)

    def execute_by_id(self, source_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke an MCP tool by source_id ('server:tool' or 'server')."""
        return self.connector.pull(source_id, opts=opts)

    def test_connections(self) -> Dict[str, Any]:
        """Test connectivity to all configured MCP servers."""
        return self.connector.test()

    def close(self) -> None:
        """Close cached stdio subprocesses."""
        self.connector.close()


# Global convenience functions
def get_local_mcp_client() -> LocalMcpClient:
    """Create a LocalMcpClient instance with configuration read from admin settings."""
    return LocalMcpClient()


def list_all_mcp_tools() -> List[Dict[str, Any]]:
    """List all available MCP tools across configured servers."""
    client = get_local_mcp_client()
    try:
        return client.list_tools()
    finally:
        client.close()


def invoke_mcp_tool(
    source_id: str,
    opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke an MCP tool by source_id ('server:tool') with options."""
    client = get_local_mcp_client()
    try:
        return client.execute_by_id(source_id, opts=opts)
    finally:
        client.close()
