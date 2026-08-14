"""Unit tests for ``api.utils.mcp`` (MCP client facade).

Covers:
- ``LocalMcpClient`` constructor + ``is_configured`` / ``list_servers`` delegation.
- ``list_tools`` (success + exception -> []).
- ``call_tool`` (source_id construction, opts pass-through).
- ``execute_by_id`` (direct source_id).
- ``test_connections`` delegation.
- ``close`` delegation.
- Module-level convenience functions: ``get_local_mcp_client``,
  ``list_all_mcp_tools``, ``invoke_mcp_tool``.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.utils import mcp as mcp_utils
from api.utils.mcp import (
    LocalMcpClient,
    get_local_mcp_client,
    invoke_mcp_tool,
    list_all_mcp_tools,
)


# ---------------------------------------------------------------------------
# LocalMcpClient — constructor + is_configured
# ---------------------------------------------------------------------------

class TestLocalMcpClientInit:
    def test_constructor_creates_connector(self):
        client = LocalMcpClient(config={"servers": [{"name": "s1"}]})
        assert client.connector is not None

    def test_constructor_none_config(self):
        client = LocalMcpClient(config=None)
        assert client.connector is not None

    def test_is_configured_delegates_to_connector(self):
        client = LocalMcpClient(config={"servers": [{"name": "s1"}]})
        assert client.is_configured() is True

    def test_is_not_configured_when_empty(self):
        client = LocalMcpClient(config={})
        assert client.is_configured() is False

    def test_list_servers_delegates(self):
        config = {"servers": [{"name": "s1", "transport": "http", "url": "http://x"}]}
        client = LocalMcpClient(config=config)
        servers = client.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "s1"

    def test_list_servers_empty(self):
        client = LocalMcpClient(config={})
        assert client.list_servers() == []


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

class TestListTools:
    def test_list_tools_success(self):
        config = {
            "servers": [
                {
                    "name": "s1",
                    "transport": "http",
                    "url": "http://localhost:8080",
                    "sources": [{"id": "t1", "title": "Tool 1", "type": "mcp_source"}],
                }
            ]
        }
        client = LocalMcpClient(config=config)
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0]["title"] == "Tool 1"
        client.close()

    def test_list_tools_returns_empty_on_exception(self, monkeypatch):
        client = LocalMcpClient(config={"servers": [{"name": "s1"}]})

        # Force list_spaces to raise
        def boom():
            raise RuntimeError("connection failed")

        monkeypatch.setattr(client.connector, "list_spaces", boom)
        assert client.list_tools() == []

    def test_list_tools_empty_when_no_servers(self):
        client = LocalMcpClient(config={})
        assert client.list_tools() == []


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------

class TestCallTool:
    def test_call_tool_with_tool_name(self):
        config = {
            "servers": [
                {
                    "name": "wiki",
                    "transport": "http",
                    "url": "http://localhost:8080",
                    "tool": "default_tool",
                    "sources": [{"id": "default_tool", "title": "Default"}],
                }
            ]
        }
        client = LocalMcpClient(config=config)
        # Monkeypatch the connector's pull to verify source_id construction
        captured = {}

        def fake_pull(source_id, opts=None):
            captured["source_id"] = source_id
            captured["opts"] = opts
            return {
                "title": source_id,
                "markdown": "result text",
                "source": "mcp",
                "server": "wiki",
                "tool": "fetch_knowledge",
            }

        client.connector.pull = fake_pull
        result = client.call_tool("wiki", "fetch_knowledge", arguments={"q": "test"})
        assert captured["source_id"] == "wiki:fetch_knowledge"
        assert captured["opts"] == {"q": "test"}
        assert result["markdown"] == "result text"

    def test_call_tool_without_tool_name(self):
        config = {
            "servers": [
                {
                    "name": "wiki",
                    "transport": "http",
                    "url": "http://localhost:8080",
                    "tool": "default_tool",
                }
            ]
        }
        client = LocalMcpClient(config=config)
        captured = {}

        def fake_pull(source_id, opts=None):
            captured["source_id"] = source_id
            return {"title": source_id, "markdown": "text", "source": "mcp"}

        client.connector.pull = fake_pull
        client.call_tool("wiki", "", arguments=None)
        # When tool_name is empty, source_id is just server_name
        assert captured["source_id"] == "wiki"

    def test_call_tool_with_none_arguments(self):
        config = {
            "servers": [
                {"name": "wiki", "transport": "http", "url": "http://localhost:8080"},
            ]
        }
        client = LocalMcpClient(config=config)
        captured = {}

        def fake_pull(source_id, opts=None):
            captured["opts"] = opts
            return {"title": source_id, "markdown": "text", "source": "mcp"}

        client.connector.pull = fake_pull
        client.call_tool("wiki", "tool1")
        assert captured["opts"] is None


# ---------------------------------------------------------------------------
# execute_by_id
# ---------------------------------------------------------------------------

class TestExecuteById:
    def test_execute_by_id_delegates(self):
        config = {
            "servers": [
                {"name": "wiki", "transport": "http", "url": "http://localhost:8080"},
            ]
        }
        client = LocalMcpClient(config=config)
        captured = {}

        def fake_pull(source_id, opts=None):
            captured["source_id"] = source_id
            captured["opts"] = opts
            return {"title": source_id, "markdown": "text", "source": "mcp"}

        client.connector.pull = fake_pull
        result = client.execute_by_id("wiki:tool1", opts={"key": "val"})
        assert captured["source_id"] == "wiki:tool1"
        assert captured["opts"] == {"key": "val"}
        assert result["markdown"] == "text"

    def test_execute_by_id_no_opts(self):
        config = {
            "servers": [
                {"name": "wiki", "transport": "http", "url": "http://localhost:8080"},
            ]
        }
        client = LocalMcpClient(config=config)
        captured = {}

        def fake_pull(source_id, opts=None):
            captured["opts"] = opts
            return {"title": source_id, "markdown": "text", "source": "mcp"}

        client.connector.pull = fake_pull
        client.execute_by_id("wiki:tool1")
        assert captured["opts"] is None


# ---------------------------------------------------------------------------
# test_connections
# ---------------------------------------------------------------------------

class TestTestConnections:
    def test_test_connections_delegates(self):
        config = {
            "servers": [
                {"name": "wiki", "transport": "http", "url": "http://localhost:8080"},
            ]
        }
        client = LocalMcpClient(config=config)
        result = client.test_connections()
        assert isinstance(result, dict)
        assert "success" in result
        assert "message" in result

    def test_test_connections_no_servers(self):
        client = LocalMcpClient(config={})
        result = client.test_connections()
        assert result["success"] is False
        assert "No MCP servers" in result["message"]


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_delegates(self):
        client = LocalMcpClient(config={"servers": [{"name": "s1"}]})
        # Should not raise
        client.close()

    def test_close_clears_stdio_cache(self):
        config = {
            "servers": [
                {"name": "s1", "transport": "stdio", "command": ["echo"]},
            ]
        }
        client = LocalMcpClient(config=config)
        client.close()
        assert client.connector._stdio_cache == {}


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

class TestModuleFunctions:
    def test_get_local_mcp_client(self):
        client = get_local_mcp_client()
        assert isinstance(client, LocalMcpClient)
        client.close()

    def test_list_all_mcp_tools(self):
        # With no servers configured, should return []
        tools = list_all_mcp_tools()
        assert isinstance(tools, list)

    def test_invoke_mcp_tool_not_found(self):
        # With no servers configured, pull raises ValueError
        with pytest.raises(ValueError, match="not found"):
            invoke_mcp_tool("nonexistent:tool")

    def test_invoke_mcp_tool_with_config(self, monkeypatch):
        # Build a client with an explicit config and mock get_local_mcp_client
        # to return it so invoke_mcp_tool uses our pre-configured connector.
        config = {
            "servers": [
                {"name": "wiki", "transport": "http", "url": "http://localhost:8080"},
            ]
        }
        client = LocalMcpClient(config=config)
        monkeypatch.setattr(mcp_utils, "get_local_mcp_client", lambda: client)
        # Mock the http call to avoid real network. _http_request returns
        # resp.json() for non-SSE; that dict is passed to
        # _result_to_markdown, which looks for a 'content' key.
        monkeypatch.setattr(
            "requests.post",
            lambda *a, **kw: type(
                "R",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "application/json"},
                    "json": lambda self: {
                        "content": [{"type": "text", "text": "tool result"}],
                    },
                    "text": "{}",
                },
            )(),
        )
        result = invoke_mcp_tool("wiki:fetch_knowledge", opts={"q": "test"})
        assert result["markdown"] == "tool result"
        assert result["source"] == "mcp"
        assert result["server"] == "wiki"
