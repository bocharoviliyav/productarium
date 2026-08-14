"""Unit tests for the integrations framework: MCP connector, registry, and base ABC.

Covers:
- ``api.integrations.mcp``    — HTTP + stdio transports, source_id parsing,
  result→markdown, list/pull/test/close, SSE parsing.
- ``api.integrations.registry`` — auto-discovery, get_connector, list_connectors,
  register, reset_registry.
- ``api.integrations.base``   — IntegrationConnector ABC defaults.

All network / subprocess calls are monkeypatched; no real MCP server or
subprocess is spawned.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ===========================================================================
# api.integrations.base — ABC + defaults
# ===========================================================================
class TestIntegrationConnectorBase:
    """Cover the abstract base class defaults and contract."""

    def test_defaults(self):
        from api.integrations.base import IntegrationConnector

        assert IntegrationConnector.name == ""
        assert IntegrationConnector.display_name == ""
        assert IntegrationConnector.description == ""
        assert IntegrationConnector.kind == "web"
        assert IntegrationConnector.requires_credentials is True

    def test_get_config_default_empty(self):
        from api.integrations.base import IntegrationConnector

        # Can't instantiate the ABC directly; make a minimal subclass.
        class _C(IntegrationConnector):
            def test(self): ...
            def list_spaces(self): ...
            def pull(self, source_id, opts=None): ...

        c = _C()
        assert _C.get_config() == {}
        assert c.is_configured() is False  # empty config

    def test_is_configured_true_when_config_non_empty(self):
        from api.integrations.base import IntegrationConnector

        class _C(IntegrationConnector):
            def test(self): ...
            def list_spaces(self): ...
            def pull(self, source_id, opts=None): ...

        c = _C({"url": "http://x"})
        assert c.is_configured() is True

    def test_config_is_copied_and_none_tolerant(self):
        from api.integrations.base import IntegrationConnector

        class _C(IntegrationConnector):
            def test(self): ...
            def list_spaces(self): ...
            def pull(self, source_id, opts=None): ...

        cfg = {"a": 1}
        c = _C(cfg)
        c.config["b"] = 2
        # Mutating the connector config does not leak to the caller dict.
        assert cfg == {"a": 1}
        # None config → empty dict
        assert _C(None).config == {}

    def test_cannot_instantiate_abc_directly(self):
        from api.integrations.base import IntegrationConnector

        with pytest.raises(TypeError):
            IntegrationConnector()  # type: ignore[abstract]

    def test_subclass_must_implement_abstract_methods(self):
        from api.integrations.base import IntegrationConnector

        class _Partial(IntegrationConnector):
            def test(self):
                return {}

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]


# ===========================================================================
# api.integrations.mcp — _StdioProcess (subprocess mocked)
# ===========================================================================
class TestStdioProcess:
    """Cover _StdioProcess JSON-RPC over stdin/stdout (subprocess mocked)."""

    def _make_proc(self, monkeypatch, responses: Dict[int, Any], procs_alive=None):
        """Create a _StdioProcess whose subprocess is replaced by a fake."""
        from api.integrations.mcp import _StdioProcess

        proc = _StdioProcess(["node", "s.js"], args=["--flag"], env={"FOO": "bar"})

        # Fake stdin (write + flush)
        written: List[str] = []
        stdin = MagicMock()
        stdin.write = lambda data: written.append(data)
        stdin.flush = lambda: None

        # Fake stdout: readline returns queued JSON lines from `responses`.
        stdout = MagicMock()
        stdout.readline = lambda: responses.pop(0) if responses else ""
        # Fake stderr
        stderr = MagicMock()
        stderr.readline = lambda: ""

        fake_proc = MagicMock()
        fake_proc.stdin = stdin
        fake_proc.stdout = stdout
        fake_proc.stderr = stderr
        # poll() controls liveness; default to always alive.
        fake_proc.poll = lambda: None if procs_alive is None else procs_alive
        fake_proc.terminate = lambda: None
        fake_proc.wait = lambda timeout=None: 0
        fake_proc.kill = lambda: None

        monkeypatch.setattr(proc, "_proc", fake_proc)
        proc.written = written  # type: ignore[attr-defined]
        return proc

    def test_start_idempotent_when_alive(self, monkeypatch):
        proc = self._make_proc(monkeypatch, [])
        calls = {"n": 0}

        def _popen(*a, **kw):
            calls["n"] += 1
            return MagicMock()

        monkeypatch.setattr(subprocess, "Popen", _popen)
        # Already has a live _proc → start() no-ops.
        proc.start()
        assert calls["n"] == 0

    def test_request_success_returns_result(self, monkeypatch):
        # Queue a notification (no id) then the matching response.
        notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
        proc = self._make_proc(monkeypatch, [notif, resp])
        result = proc._request("initialize", {"protocolVersion": "x"})
        assert result == {"ok": True}
        # The notification was skipped; the request line was written.
        assert "\"method\": \"initialize\"" in proc.written[0]

    def test_request_raises_on_jsonrpc_error(self, monkeypatch):
        resp = json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -32, "message": "boom"}})
        proc = self._make_proc(monkeypatch, [resp])
        with pytest.raises(ValueError, match="MCP JSON-RPC error"):
            proc._request("initialize")

    def test_request_raises_on_closed_stdout(self, monkeypatch):
        # readline returns "" → _recv returns None → IOError
        proc = self._make_proc(monkeypatch, [])
        with pytest.raises(IOError, match="closed stdout"):
            proc._request("initialize")

    def test_initialize_success_sends_notification(self, monkeypatch):
        resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}})
        proc = self._make_proc(monkeypatch, [resp])
        assert proc.initialize() is True
        assert proc._initialized is True
        # Two messages written: the request + the initialized notification.
        assert any("notifications/initialized" in w for w in proc.written)

    def test_initialize_failure_returns_false(self, monkeypatch):
        # Closed stdout → exception caught → returns False
        proc = self._make_proc(monkeypatch, [])
        assert proc.initialize() is False
        assert proc._initialized is False

    def test_list_tools_returns_tools(self, monkeypatch):
        resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "t1"}]}})
        proc = self._make_proc(monkeypatch, [resp])
        assert proc.list_tools() == [{"name": "t1"}]

    def test_list_tools_non_dict_result(self, monkeypatch):
        resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "not a dict"})
        proc = self._make_proc(monkeypatch, [resp])
        assert proc.list_tools() == []

    def test_call_tool_with_arguments(self, monkeypatch):
        resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "hi"}]}})
        proc = self._make_proc(monkeypatch, [resp])
        result = proc.call_tool("fetch", {"q": "x"})
        assert result["content"][0]["text"] == "hi"
        # Verify arguments were sent
        sent = json.loads(proc.written[0])
        assert sent["params"]["arguments"] == {"q": "x"}

    def test_close_no_proc(self):
        from api.integrations.mcp import _StdioProcess

        proc = _StdioProcess(["x"])
        proc._proc = None
        proc.close()  # no error
        assert proc._proc is None

    def test_close_terminates_live_proc(self, monkeypatch):
        proc = self._make_proc(monkeypatch, [], procs_alive=None)
        proc.close()
        assert proc._proc is None
        assert proc._initialized is False

    def test_close_kills_when_wait_fails(self, monkeypatch):
        proc = self._make_proc(monkeypatch, [], procs_alive=None)
        # Make wait raise to force the kill branch.
        proc._proc.wait = lambda timeout=None: (_ for _ in ()).throw(TimeoutError())
        killed = {"n": 0}
        proc._proc.kill = lambda: killed.__setitem__("n", killed["n"] + 1)
        proc.close()
        assert killed["n"] == 1
        assert proc._proc is None

    def test_close_handles_terminate_exception(self, monkeypatch):
        proc = self._make_proc(monkeypatch, [], procs_alive=None)
        proc._proc.terminate = lambda: (_ for _ in ()).throw(RuntimeError("nope"))
        proc.close()  # should not raise
        assert proc._proc is None


# ===========================================================================
# api.integrations.mcp — McpConnector config + parsing helpers
# ===========================================================================
class TestMcpConfigParsing:
    def test_no_config(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector()
        assert c.is_configured() is False
        assert c._servers() == []

    def test_servers_not_list(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": "not a list"})
        assert c._servers() == []
        assert c.is_configured() is False

    def test_servers_with_non_dict_entries(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": ["str", {"name": "ok"}, {"no_name": True}, 42]})
        assert [s["name"] for s in c._servers()] == ["ok"]

    def test_server_by_name_found(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "wiki"}, {"name": "fs"}]})
        assert c._server_by_name("fs") == {"name": "fs"}

    def test_server_by_name_missing(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "wiki"}]})
        assert c._server_by_name("nope") is None

    def test_parse_source_id_with_tool(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector._parse_source_id("wiki:fetch_knowledge") == ("wiki", "fetch_knowledge")

    def test_parse_source_id_tool_empty(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector._parse_source_id("wiki:") == ("wiki", None)

    def test_parse_source_id_no_colon(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector._parse_source_id("  wiki  ") == ("wiki", None)

    def test_get_config_reads_settings_store(self, isolated_db):
        from api.integrations.mcp import McpConnector
        from api.config.settings import set_setting

        set_setting("integrations.mcp", json.dumps({"servers": [{"name": "x"}]}))
        assert McpConnector.get_config() == {"servers": [{"name": "x"}]}


# ===========================================================================
# api.integrations.mcp — _result_to_markdown
# ===========================================================================
class TestResultToMarkdown:
    def test_none(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector._result_to_markdown(None) == ""

    def test_content_list(self):
        from api.integrations.mcp import McpConnector

        result = {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}
        assert McpConnector._result_to_markdown(result) == "hello\n\nworld"

    def test_content_list_skips_empty(self):
        from api.integrations.mcp import McpConnector

        result = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": ""}, {"type": "text", "text": "b"}]}
        assert McpConnector._result_to_markdown(result) == "a\n\nb"

    def test_content_text_key(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector._result_to_markdown({"text": "plain"}) == "plain"

    def test_string_result(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector._result_to_markdown("just a string") == "just a string"


# ===========================================================================
# api.integrations.mcp — SSE response parsing
# ===========================================================================
class TestSseParsing:
    def test_parse_sse_single_event(self):
        from api.integrations.mcp import McpConnector

        sse = 'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
        result = McpConnector._parse_sse_response(sse, 1)
        assert result == {"ok": True}

    def test_parse_sse_multiline_data(self):
        from api.integrations.mcp import McpConnector

        # Data split across multiple `data:` lines must be joined.
        sse = 'data: {"jsonrpc":"2.0","id":1,\ndata: "result":{"ok":true}}\n\n'
        result = McpConnector._parse_sse_response(sse, 1)
        assert result == {"ok": True}

    def test_parse_sse_error_raises(self):
        from api.integrations.mcp import McpConnector

        sse = 'data: {"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"x"}}\n\n'
        with pytest.raises(ValueError, match="MCP JSON-RPC error"):
            McpConnector._parse_sse_response(sse, 1)

    def test_parse_sse_no_matching_id(self):
        from api.integrations.mcp import McpConnector

        sse = 'data: {"jsonrpc":"2.0","id":99,"result":{"ok":true}}\n\n'
        assert McpConnector._parse_sse_response(sse, 1) is None

    def test_parse_sse_invalid_json_skipped(self):
        from api.integrations.mcp import McpConnector

        sse = 'data: not-json\n\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
        assert McpConnector._parse_sse_response(sse, 1) == {"ok": True}

    def test_parse_sse_empty(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector._parse_sse_response("", 1) is None

    def test_parse_sse_returns_obj_when_no_result_key(self):
        from api.integrations.mcp import McpConnector

        sse = 'data: {"jsonrpc":"2.0","id":1,"data":"v"}\n\n'
        assert McpConnector._parse_sse_response(sse, 1) == {"jsonrpc": "2.0", "id": 1, "data": "v"}


# ===========================================================================
# api.integrations.mcp — HTTP transport (requests.post monkeypatched)
# ===========================================================================
class _FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code=200, text="", headers=None, json_data=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self._json_data = json_data

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


class TestMcpHttpTransport:
    def _http_connector(self, **server_kw):
        from api.integrations.mcp import McpConnector

        server = {"name": "wiki", "transport": "http", **server_kw}
        return McpConnector({"servers": [server]})

    def test_http_request_json_response(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp")

        captured: Dict[str, Any] = {}

        def _post(url, headers=None, json=None, timeout=None, verify=True):
            captured.update(url=url, headers=headers, json=json, timeout=timeout, verify=verify)
            return _FakeResp(200, text='{"ok":true}', headers={"content-type": "application/json"}, json_data={"ok": True})

        import api.integrations.mcp as mod
        monkeypatch.setattr(mod, "_default_timeout", lambda: 5.0)
        monkeypatch.setattr("requests.post", _post)

        result = c._http_request("http://x/mcp", {"jsonrpc": "2.0", "id": 1})
        assert result == {"ok": True}
        assert captured["timeout"] == 5.0
        assert "application/json, text/event-stream" in captured["headers"]["Accept"]

    def test_http_request_custom_headers_merged(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp", headers={"Authorization": "Bearer t"})

        captured = {}

        def _post(url, headers=None, json=None, timeout=None, verify=True):
            captured["headers"] = headers
            return _FakeResp(200, text='{}', headers={"content-type": "application/json"}, json_data={})

        monkeypatch.setattr("requests.post", _post)
        # _http_request reads custom headers only from its ``headers`` arg — it
        # does NOT pull them from the server config (callers like
        # _http_initialize/_http_call_tool pass server.get("headers")). So pass
        # the headers explicitly here to exercise the merge with the default
        # Accept header.
        c._http_request("http://x/mcp", {"id": 1}, headers={"Authorization": "Bearer t"})
        assert captured["headers"]["Authorization"] == "Bearer t"
        # Default Accept header is still present (merged, not replaced).
        assert "application/json, text/event-stream" in captured["headers"]["Accept"]

    def test_http_request_sse_response(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp")
        sse_text = 'data: {"jsonrpc":"2.0","id":42,"result":{"v":1}}\n\n'
        resp = _FakeResp(200, text=sse_text, headers={"content-type": "text/event-stream"})

        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        result = c._http_request("http://x/mcp", {"id": 42})
        assert result == {"v": 1}

    def test_http_request_http_error_raises(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp")
        resp = _FakeResp(500, text="server error", headers={})
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        with pytest.raises(ValueError, match="MCP HTTP 500"):
            c._http_request("http://x/mcp", {"id": 1})

    def test_http_initialize_success(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp")
        resp = _FakeResp(200, text='{"ok":true}', headers={"content-type": "application/json"}, json_data={"ok": True})
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        assert c._http_initialize({"name": "wiki", "url": "http://x/mcp"}) is True

    def test_http_initialize_no_url(self):
        c = self._http_connector()
        assert c._http_initialize({"name": "wiki"}) is False

    def test_http_initialize_request_failure(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp")

        def _boom(*a, **kw):
            raise ConnectionError("down")

        monkeypatch.setattr("requests.post", _boom)
        assert c._http_initialize({"name": "wiki", "url": "http://x/mcp"}) is False

    def test_http_list_tools_success(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp")
        resp = _FakeResp(
            200,
            text='{"tools":[{"name":"t1"}]}',
            headers={"content-type": "application/json"},
            json_data={"tools": [{"name": "t1"}]},
        )
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        assert c._http_list_tools({"name": "wiki", "url": "http://x/mcp"}) == [{"name": "t1"}]

    def test_http_list_tools_no_url(self):
        c = self._http_connector()
        assert c._http_list_tools({"name": "wiki"}) == []

    def test_http_list_tools_request_failure(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp")
        monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError()))
        assert c._http_list_tools({"name": "wiki", "url": "http://x/mcp"}) == []

    def test_http_call_tool_success(self, monkeypatch):
        c = self._http_connector(url="http://x/mcp")
        resp = _FakeResp(
            200,
            text='{"content":[{"type":"text","text":"data"}]}',
            headers={"content-type": "application/json"},
            json_data={"content": [{"type": "text", "text": "data"}]},
        )
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        result = c._http_call_tool({"name": "wiki", "url": "http://x/mcp"}, "fetch", {"q": "x"})
        assert result["content"][0]["text"] == "data"

    def test_http_call_tool_no_url_raises(self):
        c = self._http_connector()
        with pytest.raises(ValueError, match="no url"):
            c._http_call_tool({"name": "wiki"}, "fetch")


# ===========================================================================
# api.integrations.mcp — test() / list_spaces() / pull()
# ===========================================================================
class TestMcpConnectorInterface:
    def test_test_no_servers(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({})
        result = c.test()
        assert result["success"] is False
        assert "No MCP servers" in result["message"]

    def test_test_invalid_transport(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "bad", "transport": "ftp"}]})
        result = c.test()
        assert result["success"] is False
        assert result["servers"][0]["success"] is False
        assert "Invalid transport" in result["servers"][0]["message"]

    def test_test_http_ok(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "wiki", "transport": "http", "url": "http://x/mcp"}]})
        resp = _FakeResp(200, text='{"ok":true}', headers={"content-type": "application/json"}, json_data={"ok": True})
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        result = c.test()
        assert result["success"] is True
        assert result["servers"][0]["success"] is True

    def test_test_http_fail(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "wiki", "transport": "http", "url": "http://x/mcp"}]})
        monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError()))
        result = c.test()
        assert result["success"] is False
        assert result["servers"][0]["success"] is False

    def test_test_stdio_ok(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]})

        # Monkeypatch _get_stdio to return a fake process.
        fake_proc = MagicMock()
        fake_proc._initialized = True
        monkeypatch.setattr(c, "_get_stdio", lambda server: fake_proc)
        result = c.test()
        assert result["success"] is True
        assert result["servers"][0]["success"] is True

    def test_test_stdio_unreachable(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]})
        monkeypatch.setattr(c, "_get_stdio", lambda server: None)
        result = c.test()
        assert result["success"] is False
        assert result["servers"][0]["success"] is False

    def test_test_mixed_servers(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [
                {"name": "bad", "transport": "ftp"},
                {"name": "wiki", "transport": "http", "url": "http://x/mcp"},
            ]
        })
        resp = _FakeResp(200, text='{"ok":true}', headers={"content-type": "application/json"}, json_data={"ok": True})
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        result = c.test()
        assert result["success"] is True  # at least one ok
        assert "1/2" in result["message"]

    # --- list_spaces ---
    def test_list_spaces_explicit_sources(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{
                "name": "wiki",
                "sources": [{"id": "s1", "title": "Wiki", "type": "mcp_source"}],
            }]
        })
        spaces = c.list_spaces()
        assert len(spaces) == 1
        assert spaces[0]["id"] == "wiki:s1"
        assert spaces[0]["title"] == "Wiki"
        assert spaces[0]["server"] == "wiki"

    def test_list_spaces_explicit_sources_missing_fields(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{
                "name": "wiki",
                "sources": [{"title": "NoId"}],  # no id → falls back to name, then server name
            }]
        })
        spaces = c.list_spaces()
        assert spaces[0]["id"] == "wiki:"
        assert spaces[0]["title"] == "NoId"

    def test_list_spaces_http_tools(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "wiki", "transport": "http", "url": "http://x/mcp"}]
        })
        resp = _FakeResp(
            200,
            text='{"tools":[{"name":"t1","description":"Tool one"}]}',
            headers={"content-type": "application/json"},
            json_data={"tools": [{"name": "t1", "description": "Tool one"}]},
        )
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        spaces = c.list_spaces()
        assert spaces[0]["id"] == "wiki:t1"
        assert spaces[0]["title"] == "Tool one"
        assert spaces[0]["type"] == "mcp_tool"

    def test_list_spaces_stdio_tools(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]
        })
        fake_proc = MagicMock()
        fake_proc.list_tools = lambda: [{"name": "read_file", "description": "Read a file"}]
        monkeypatch.setattr(c, "_get_stdio", lambda server: fake_proc)
        spaces = c.list_spaces()
        assert spaces[0]["id"] == "fs:read_file"
        assert spaces[0]["title"] == "Read a file"

    def test_list_spaces_stdio_tools_failure(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]
        })
        monkeypatch.setattr(c, "_get_stdio", lambda server: None)
        assert c.list_spaces() == []

    def test_list_spaces_stdio_list_tools_raises(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]
        })
        fake_proc = MagicMock()
        fake_proc.list_tools = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        monkeypatch.setattr(c, "_get_stdio", lambda server: fake_proc)
        assert c.list_spaces() == []

    def test_list_spaces_unknown_transport(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "x", "transport": "ftp"}]})
        assert c.list_spaces() == []

    def test_list_spaces_exception_caught(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "x", "transport": "http", "url": "http://x"}]})
        monkeypatch.setattr(c, "_server_list_sources", lambda s: (_ for _ in ()).throw(ValueError("boom")))
        assert c.list_spaces() == []

    # --- pull ---
    def test_pull_server_not_found(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "wiki"}]})
        with pytest.raises(ValueError, match="not found"):
            c.pull("nope")

    def test_pull_http_success(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "wiki", "transport": "http", "url": "http://x/mcp", "tool": "fetch_knowledge"}]
        })
        resp = _FakeResp(
            200,
            text='{"content":[{"type":"text","text":"page content"}]}',
            headers={"content-type": "application/json"},
            json_data={"content": [{"type": "text", "text": "page content"}]},
        )
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        result = c.pull("wiki:custom_tool", opts={"depth": 2})
        assert result["title"] == "wiki:custom_tool"
        assert result["markdown"] == "page content"
        assert result["server"] == "wiki"
        assert result["tool"] == "custom_tool"
        assert result["transport"] == "http"

    def test_pull_uses_default_tool(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "wiki", "transport": "http", "url": "http://x/mcp", "tool": "default_tool"}]
        })
        resp = _FakeResp(
            200,
            text='{"content":[{"type":"text","text":"x"}]}',
            headers={"content-type": "application/json"},
            json_data={"content": [{"type": "text", "text": "x"}]},
        )
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        result = c.pull("wiki")  # no tool in source_id
        assert result["tool"] == "default_tool"

    def test_pull_empty_result(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "wiki", "transport": "http", "url": "http://x/mcp"}]
        })
        resp = _FakeResp(
            200,
            text='{"content":[]}',
            headers={"content-type": "application/json"},
            json_data={"content": []},
        )
        monkeypatch.setattr("requests.post", lambda *a, **kw: resp)
        result = c.pull("wiki")
        assert "returned no text" in result["markdown"]

    def test_pull_stdio_success(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"], "tool": "read_file"}]
        })
        fake_proc = MagicMock()
        fake_proc.call_tool = lambda name, args: {"content": [{"type": "text", "text": "file contents"}]}
        monkeypatch.setattr(c, "_get_stdio", lambda server: fake_proc)
        result = c.pull("fs")
        assert result["markdown"] == "file contents"
        assert result["transport"] == "stdio"

    def test_pull_stdio_unreachable(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({
            "servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]
        })
        monkeypatch.setattr(c, "_get_stdio", lambda server: None)
        with pytest.raises(ValueError, match="not reachable"):
            c.pull("fs")

    def test_pull_unknown_transport(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "x", "transport": "ftp"}]})
        with pytest.raises(ValueError, match="Unknown transport"):
            c.pull("x")

    # --- _server_call / _get_stdio ---
    def test_get_stdio_no_command(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "fs", "transport": "stdio"}]})
        assert c._get_stdio({"name": "fs", "transport": "stdio"}) is None

    def test_get_stdio_start_failure(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "fs", "transport": "stdio", "command": ["badcmd"]}]})
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no")))
        assert c._get_stdio({"name": "fs", "transport": "stdio", "command": ["badcmd"]}) is None

    def test_get_stdio_initialize_failure(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]})
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("api.integrations.mcp._StdioProcess.initialize", lambda self: False)
        assert c._get_stdio({"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}) is None

    def test_get_stdio_caches_live_process(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]})
        fake_proc = MagicMock()
        fake_proc._proc = MagicMock()
        fake_proc._proc.poll = lambda: None  # alive
        c._stdio_cache["fs"] = fake_proc
        result = c._get_stdio({"name": "fs", "transport": "stdio", "command": ["node", "s.js"]})
        assert result is fake_proc

    def test_get_stdio_recreates_dead_process(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]})
        dead_proc = MagicMock()
        dead_proc._proc = MagicMock()
        dead_proc._proc.poll = lambda: 1  # dead
        c._stdio_cache["fs"] = dead_proc

        # The recreation path: Popen succeeds, initialize returns True.
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("api.integrations.mcp._StdioProcess.initialize", lambda self: True)
        result = c._get_stdio({"name": "fs", "transport": "stdio", "command": ["node", "s.js"]})
        assert result is not None
        assert "fs" in c._stdio_cache

    # --- close / __del__ ---
    def test_close_cleans_cache(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({"servers": [{"name": "fs", "transport": "stdio", "command": ["node", "s.js"]}]})
        fake_proc = MagicMock()
        c._stdio_cache["fs"] = fake_proc
        c.close()
        assert c._stdio_cache == {}
        fake_proc.close.assert_called_once()

    def test_close_handles_exceptions(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({})
        bad_proc = MagicMock()
        bad_proc.close = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        c._stdio_cache["bad"] = bad_proc
        c.close()  # should not raise
        assert c._stdio_cache == {}

    def test_del_calls_close(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector({})
        c.close = MagicMock()  # type: ignore[method-assign]
        c.__del__()
        c.close.assert_called_once()

    def test_del_swallows_exceptions(self):
        from api.integrations.mcp import McpConnector

        c = McpConnector.__new__(McpConnector)
        c._stdio_cache = {}
        # close() will raise → __del__ swallows.
        c.close = lambda: (_ for _ in ()).throw(RuntimeError("x"))  # type: ignore[method-assign]
        c.__del__()  # should not raise


# ===========================================================================
# api.integrations.registry — auto-discovery + get/list/register/reset
# ===========================================================================
class TestRegistry:
    def test_get_connector_class_known(self):
        from api.integrations.registry import get_connector_class, reset_registry

        reset_registry()
        cls = get_connector_class("mcp")
        assert cls is not None
        assert cls.name == "mcp"

    def test_get_connector_class_unknown(self):
        from api.integrations.registry import get_connector_class, reset_registry

        reset_registry()
        assert get_connector_class("does_not_exist") is None

    def test_get_connector_instantiates(self, isolated_db, monkeypatch):
        from api.integrations.registry import get_connector, reset_registry

        reset_registry()
        # Make get_config return empty so the instance is lightweight.
        monkeypatch.setattr("api.integrations.mcp.McpConnector.get_config", classmethod(lambda cls: {}))
        conn = get_connector("mcp")
        assert conn is not None
        assert conn.name == "mcp"

    def test_get_connector_unknown_returns_none(self):
        from api.integrations.registry import get_connector, reset_registry

        reset_registry()
        assert get_connector("nope") is None

    def test_get_connector_config_failure_is_safe(self, isolated_db, monkeypatch):
        from api.integrations.registry import get_connector, reset_registry

        reset_registry()

        def _bad_config(cls):
            raise RuntimeError("settings store down")

        monkeypatch.setattr("api.integrations.mcp.McpConnector.get_config", classmethod(_bad_config))
        conn = get_connector("mcp")
        assert conn is not None
        assert conn.config == {}

    def test_list_connectors_includes_known(self):
        from api.integrations.registry import list_connectors, reset_registry

        reset_registry()
        connectors = list_connectors()
        names = [c["name"] for c in connectors]
        assert "mcp" in names
        assert "confluence" in names
        # Each entry has the full shape.
        mcp = [c for c in connectors if c["name"] == "mcp"][0]
        assert "display_name" in mcp
        assert "description" in mcp
        assert "kind" in mcp
        assert "requires_credentials" in mcp
        assert "configured" in mcp

    def test_list_connectors_sorted(self):
        from api.integrations.registry import list_connectors, reset_registry

        reset_registry()
        connectors = list_connectors()
        names = [c["name"] for c in connectors]
        assert names == sorted(names)

    def test_register_decorator(self):
        from api.integrations.base import IntegrationConnector
        from api.integrations.registry import register, reset_registry, get_connector_class

        reset_registry()

        @register
        class _MyConn(IntegrationConnector):
            name = "my_custom"
            display_name = "Custom"
            description = "test"
            kind = "web"
            requires_credentials = False

            def test(self):
                return {"success": True, "message": "ok"}

            def list_spaces(self):
                return []

            def pull(self, source_id, opts=None):
                return {}

        try:
            assert get_connector_class("my_custom") is _MyConn
        finally:
            reset_registry()

    def test_register_requires_name(self):
        from api.integrations.base import IntegrationConnector
        from api.integrations.registry import register, reset_registry

        reset_registry()

        class _NoName(IntegrationConnector):
            def test(self): ...
            def list_spaces(self): ...
            def pull(self, source_id, opts=None): ...

        with pytest.raises(ValueError, match="no `name`"):
            register(_NoName)
        reset_registry()

    def test_reset_registry_forces_rediscovery(self):
        import api.integrations.registry as reg
        from api.integrations.registry import get_connector_class, reset_registry

        reset_registry()
        # Read the LIVE module attribute (reg._DISCOVERED), NOT a `from ... import
        # _DISCOVERED` snapshot — reset_registry() and _autodiscover() REBIND the
        # module global, so a value-imported name would stay stale.
        assert reg._DISCOVERED is False
        get_connector_class("mcp")  # triggers discovery
        assert reg._DISCOVERED is True
        assert "mcp" in reg._REGISTRY
        reset_registry()
        assert reg._REGISTRY == {}
        assert reg._DISCOVERED is False

    def test_list_connectors_is_configured_safe_on_failure(self, isolated_db, monkeypatch):
        from api.integrations.registry import list_connectors, reset_registry

        reset_registry()

        # Break is_configured for mcp to exercise the except branch.
        def _broken(self):
            raise RuntimeError("boom")

        monkeypatch.setattr("api.integrations.mcp.McpConnector.is_configured", _broken)
        connectors = list_connectors()
        mcp = [c for c in connectors if c["name"] == "mcp"][0]
        # Falls back to bool(config).
        assert mcp["configured"] in (True, False)  # didn't crash

    def test_autodiscover_skips_underscore_and_base_modules(self):
        from api.integrations.registry import reset_registry, _REGISTRY

        reset_registry()
        # Trigger discovery by importing via the public API.
        from api.integrations import list_connectors  # noqa: F401

        list_connectors()
        # _git_base is underscore-prefixed → not registered as a connector.
        assert "_git_base" not in _REGISTRY
        assert "base" not in _REGISTRY
        assert "registry" not in _REGISTRY


# ===========================================================================
# api.integrations package __init__ re-exports
# ===========================================================================
class TestPackageInit:
    def test_reexports(self):
        import api.integrations as pkg

        assert hasattr(pkg, "IntegrationConnector")
        assert hasattr(pkg, "get_connector")
        assert hasattr(pkg, "list_connectors")
        assert hasattr(pkg, "register")
        assert hasattr(pkg, "reset_registry")
        assert "IntegrationConnector" in pkg.__all__
