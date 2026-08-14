"""MCP (Model Context Protocol) connector — multi-server adapter (plan G).

Supports **multiple** MCP servers configured via the settings store key
``integrations.mcp``. Each server uses either an ``http`` (Streamable HTTP /
SSE) or ``stdio`` (subprocess JSON-RPC over stdin/stdout) transport.

Config shape (stored as JSON under ``integrations.mcp``)::

    {
      "servers": [
        {
          "name": "wiki",
          "transport": "http",
          "url": "http://localhost:8080/mcp",
          "headers": {"Authorization": "Bearer x"},
          "tool": "fetch_knowledge",
          "sources": [{"id": "s1", "title": "Wiki", "type": "mcp_source"}]
        },
        {
          "name": "fs",
          "transport": "stdio",
          "command": ["node", "mcp-server.js"],
          "args": [],
          "env": {"FOO": "bar"},
          "tool": "read_file",
          "sources": [{"id": "s2", "title": "Filesystem", "type": "mcp_source"}]
        }
      ]
    }

Contract:
- ``test()``        — pings every server (initialize + notifications/initialized
  for stdio); returns ``{success, message, servers: [...]}``.
- ``list_spaces()`` — for each server: if ``sources`` are declared explicitly
  they are returned as-is; otherwise the server's ``tools/list`` is called and
  each tool becomes a source.
- ``pull(source_id, opts)`` — ``source_id`` encodes ``serverName:toolName`` (or
  just ``serverName`` to use the server's default ``tool``). Calls
  ``tools/call`` and returns the text result as markdown.
- ``is_configured()`` — True when at least one server is configured.

All network / subprocess calls are wrapped in try/except so the connector
degrades gracefully. Subprocesses are cached per-connector-instance and
terminated on close / GC.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional

from api.integrations.base import IntegrationConnector

logger = logging.getLogger(__name__)

_VALID_TRANSPORTS = ("http", "stdio")
# MCP protocol version we advertise during initialize.
_MCP_PROTOCOL_VERSION = "2024-11-05"
_MCP_CLIENT_INFO = {"name": "productarium", "version": "1.0"}
# Default timeout (seconds) for JSON-RPC calls. Resolved lazily through the
# central timeout config (admin > env > default) so an admin save takes effect
# without a restart. Kept as a helper (not a module constant) because the
# central resolver is read-through.
def _default_timeout() -> float:
    try:
        from api.config.timeout import resolve_integration_http_timeout
        return resolve_integration_http_timeout()
    except Exception:
        return 30.0


class _StdioProcess:
    """A managed MCP stdio subprocess speaking newline-delimited JSON-RPC.

    MCP stdio transport frames each message as a single JSON object followed by
    a newline. We write requests to stdin and read responses from stdout.
    """

    def __init__(
        self,
        command: List[str],
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._cmd = list(command) + list(args or [])
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        self._proc: Optional[subprocess.Popen] = None
        self._env = full_env
        self._lock = threading.Lock()
        self._next_id = 1
        self._initialized = False

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            text=True,
            bufsize=1,  # line-buffered
        )
        logger.info("MCP stdio process started: %s", self._cmd)

    def _send(self, message: Dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> Optional[Dict[str, Any]]:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            return None
        return json.loads(line)

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC request and read the matching response.

        Notifications from the server (no ``id``) are skipped until we get the
        response with our ``id``.
        """
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self.start()
            msg_id = self._next_id
            self._next_id += 1
            payload: Dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": method,
            }
            if params is not None:
                payload["params"] = params
            self._send(payload)
            # Read lines until we get the response for our id.
            while True:
                resp = self._recv()
                if resp is None:
                    raise IOError("MCP stdio process closed stdout unexpectedly")
                if resp.get("id") == msg_id:
                    if "error" in resp:
                        err = resp["error"]
                        raise ValueError(
                            f"MCP JSON-RPC error {err.get('code')}: {err.get('message')}"
                        )
                    return resp.get("result")
                # A notification or unrelated message — skip.

    def initialize(self) -> bool:
        """Perform the MCP initialize handshake."""
        try:
            result = self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _MCP_CLIENT_INFO,
                },
            )
            # Send the initialized notification (no id, no response expected).
            try:
                self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            except Exception:
                pass
            self._initialized = result is not None
            return self._initialized
        except Exception as e:
            logger.warning("MCP stdio initialize failed: %s", e)
            return False

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self._request("tools/list", {})
        if not isinstance(result, dict):
            return []
        return result.get("tools") or []

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        params: Dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return self._request("tools/call", params)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    from api.config.timeout import resolve_mcp_stdio_wait_timeout
                    self._proc.wait(timeout=resolve_mcp_stdio_wait_timeout())
                except Exception:
                    self._proc.kill()
        except Exception:
            pass
        finally:
            self._proc = None
            self._initialized = False


class McpConnector(IntegrationConnector):
    name = "mcp"
    display_name = "MCP Servers"
    description = (
        "Pull knowledge from one or more Model Context Protocol servers "
        "(http/SSE and stdio transports)."
    )
    kind = "mcp"
    requires_credentials = False  # config-driven; no admin creds strictly required

    # Per-instance stdio process cache: {server_name: _StdioProcess}.
    # Processes are reused across calls within the connector lifetime and
    # terminated on close().
    _stdio_cache: Dict[str, _StdioProcess]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._stdio_cache = {}

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        from api.config.settings import get_integration_config

        return get_integration_config("mcp")

    # ---- server config helpers -------------------------------------------
    def _servers(self) -> List[Dict[str, Any]]:
        cfg = self.config or {}
        servers = cfg.get("servers")
        if not isinstance(servers, list):
            return []
        out: List[Dict[str, Any]] = []
        for s in servers:
            if isinstance(s, dict) and s.get("name"):
                out.append(s)
        return out

    def _server_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for s in self._servers():
            if s.get("name") == name:
                return s
        return None

    def is_configured(self) -> bool:
        return len(self._servers()) > 0

    # ---- source_id parsing -----------------------------------------------
    @staticmethod
    def _parse_source_id(source_id: str) -> tuple:
        """Parse ``server:tool`` or ``server`` into (server_name, tool_or_None)."""
        if ":" in source_id:
            server, tool = source_id.split(":", 1)
            return server.strip(), tool.strip() or None
        return source_id.strip(), None

    # ---- stdio process management ----------------------------------------
    def _get_stdio(self, server: Dict[str, Any]) -> Optional[_StdioProcess]:
        name = server.get("name", "")
        if name in self._stdio_cache:
            proc = self._stdio_cache[name]
            if proc._proc is not None and proc._proc.poll() is None:
                return proc
            # Dead process — remove and recreate.
            proc.close()
            del self._stdio_cache[name]
        command = server.get("command") or []
        if not command:
            return None
        proc = _StdioProcess(
            command=list(command),
            args=server.get("args") or [],
            env=server.get("env") or {},
        )
        try:
            proc.start()
            if proc.initialize():
                self._stdio_cache[name] = proc
                return proc
            else:
                proc.close()
                return None
        except Exception as e:
            logger.warning("MCP stdio start for %r failed: %s", name, e)
            proc.close()
            return None

    # ---- http transport --------------------------------------------------
    def _http_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC POST and return the parsed response dict."""
        import requests
        from api.config.ssl import requests_verify

        if timeout is None:
            timeout = _default_timeout()
        hdrs = {"Accept": "application/json, text/event-stream"}
        if headers:
            hdrs.update(headers)
        resp = requests.post(
            url, headers=hdrs, json=payload, timeout=timeout,
            verify=requests_verify(),
        )
        if resp.status_code >= 400:
            raise ValueError(f"MCP HTTP {resp.status_code}: {resp.text[:200]}")
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            return self._parse_sse_response(resp.text, payload.get("id"))
        return resp.json()

    @staticmethod
    def _parse_sse_response(text: str, expected_id: Any) -> Optional[Dict[str, Any]]:
        """Parse an SSE event stream for the JSON-RPC response matching ``expected_id``."""
        for event_block in text.split("\n\n"):
            data_lines = []
            for line in event_block.split("\n"):
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if not data_lines:
                continue
            data_str = "\n".join(data_lines)
            try:
                obj = json.loads(data_str)
            except Exception:
                continue
            if isinstance(obj, dict) and (expected_id is None or obj.get("id") == expected_id):
                if "error" in obj:
                    err = obj["error"]
                    raise ValueError(
                        f"MCP JSON-RPC error {err.get('code')}: {err.get('message')}"
                    )
                return obj.get("result") if "result" in obj else obj
        return None

    def _http_initialize(self, server: Dict[str, Any]) -> bool:
        url = server.get("url")
        if not url:
            return False
        try:
            result = self._http_request(
                url,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": _MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": _MCP_CLIENT_INFO,
                    },
                },
                headers=server.get("headers"),
                timeout=_default_timeout(),
            )
            return result is not None
        except Exception as e:
            logger.warning("MCP http initialize for %r failed: %s", server.get("name"), e)
            return False

    def _http_list_tools(self, server: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = server.get("url")
        if not url:
            return []
        try:
            result = self._http_request(
                url,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                },
                headers=server.get("headers"),
            )
            if isinstance(result, dict):
                return result.get("tools") or []
        except Exception as e:
            logger.warning("MCP http tools/list for %r failed: %s", server.get("name"), e)
        return []

    def _http_call_tool(
        self, server: Dict[str, Any], tool: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        url = server.get("url")
        if not url:
            raise ValueError("MCP http server has no url.")
        params: Dict[str, Any] = {"name": tool}
        if arguments is not None:
            params["arguments"] = arguments
        return self._http_request(
            url,
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": params},
            headers=server.get("headers"),
        )

    # ---- tool result → markdown ------------------------------------------
    @staticmethod
    def _result_to_markdown(result: Any) -> str:
        if result is None:
            return ""
        # MCP tool result: {content: [{type: "text", text: "..."}, ...]}
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                texts = []
                for c in content:
                    if isinstance(c, dict):
                        texts.append(str(c.get("text", "")))
                return "\n\n".join(t for t in texts if t)
            if "text" in result:
                return str(result["text"])
        return str(result)

    # ---- per-server list/call dispatch -----------------------------------
    def _server_list_sources(self, server: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return sources for a single server (explicit or via tools/list)."""
        # Explicit sources take priority.
        srcs = server.get("sources")
        if isinstance(srcs, list) and srcs:
            out: List[Dict[str, Any]] = []
            for s in srcs:
                if isinstance(s, dict):
                    out.append(
                        {
                            "id": f"{server['name']}:{s.get('id') or s.get('name') or ''}",
                            "title": s.get("title") or s.get("id") or server["name"],
                            "type": s.get("type") or "mcp_source",
                            "server": server["name"],
                        }
                    )
            return out
        # Fall back to live tools/list.
        transport = server.get("transport")
        tools: List[Dict[str, Any]] = []
        if transport == "stdio":
            proc = self._get_stdio(server)
            if proc is not None:
                try:
                    tools = proc.list_tools()
                except Exception as e:
                    logger.warning("MCP stdio tools/list for %r failed: %s", server["name"], e)
        elif transport == "http":
            tools = self._http_list_tools(server)
        out = []
        for t in tools:
            if isinstance(t, dict):
                tname = t.get("name", "")
                out.append(
                    {
                        "id": f"{server['name']}:{tname}",
                        "title": t.get("description") or tname or server["name"],
                        "type": "mcp_tool",
                        "server": server["name"],
                    }
                )
        return out

    def _server_call(
        self, server: Dict[str, Any], tool: str, arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        transport = server.get("transport")
        if transport == "stdio":
            proc = self._get_stdio(server)
            if proc is None:
                raise ValueError(
                    f"MCP stdio server {server.get('name')} not reachable."
                )
            result = proc.call_tool(tool, arguments)
            return self._result_to_markdown(result)
        elif transport == "http":
            result = self._http_call_tool(server, tool, arguments)
            return self._result_to_markdown(result)
        raise ValueError(f"Unknown transport {transport!r} for server {server.get('name')}")

    # ---- IntegrationConnector interface ----------------------------------
    def test(self) -> Dict[str, Any]:
        servers = self._servers()
        if not servers:
            return {"success": False, "message": "No MCP servers configured."}
        results: List[Dict[str, Any]] = []
        any_ok = False
        for s in servers:
            name = s.get("name", "?")
            transport = s.get("transport")
            entry: Dict[str, Any] = {"name": name, "transport": transport}
            if transport not in _VALID_TRANSPORTS:
                entry["success"] = False
                entry["message"] = f"Invalid transport {transport!r}."
                results.append(entry)
                continue
            if transport == "http":
                ok = self._http_initialize(s)
            else:  # stdio
                proc = self._get_stdio(s)
                ok = proc is not None and proc._initialized
            entry["success"] = ok
            entry["message"] = "reachable" if ok else "unreachable"
            results.append(entry)
            if ok:
                any_ok = True
        overall_msg = f"{sum(1 for r in results if r.get('success'))}/{len(results)} server(s) reachable."
        return {"success": any_ok, "message": overall_msg, "servers": results}

    def list_spaces(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in self._servers():
            try:
                out.extend(self._server_list_sources(s))
            except Exception as e:
                logger.warning("MCP list_spaces for %r failed: %s", s.get("name"), e)
        return out

    def pull(self, source_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        server_name, tool_override = self._parse_source_id(source_id)
        server = self._server_by_name(server_name)
        if server is None:
            raise ValueError(f"MCP server {server_name!r} not found in config.")
        tool = tool_override or server.get("tool", "fetch_knowledge")
        # Build arguments: pass source_id / opts through.
        arguments: Dict[str, Any] = {"source_id": source_id}
        if opts and isinstance(opts, dict):
            arguments.update(opts)
        text = self._server_call(server, tool, arguments)
        return {
            "title": source_id,
            "markdown": text or f"<!-- MCP tool {tool} returned no text for {source_id!r} -->\n",
            "attachments": [],
            "source": "mcp",
            "server": server_name,
            "tool": tool,
            "transport": server.get("transport"),
        }

    # ---- lifecycle -------------------------------------------------------
    def close(self) -> None:
        for proc in self._stdio_cache.values():
            try:
                proc.close()
            except Exception:
                pass
        self._stdio_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
