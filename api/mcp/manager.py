"""Outbound MCP tool manager (Wave C).

A process-wide cache/pool over :class:`langchain_mcp_adapters.client.MultiServerMCPClient`
for the MCP servers registered by admins (``McpServerORM``) and bound to
products (``ProductMcpServerORM``).

Design:
- **Per-server clients** keyed by server id + config fingerprint (transport,
  url/command/args, headers/env ciphertext, updated_at) — any config change
  produces a new fingerprint, so the cached client is dropped automatically.
- **Discovery cache**: ``tools/list`` results per server (same fingerprint
  key), so repeated agent builds / the admin "tools" endpoint never
  re-connect. The cache also stores a small ``[{name, description}]`` meta
  list for ``GET /api/admin/mcp/servers/{id}/tools`` (no reconnect).
- **Timeouts**: every connect/discovery is wrapped in ``asyncio.wait_for``
  (default 10 s, ``MCP_DISCOVERY_TIMEOUT_SECONDS`` env override) so a dead
  server can never hang a request; http connections additionally carry the
  same timeout at the transport level.
- **Allowlist**: ``ProductMcpServerORM.allowed_tools`` is applied by tool
  NAME after discovery, per binding — ``None`` means ALL tools, an explicit
  list means ONLY those tools (so ``[]`` deliberately exposes NO tools).
- **Parallel + negatively cached discovery**: ``get_tools_for_product``
  discovers all bound servers concurrently (a dead server costs its own
  timeout, not the sum of all timeouts) and remembers failed discoveries
  for ``MCP_NEGATIVE_CACHE_SECONDS`` (default 30 s) so known-dead servers
  are not re-dialed on every agent build. ``health_check`` always bypasses
  the negative cache (the admin "test" button must be truthful).
- **Bounded tool calls**: tools handed to the agent are wrapped with a
  per-call timeout (``MCP_TOOL_CALL_TIMEOUT_SECONDS``, default 60 s) and a
  result size cap (``MCP_TOOL_RESULT_MAX_CHARS``, default 100 000) — a slow
  or chatty external server can neither hang the agent nor flood its LLM
  context. A timed-out call returns a short error string, never raises.
- **stdio policy**: shells/interpreters/runners and behavior-shaping env
  (``PATH``, ``LD_*`` …) are rejected at connect time too
  (``api/mcp/policy.py``), defending rows registered before the policy.
- **Health-check**: :meth:`McpToolManager.health_check` performs one bounded
  discovery attempt and returns ``(ok, short_detail, tool_meta)`` — the admin
  router persists it into ``status``/``status_checked_at``/``status_error``.
  Error details are sanitized (class name + short, URL-stripped reason; the
  full traceback goes to the log with ``exc_info``).
- **Best-effort semantics**: :meth:`get_tools_for_product` skips a server that
  fails to connect (logged, never raised) so an unreachable external server
  cannot break the expert agent.

All methods are import-safe when the DB is down (the DB-reading helpers raise
to their callers, which wrap them best-effort) and never touch the network at
import time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from api.mcp.policy import stdio_command_error, stdio_env_error
from api.mcp.secrets import decrypt_secret_dict

logger = logging.getLogger(__name__)

#: Bounded time for one connect+tools/list attempt (seconds).
DEFAULT_DISCOVERY_TIMEOUT = 10.0
#: Bounded time for one outbound MCP tool CALL (seconds).
DEFAULT_TOOL_CALL_TIMEOUT = 60.0
#: Max characters of a tool result that may enter the agent/LLM context.
DEFAULT_TOOL_RESULT_MAX_CHARS = 100_000
#: How long a failed discovery stays negatively cached (seconds; 0 disables).
DEFAULT_NEGATIVE_CACHE_TTL = 30.0


def discovery_timeout() -> float:
    """Resolve the discovery timeout (env override, floor 1s, never raises)."""
    try:
        raw = float(os.environ.get("MCP_DISCOVERY_TIMEOUT_SECONDS", "") or DEFAULT_DISCOVERY_TIMEOUT)
    except (TypeError, ValueError):
        return DEFAULT_DISCOVERY_TIMEOUT
    return max(1.0, raw)


def tool_call_timeout() -> float:
    """Resolve the tool-call timeout (env override, floor 0.1s, never raises)."""
    try:
        raw = float(os.environ.get("MCP_TOOL_CALL_TIMEOUT_SECONDS", "") or DEFAULT_TOOL_CALL_TIMEOUT)
    except (TypeError, ValueError):
        return DEFAULT_TOOL_CALL_TIMEOUT
    return max(0.1, raw)


def tool_result_max_chars() -> int:
    """Resolve the tool-result size cap (env override, floor 1000, never raises)."""
    try:
        raw = int(os.environ.get("MCP_TOOL_RESULT_MAX_CHARS", "") or DEFAULT_TOOL_RESULT_MAX_CHARS)
    except (TypeError, ValueError):
        return DEFAULT_TOOL_RESULT_MAX_CHARS
    return max(1000, raw)


def negative_cache_ttl() -> float:
    """Resolve the negative-cache TTL (env override, floor 0s, never raises)."""
    try:
        raw = float(os.environ.get("MCP_NEGATIVE_CACHE_SECONDS", "") or DEFAULT_NEGATIVE_CACHE_TTL)
    except (TypeError, ValueError):
        return DEFAULT_NEGATIVE_CACHE_TTL
    return max(0.0, raw)


def server_fingerprint(server: Any) -> str:
    """Stable fingerprint of everything that affects the connection/tools."""
    return "|".join(
        [
            str(getattr(server, "transport", "")),
            str(getattr(server, "url", "") or ""),
            str(getattr(server, "command", "") or ""),
            repr(getattr(server, "args", None)),
            str(getattr(server, "headers", None) or ""),
            str(getattr(server, "env", None) or ""),
            str(getattr(server, "updated_at", "") or ""),
        ]
    )


def build_connection(server: Any) -> Dict[str, Any]:
    """Build a langchain-mcp-adapters Connection dict for an McpServerORM row.

    Decrypts headers/env (secrets) only here, on the way OUT to the client —
    never stored, never logged.
    """
    timeout = discovery_timeout()
    if server.transport in ("http", "sse"):
        conn: Dict[str, Any] = {
            "transport": server.transport,
            "url": server.url,
            "timeout": timeout,
        }
        headers = decrypt_secret_dict(server.headers)
        if headers:
            conn["headers"] = headers
        return conn
    if server.transport == "stdio":
        env = decrypt_secret_dict(server.env)
        preset_key = getattr(server, "preset_key", None)
        if preset_key:
            # System-managed preset row (api/mcp/presets.py): the interpreter
            # ban does not apply (dbhub needs Node, oracle needs a venv
            # python). Instead the FULL row — command + args + env — must
            # match the hardcoded registry exactly (rebuilt from the stored
            # DSN), so tampering the row still cannot execute anything
            # outside the preset launchers. Env keys keep the policy check.
            from api.mcp.presets import preset_row_mismatch

            error = preset_row_mismatch(
                preset_key, server.command, server.args or [], env
            )
            if error:
                raise ValueError(error)
        else:
            # Defense in depth: rows registered before the router policy
            # existed are still refused here (best-effort skip upstream),
            # never spawned.
            error = stdio_command_error(server.command)
            if error:
                raise ValueError(error)
        if env:
            error = stdio_env_error(env)
            if error:
                raise ValueError(error)
        conn = {
            "transport": "stdio",
            "command": server.command,
            "args": list(server.args or []),
        }
        if env:
            conn["env"] = env
        return conn
    raise ValueError(f"unsupported MCP transport: {server.transport!r}")


def _short_error(exc: BaseException) -> str:
    """Short, sanitized error description for API responses (no internals).

    Exception messages from HTTP clients can embed full URLs (which may carry
    credentials in query params), so only the exception class name plus a
    heavily truncated message is returned; the full context goes to the log.
    """
    name = type(exc).__name__
    msg = str(exc) or ""
    msg = " ".join(msg.split())[:160]
    return f"{name}: {msg}" if msg else name


def _tool_meta(tools: List[Any]) -> List[Dict[str, Optional[str]]]:
    """Small ``[{name, description}]`` view of discovered tools for the API."""
    out: List[Dict[str, Optional[str]]] = []
    for t in tools:
        name = getattr(t, "name", "") or ""
        if not name:
            continue
        desc = (getattr(t, "description", "") or "").strip() or None
        out.append({"name": name, "description": desc})
    return out


def _cap_tool_result(result: Any, limit: int) -> Any:
    """Cap a tool result at ``limit`` chars (with a truncation marker).

    Strings pass through untouched when small enough; anything else is
    stringified (JSON with ``default=str`` so content blocks / dicts stay
    readable). Never raises.
    """
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            text = str(result)
    if len(text) <= limit:
        return result
    return (
        text[:limit]
        + f"\n…[MCP tool result truncated: {len(text)} chars total, kept first {limit}]"
    )


def _wrap_bounded_tool(tool: Any) -> Any:
    """Bound one discovered MCP tool: call timeout + result size cap.

    Returns a copy of the tool whose coroutine is wrapped in
    ``asyncio.wait_for`` with its result capped — a slow or chatty external
    server can neither hang the agent nor flood its LLM context. A timed-out
    call yields a short error STRING (agents turn exceptions into failures;
    a string keeps the conversation alive). Non-BaseTool objects (test
    fakes) pass through unchanged; any wrapping problem degrades to the
    original tool rather than raising.
    """
    try:
        from langchain_core.tools import BaseTool
    except Exception:
        return tool
    if not isinstance(tool, BaseTool):
        return tool
    name = getattr(tool, "name", "") or "mcp-tool"
    timeout = tool_call_timeout()
    limit = tool_result_max_chars()

    async def _bounded(**kwargs: Any) -> Any:
        try:
            result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "mcp manager: MCP tool %r timed out after %.1fs; aborted",
                name,
                timeout,
            )
            return f"MCP tool {name!r} timed out after {timeout:.0f}s and was aborted."
        return _cap_tool_result(result, limit)

    try:
        return tool.model_copy(update={"coroutine": _bounded})
    except Exception:
        return tool


class McpToolManager:
    """Process-wide outbound MCP client/tool cache (see module docstring)."""

    def __init__(self) -> None:
        # server_id -> (fingerprint, MultiServerMCPClient)
        self._clients: Dict[str, Tuple[str, Any]] = {}
        # server_id -> (fingerprint, list[BaseTool])
        self._tools: Dict[str, Tuple[str, List[Any]]] = {}
        # server_id -> [{name, description}] (meta view of the last discovery)
        self._tools_meta: Dict[str, List[Dict[str, Optional[str]]]] = {}
        # server_id -> (fingerprint, failed_at_monotonic) — negative cache so
        # known-dead servers aren't re-dialed on every agent build.
        self._tools_failed: Dict[str, Tuple[str, float]] = {}

    # -- cache ---------------------------------------------------------------
    def invalidate(self, server_id: Optional[str] = None) -> None:
        """Drop cached clients/tools for one server (or everything)."""
        if server_id is None:
            self._clients.clear()
            self._tools.clear()
            self._tools_meta.clear()
            self._tools_failed.clear()
            return
        self._clients.pop(server_id, None)
        self._tools.pop(server_id, None)
        self._tools_meta.pop(server_id, None)
        self._tools_failed.pop(server_id, None)

    def cached_tools_meta(self, server_id: str) -> List[Dict[str, Optional[str]]]:
        """The last discovery result (meta view) — never connects."""
        return list(self._tools_meta.get(server_id) or [])

    # -- clients -------------------------------------------------------------
    def get_client(self, server: Any) -> Any:
        """A cached MultiServerMCPClient for the server row (1 connection).

        The cache key includes the config fingerprint, so an updated server
        transparently drops the stale client.
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient

        fp = server_fingerprint(server)
        cached = self._clients.get(server.id)
        if cached is not None and cached[0] == fp:
            return cached[1]
        client = MultiServerMCPClient({server.name: build_connection(server)})
        self._clients[server.id] = (fp, client)
        return client

    # -- discovery -------------------------------------------------------------
    async def discover_tools(self, server: Any, *, use_cache: bool = True) -> List[Any]:
        """``tools/list`` for one server (cached by config fingerprint).

        Raises on connect/protocol failure after ``MCP_DISCOVERY_TIMEOUT_SECONDS``
        — callers decide between health-check persistence and best-effort skip.
        A FAILED discovery is negatively cached for ``MCP_NEGATIVE_CACHE_SECONDS``
        (same fingerprint key): a repeat call within the TTL raises fast
        without touching the network. ``use_cache=False`` bypasses both caches
        (used by the truthful admin health-check).
        """
        fp = server_fingerprint(server)
        if use_cache:
            cached = self._tools.get(server.id)
            if cached is not None and cached[0] == fp:
                return cached[1]
            failed = self._tools_failed.get(server.id)
            if failed is not None and failed[0] == fp:
                ttl = negative_cache_ttl()
                age = time.monotonic() - failed[1]
                if age < ttl:
                    raise RuntimeError(
                        f"cached unreachable ({age:.0f}s ago, "
                        f"negative-cache TTL {ttl:.0f}s)"
                    )
        client = self.get_client(server)
        try:
            tools = await asyncio.wait_for(
                client.get_tools(), timeout=discovery_timeout()
            )
        except Exception:
            self._tools_failed[server.id] = (fp, time.monotonic())
            raise
        self._tools_failed.pop(server.id, None)
        self._tools[server.id] = (fp, list(tools))
        self._tools_meta[server.id] = _tool_meta(tools)
        return list(tools)

    # -- health check ----------------------------------------------------------
    async def health_check(
        self, server: Any, *, use_cache: bool = False
    ) -> Tuple[bool, Optional[str], List[Dict[str, Optional[str]]]]:
        """One bounded connect+discovery attempt.

        Returns ``(ok, short_detail_or_None, tool_meta)``. Never raises; the
        full traceback of a failure is logged with ``exc_info``.
        """
        try:
            tools = await self.discover_tools(server, use_cache=use_cache)
            return True, None, _tool_meta(tools)
        except asyncio.TimeoutError:
            logger.warning(
                "MCP health-check for server %s timed out after %.1fs",
                server.id,
                discovery_timeout(),
            )
            return False, "timeout", []
        except Exception as e:
            logger.warning(
                "MCP health-check for server %s failed: %s",
                server.id,
                e,
                exc_info=True,
            )
            return False, _short_error(e), []

    # -- product tools ----------------------------------------------------------
    def _load_bindings(self, product_id: str, session_factory: Optional[Any]) -> List[Any]:
        """Enabled bindings of enabled servers for the product.

        Returns ``[(binding, server), ...]``. Raises on DB failure (the caller
        wraps best-effort).
        """
        from api.models import McpServerORM, ProductMcpServerORM

        if session_factory is None:
            from api.db import SessionLocal as session_factory
        session = session_factory()
        try:
            rows = (
                session.query(ProductMcpServerORM, McpServerORM)
                .join(McpServerORM, ProductMcpServerORM.mcp_server_id == McpServerORM.id)
                .filter(
                    ProductMcpServerORM.product_id == product_id,
                    ProductMcpServerORM.enabled.is_(True),
                    McpServerORM.enabled.is_(True),
                )
                .order_by(ProductMcpServerORM.created_at, ProductMcpServerORM.id)
                .all()
            )
            return list(rows)
        finally:
            session.close()

    async def get_tools_for_product(
        self,
        product_id: str,
        session_factory: Optional[Any] = None,
    ) -> List[Any]:
        """LangChain tools of every ENABLED binding to an ENABLED server.

        Discovery for all bound servers runs in PARALLEL (a dead server
        costs its own timeout, not the sum of all timeouts) and failed
        discoveries are negatively cached (short TTL), so repeated agent
        builds don't re-dial known-dead servers. Per binding: name allowlist
        filter — ``allowed_tools: None`` means ALL tools, an explicit list
        (including the EMPTY list) means only the named tools (i.e. ``[]``
        exposes NO tools). A server that fails to connect is logged and
        skipped (best-effort). Duplicate tool names across servers are
        deduplicated (first binding in (created_at, id) order wins).
        """
        try:
            # Sync SQLAlchemy off the event loop (it blocks all requests on
            # this worker while it runs).
            bindings = await asyncio.to_thread(
                self._load_bindings, product_id, session_factory
            )
        except Exception as e:
            logger.warning(
                "mcp manager: could not load bindings for product %s: %s",
                product_id,
                e,
            )
            return []
        results = await asyncio.gather(
            *(self.discover_tools(server) for _binding, server in bindings),
            return_exceptions=True,
        )
        tools: List[Any] = []
        seen_names: set = set()
        # gather preserves input order, so zip(bindings, results) keeps the
        # deterministic (created_at, id) binding order for dedup decisions.
        for (binding, server), server_tools in zip(bindings, results):
            if isinstance(server_tools, BaseException):
                logger.warning(
                    "mcp manager: server %s (%s) unreachable for product %s "
                    "(%s); skipping its tools",
                    server.id,
                    server.name,
                    product_id,
                    _short_error(server_tools),
                )
                continue
            allow = binding.allowed_tools if isinstance(binding.allowed_tools, list) else None
            allow_set = {str(t) for t in allow} if allow is not None else None
            for tool in server_tools:
                name = getattr(tool, "name", "") or ""
                if allow_set is not None and name not in allow_set:
                    continue
                if name and name in seen_names:
                    logger.warning(
                        "mcp manager: duplicate MCP tool name %r (server %s) "
                        "dropped for product %s",
                        name,
                        server.name,
                        product_id,
                    )
                    continue
                if name:
                    seen_names.add(name)
                tools.append(_wrap_bounded_tool(tool))
        if tools:
            logger.info(
                "mcp manager: %d MCP tool(s) available for product %s",
                len(tools),
                product_id,
            )
        return tools


#: Process-wide manager instance (import-safe, no I/O at import).
_manager: Optional[McpToolManager] = None


def get_mcp_manager() -> McpToolManager:
    """The process-wide :class:`McpToolManager` singleton."""
    global _manager
    if _manager is None:
        _manager = McpToolManager()
    return _manager


def reset_mcp_manager() -> McpToolManager:
    """Drop the singleton (and its caches) — test helper."""
    global _manager
    _manager = McpToolManager()
    return _manager


async def gather_mcp_agent_tools(
    product_id: str,
    session_factory: Optional[Any] = None,
) -> List[Any]:
    """Best-effort MCP tools for the expert agent (never raises, never hangs).

    Wraps :meth:`McpToolManager.get_tools_for_product`; any failure (DB down,
    dependency missing) logs and returns ``[]`` so an MCP outage can never
    break the agent.
    """
    try:
        return await get_mcp_manager().get_tools_for_product(
            product_id, session_factory=session_factory
        )
    except Exception as e:
        logger.warning(
            "gather_mcp_agent_tools failed for product %s: %s", product_id, e
        )
        return []


__all__ = [
    "DEFAULT_DISCOVERY_TIMEOUT",
    "DEFAULT_NEGATIVE_CACHE_TTL",
    "DEFAULT_TOOL_CALL_TIMEOUT",
    "DEFAULT_TOOL_RESULT_MAX_CHARS",
    "McpToolManager",
    "build_connection",
    "discovery_timeout",
    "gather_mcp_agent_tools",
    "get_mcp_manager",
    "negative_cache_ttl",
    "reset_mcp_manager",
    "server_fingerprint",
    "tool_call_timeout",
    "tool_result_max_chars",
]
