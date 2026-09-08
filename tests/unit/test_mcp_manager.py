#!/usr/bin/env python3
"""Unit tests for ``api.mcp.manager`` (Wave C outbound MCP tool manager).

Everything runs against fake clients (no subprocesses, no network):

- ``discovery_timeout`` resolution (default / env / invalid / floor).
- ``server_fingerprint`` stability + sensitivity.
- ``build_connection`` shapes for http/stdio (headers/env decrypted) and the
  unsupported-transport error.
- client cache + discovery cache keyed by config fingerprint, ``invalidate``.
- ``health_check`` success / failure / timeout with sanitized short errors.
- ``get_tools_for_product``: allowlist, disabled binding/server filters, dead
  server skipped, duplicate tool names deduped, DB failure -> ``[]``.
- ``gather_mcp_agent_tools`` never raises.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.mcp import manager as manager_mod  # noqa: E402


class _FakeTool:
    def __init__(self, name: str, description: str = "fake tool"):
        self.name = name
        self.description = description


class _FakeClient:
    """``MultiServerMCPClient`` stand-in: ``get_tools`` returns canned tools."""

    def __init__(self, tools=None, error=None):
        self.tools = list(tools or [])
        self.error = error
        self.calls = 0

    async def get_tools(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.tools)


def _server(**kw):
    base = dict(
        id="mcp_1",
        name="srv",
        transport="http",
        url="http://localhost:9000/mcp",
        command=None,
        args=None,
        headers=None,
        env=None,
        updated_at="2026-01-01T00:00:00",
    )
    base.update(kw)
    return SimpleNamespace(**base)


# --- discovery timeout ---------------------------------------------------------
class TestDiscoveryTimeout:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MCP_DISCOVERY_TIMEOUT_SECONDS", raising=False)
        assert manager_mod.discovery_timeout() == 10.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MCP_DISCOVERY_TIMEOUT_SECONDS", "5.5")
        assert manager_mod.discovery_timeout() == 5.5

    def test_invalid_value_falls_back(self, monkeypatch):
        monkeypatch.setenv("MCP_DISCOVERY_TIMEOUT_SECONDS", "nope")
        assert manager_mod.discovery_timeout() == 10.0

    def test_floor_one_second(self, monkeypatch):
        monkeypatch.setenv("MCP_DISCOVERY_TIMEOUT_SECONDS", "0.2")
        assert manager_mod.discovery_timeout() == 1.0


# --- fingerprint + connection building -----------------------------------------
class TestFingerprintAndConnection:
    def test_fingerprint_stable_and_sensitive(self):
        a = _server()
        assert manager_mod.server_fingerprint(a) == manager_mod.server_fingerprint(
            _server()
        )
        assert manager_mod.server_fingerprint(a) != manager_mod.server_fingerprint(
            _server(url="http://other:1/mcp")
        )
        assert manager_mod.server_fingerprint(a) != manager_mod.server_fingerprint(
            _server(updated_at="2026-06-01T00:00:00")
        )

    def test_http_connection_shape(self, monkeypatch):
        monkeypatch.delenv("MCP_DISCOVERY_TIMEOUT_SECONDS", raising=False)
        conn = manager_mod.build_connection(_server())
        assert conn["transport"] == "http"
        assert conn["url"] == "http://localhost:9000/mcp"
        assert conn["timeout"] == 10.0
        assert "headers" not in conn  # no secrets -> no headers key

    def test_http_connection_decrypts_headers(self):
        from api.mcp.secrets import encrypt_secret_dict

        enc = encrypt_secret_dict({"Authorization": "Bearer s3cret"})
        conn = manager_mod.build_connection(_server(headers=enc))
        assert conn["headers"] == {"Authorization": "Bearer s3cret"}

    def test_stdio_connection_shape(self):
        from api.mcp.secrets import encrypt_secret_dict

        enc = encrypt_secret_dict({"FOO": "bar"})
        conn = manager_mod.build_connection(
            _server(
                transport="stdio",
                url=None,
                command="/usr/local/bin/mcp-server",
                args=["--verbose"],
                env=enc,
            )
        )
        assert conn == {
            "transport": "stdio",
            "command": "/usr/local/bin/mcp-server",
            "args": ["--verbose"],
            "env": {"FOO": "bar"},
        }

    def test_corrupt_ciphertext_yields_no_headers(self):
        conn = manager_mod.build_connection(_server(headers="not-a-ciphertext"))
        assert "headers" not in conn

    def test_unsupported_transport_raises(self):
        with pytest.raises(ValueError):
            manager_mod.build_connection(_server(transport="bogus", url=None))


# --- client cache + discovery cache ---------------------------------------------
class TestClientAndDiscoveryCache:
    def test_discover_tools_cached_per_fingerprint(self):
        m = manager_mod.McpToolManager()
        client = _FakeClient(tools=[_FakeTool("t1"), _FakeTool("t2", "")])
        m.get_client = lambda server: client

        first = asyncio.run(m.discover_tools(_server()))
        second = asyncio.run(m.discover_tools(_server()))

        assert client.calls == 1  # second call served from cache
        assert [t.name for t in first] == ["t1", "t2"]
        assert [t.name for t in second] == ["t1", "t2"]
        assert m.cached_tools_meta("mcp_1") == [
            {"name": "t1", "description": "fake tool"},
            {"name": "t2", "description": None},
        ]

    def test_discover_tools_cache_bypass(self):
        m = manager_mod.McpToolManager()
        client = _FakeClient(tools=[_FakeTool("t1")])
        m.get_client = lambda server: client

        asyncio.run(m.discover_tools(_server()))
        asyncio.run(m.discover_tools(_server(), use_cache=False))
        assert client.calls == 2

    def test_config_change_drops_cache(self):
        m = manager_mod.McpToolManager()
        client = _FakeClient(tools=[_FakeTool("t1")])
        m.get_client = lambda server: client

        asyncio.run(m.discover_tools(_server()))
        # Same id, new URL -> different fingerprint -> cache miss.
        asyncio.run(m.discover_tools(_server(url="http://other:1/mcp")))
        assert client.calls == 2

    def test_invalidate_drops_cache(self):
        m = manager_mod.McpToolManager()
        client = _FakeClient(tools=[_FakeTool("t1")])
        m.get_client = lambda server: client

        asyncio.run(m.discover_tools(_server()))
        m.invalidate("mcp_1")
        assert m.cached_tools_meta("mcp_1") == []
        asyncio.run(m.discover_tools(_server()))
        assert client.calls == 2

    def test_invalidate_all_clears_everything(self):
        m = manager_mod.McpToolManager()
        client = _FakeClient(tools=[_FakeTool("t1")])
        m.get_client = lambda server: client
        asyncio.run(m.discover_tools(_server()))
        m.invalidate()
        assert m.cached_tools_meta("mcp_1") == []
        assert m._clients == {}

    def test_get_client_reuses_until_config_changes(self, monkeypatch):
        import langchain_mcp_adapters.client as lmc

        made = []

        def _fake_client_cls(connections):
            made.append(connections)
            return object()

        monkeypatch.setattr(lmc, "MultiServerMCPClient", _fake_client_cls)
        m = manager_mod.McpToolManager()

        c1 = m.get_client(_server())
        c2 = m.get_client(_server())
        assert c1 is c2
        assert len(made) == 1

        c3 = m.get_client(_server(url="http://other:1/mcp"))
        assert c3 is not c1
        assert len(made) == 2
        assert made[1]["srv"]["url"] == "http://other:1/mcp"


# --- negative discovery cache ------------------------------------------------------
class TestNegativeDiscoveryCache:
    def test_failure_short_circuits_within_ttl(self, monkeypatch):
        monkeypatch.delenv("MCP_NEGATIVE_CACHE_SECONDS", raising=False)
        m = manager_mod.McpToolManager()
        client = _FakeClient(error=RuntimeError("boom"))
        m.get_client = lambda server: client

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(m.discover_tools(_server()))
        assert client.calls == 1

        # Second call within the TTL raises FAST without dialing again.
        with pytest.raises(RuntimeError, match="cached unreachable"):
            asyncio.run(m.discover_tools(_server()))
        assert client.calls == 1

        # use_cache=False (the truthful admin health-check) still dials.
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(m.discover_tools(_server(), use_cache=False))
        assert client.calls == 2

        # invalidate() clears the negative entry -> a real retry happens.
        m.invalidate("mcp_1")
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(m.discover_tools(_server()))
        assert client.calls == 3

    def test_ttl_zero_disables_negative_cache(self, monkeypatch):
        monkeypatch.setenv("MCP_NEGATIVE_CACHE_SECONDS", "0")
        m = manager_mod.McpToolManager()
        client = _FakeClient(error=RuntimeError("boom"))
        m.get_client = lambda server: client

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(m.discover_tools(_server()))
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(m.discover_tools(_server()))
        assert client.calls == 2  # every call dials

    def test_recovery_clears_negative_entry(self, monkeypatch):
        monkeypatch.delenv("MCP_NEGATIVE_CACHE_SECONDS", raising=False)
        m = manager_mod.McpToolManager()
        client = _FakeClient(error=RuntimeError("boom"))
        m.get_client = lambda server: client

        with pytest.raises(RuntimeError):
            asyncio.run(m.discover_tools(_server()))
        client.error = None
        client.tools = [_FakeTool("t1")]
        # A forced (cache-bypassing) success clears the negative entry …
        assert [t.name for t in asyncio.run(m.discover_tools(_server(), use_cache=False))] == [
            "t1"
        ]
        assert m._tools_failed == {}
        # … so subsequent cached calls succeed instead of raising fast.
        assert [t.name for t in asyncio.run(m.discover_tools(_server()))] == ["t1"]

    def test_config_change_ignores_stale_negative_entry(self):
        m = manager_mod.McpToolManager()
        client = _FakeClient(error=RuntimeError("boom"))
        m.get_client = lambda server: client

        with pytest.raises(RuntimeError):
            asyncio.run(m.discover_tools(_server()))
        # Same id, NEW url -> different fingerprint -> negative entry ignored.
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(m.discover_tools(_server(url="http://other:1/mcp")))
        assert client.calls == 2


# --- stdio policy (shared with the router) ------------------------------------------
class TestStdioPolicy:
    def test_command_policy(self):
        from api.mcp.policy import stdio_command_error

        for bad in ("/bin/sh", "bash", "/usr/bin/zsh", "/usr/bin/env", "npx",
                    "uvx", "/usr/local/bin/python3.12", "/usr/bin/node",
                    "/usr/bin/osascript", "sh -c run.sh", ""):
            assert stdio_command_error(bad) is not None, bad
        for good in ("/usr/local/bin/mcp-server", "/bin/srv", "mcp-server-foo"):
            assert stdio_command_error(good) is None, good

    def test_env_policy(self):
        from api.mcp.policy import stdio_env_error

        for bad in ({"PATH": "/tmp"}, {"LD_PRELOAD": "/tmp/x.so"},
                    {"LD_LIBRARY_PATH": "/tmp"}, {"DYLD_INSERT_LIBRARIES": "x"},
                    {"PYTHONPATH": "x"}, {"NODE_OPTIONS": "--require /tmp/x"},
                    {"BASH_ENV": "/tmp/x"}):
            assert stdio_env_error(bad) is not None, bad
        for good in ({}, None, {"API_TOKEN": "t"}, {"HOME": "/tmp"}):
            assert stdio_env_error(good) is None, good

    def test_build_connection_refuses_interpreter_row(self):
        with pytest.raises(ValueError, match="shell/interpreter/runner"):
            manager_mod.build_connection(
                _server(transport="stdio", url=None, command="/bin/sh", args=["-c", "id"])
            )

    def test_build_connection_refuses_loader_env_row(self):
        from api.mcp.secrets import encrypt_secret_dict

        with pytest.raises(ValueError, match="not allowed"):
            manager_mod.build_connection(
                _server(
                    transport="stdio",
                    url=None,
                    command="/usr/local/bin/mcp-server",
                    env=encrypt_secret_dict({"LD_PRELOAD": "/tmp/x.so"}),
                )
            )


# --- bounded tool wrapper (timeout + result cap) --------------------------------------
class TestBoundedToolWrapper:
    def _structured(self, name, coro):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        class _NoArgs(BaseModel):
            pass

        # langchain-core 1.6 requires an explicit args_schema (no inference
        # from func) — an empty schema models a no-argument MCP tool.
        return StructuredTool(
            name=name,
            description="bounded",
            args_schema=_NoArgs,
            func=lambda **kwargs: "sync-fallback",
            coroutine=coro,
        )

    def test_fake_tools_pass_through_unwrapped(self):
        fake = _FakeTool("f")
        assert manager_mod._wrap_bounded_tool(fake) is fake

    def test_timeout_returns_string_not_raises(self, monkeypatch):
        monkeypatch.setenv("MCP_TOOL_CALL_TIMEOUT_SECONDS", "0.2")

        async def _slow(**kwargs):
            await asyncio.sleep(5)
            return "never"

        tool = self._structured("slow", _slow)
        wrapped = manager_mod._wrap_bounded_tool(tool)
        start = time.monotonic()
        out = asyncio.run(wrapped.ainvoke({}))
        elapsed = time.monotonic() - start
        assert "timed out" in out
        assert elapsed < 2.0  # bounded well below the 5s body

    def test_huge_result_capped_with_marker(self, monkeypatch):
        monkeypatch.setenv("MCP_TOOL_RESULT_MAX_CHARS", "1000")

        async def _huge(**kwargs):
            return "x" * 300_000

        wrapped = manager_mod._wrap_bounded_tool(self._structured("huge", _huge))
        out = asyncio.run(wrapped.ainvoke({}))
        assert len(out) < 1200  # 1000-char cap + marker
        assert "truncated" in out
        assert out.endswith("]")

    def test_small_result_untouched(self, monkeypatch):
        monkeypatch.delenv("MCP_TOOL_RESULT_MAX_CHARS", raising=False)

        async def _small(**kwargs):
            return {"items": [1, 2, 3]}

        wrapped = manager_mod._wrap_bounded_tool(self._structured("small", _small))
        out = asyncio.run(wrapped.ainvoke({}))
        assert out == {"items": [1, 2, 3]}


# --- health check ----------------------------------------------------------------
class TestHealthCheck:
    def test_ok(self):
        m = manager_mod.McpToolManager()
        m.get_client = lambda server: _FakeClient(tools=[_FakeTool("t1")])
        ok, detail, meta = asyncio.run(m.health_check(_server(), use_cache=False))
        assert ok is True
        assert detail is None
        assert meta == [{"name": "t1", "description": "fake tool"}]

    def test_failure_short_sanitized_detail(self):
        m = manager_mod.McpToolManager()
        long_url = "http://user:pass@secret-host:1234/very/long/path" * 10
        m.get_client = lambda server: _FakeClient(error=RuntimeError(f"boom {long_url}"))
        ok, detail, meta = asyncio.run(m.health_check(_server(), use_cache=False))
        assert ok is False
        assert meta == []
        assert detail.startswith("RuntimeError:")
        assert len(detail) <= 200  # heavily truncated, never the full traceback

    def test_timeout(self, monkeypatch):
        monkeypatch.setenv("MCP_DISCOVERY_TIMEOUT_SECONDS", "1")
        m = manager_mod.McpToolManager()

        class _SlowClient:
            async def get_tools(self):
                await asyncio.sleep(30)
                return []

        m.get_client = lambda server: _SlowClient()
        ok, detail, meta = asyncio.run(m.health_check(_server(), use_cache=False))
        assert ok is False
        assert detail == "timeout"
        assert meta == []


# --- get_tools_for_product --------------------------------------------------------
def _seed_registry(db_mod):
    from api.models import McpServerORM, ProductMcpServerORM, ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id="prod_1", name="P1"))
        db.add(
            McpServerORM(
                id="mcp_a", name="alpha", transport="http",
                url="http://a/mcp", enabled=True, status="ok",
            )
        )
        db.add(
            McpServerORM(
                id="mcp_b", name="beta", transport="http",
                url="http://b/mcp", enabled=True, status="ok",
            )
        )
        # Registered but DISABLED server with an enabled binding -> filtered.
        db.add(
            McpServerORM(
                id="mcp_off", name="off", transport="http",
                url="http://off/mcp", enabled=False, status="unknown",
            )
        )
        db.add(
            ProductMcpServerORM(
                id="pmb_a", product_id="prod_1", mcp_server_id="mcp_a",
                enabled=True, allowed_tools=None,
            )
        )
        db.add(
            ProductMcpServerORM(
                id="pmb_b", product_id="prod_1", mcp_server_id="mcp_b",
                enabled=True, allowed_tools=["beta_only"],
            )
        )
        db.add(
            ProductMcpServerORM(
                id="pmb_off", product_id="prod_1", mcp_server_id="mcp_off",
                enabled=True,
            )
        )
        # Enabled server with a DISABLED binding -> filtered.
        db.add(
            McpServerORM(
                id="mcp_c", name="gamma", transport="http",
                url="http://c/mcp", enabled=True, status="ok",
            )
        )
        db.add(
            ProductMcpServerORM(
                id="pmb_c", product_id="prod_1", mcp_server_id="mcp_c",
                enabled=False,
            )
        )
        db.commit()


class TestGetToolsForProduct:
    def test_allowlist_and_enabled_filters(self, isolated_db):
        _seed_registry(isolated_db)
        m = manager_mod.McpToolManager()
        contacted = []

        async def _fake_discover(server, use_cache=True):
            contacted.append(server.id)
            if server.id == "mcp_a":
                return [_FakeTool("alpha_one"), _FakeTool("alpha_two")]
            if server.id == "mcp_b":
                return [_FakeTool("beta_only"), _FakeTool("beta_extra")]
            return []

        m.discover_tools = _fake_discover
        tools = asyncio.run(
            m.get_tools_for_product("prod_1", session_factory=isolated_db.SessionLocal)
        )
        # beta_extra dropped by the allowlist; disabled server/binding never
        # contacted at all.
        assert [t.name for t in tools] == ["alpha_one", "alpha_two", "beta_only"]
        assert set(contacted) == {"mcp_a", "mcp_b"}

    def test_dead_server_skipped_best_effort(self, isolated_db):
        _seed_registry(isolated_db)
        m = manager_mod.McpToolManager()

        async def _fake_discover(server, use_cache=True):
            if server.id == "mcp_a":
                raise RuntimeError("unreachable")
            return [_FakeTool("beta_only")]

        m.discover_tools = _fake_discover
        tools = asyncio.run(
            m.get_tools_for_product("prod_1", session_factory=isolated_db.SessionLocal)
        )
        assert [t.name for t in tools] == ["beta_only"]

    def test_duplicate_tool_names_deduped(self, isolated_db):
        from api.models import McpServerORM, ProductMcpServerORM, ProductORM

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_2", name="P2"))
            for sid in ("mcp_x", "mcp_y"):
                db.add(
                    McpServerORM(
                        id=sid, name=sid, transport="http",
                        url=f"http://{sid}/mcp", enabled=True, status="ok",
                    )
                )
                db.add(
                    ProductMcpServerORM(
                        id=f"pmb_{sid}", product_id="prod_2", mcp_server_id=sid,
                        enabled=True,
                    )
                )
            db.commit()

        m = manager_mod.McpToolManager()

        async def _fake_discover(server, use_cache=True):
            # Both servers expose a tool named "shared" + one unique tool.
            return [_FakeTool("shared"), _FakeTool(f"uniq_{server.id}")]

        m.discover_tools = _fake_discover
        tools = asyncio.run(
            m.get_tools_for_product("prod_2", session_factory=isolated_db.SessionLocal)
        )
        names = [t.name for t in tools]
        assert names.count("shared") == 1  # first server wins
        assert set(names) == {"shared", "uniq_mcp_x", "uniq_mcp_y"}

    def test_empty_allowlist_exposes_no_tools(self, isolated_db):
        from api.models import ProductMcpServerORM

        _seed_registry(isolated_db)
        with isolated_db.SessionLocal() as db:
            db.get(ProductMcpServerORM, "pmb_b").allowed_tools = []
            db.commit()

        m = manager_mod.McpToolManager()

        async def _fake_discover(server, use_cache=True):
            if server.id == "mcp_a":
                return [_FakeTool("alpha_one"), _FakeTool("alpha_two")]
            return [_FakeTool("beta_only"), _FakeTool("beta_extra")]

        m.discover_tools = _fake_discover
        tools = asyncio.run(
            m.get_tools_for_product("prod_1", session_factory=isolated_db.SessionLocal)
        )
        # beta's explicit EMPTY allowlist means NO beta tools (not "all").
        assert [t.name for t in tools] == ["alpha_one", "alpha_two"]

    def test_discovery_runs_in_parallel(self, isolated_db):
        from api.models import ProductMcpServerORM

        _seed_registry(isolated_db)
        # beta's allowlist would filter its tool out — open it for this test.
        with isolated_db.SessionLocal() as db:
            db.get(ProductMcpServerORM, "pmb_b").allowed_tools = None
            db.commit()

        m = manager_mod.McpToolManager()

        async def _fake_discover(server, use_cache=True):
            await asyncio.sleep(0.25)
            return [_FakeTool(f"t_{server.id}")]

        m.discover_tools = _fake_discover
        start = time.monotonic()
        tools = asyncio.run(
            m.get_tools_for_product("prod_1", session_factory=isolated_db.SessionLocal)
        )
        elapsed = time.monotonic() - start
        assert [t.name for t in tools] == ["t_mcp_a", "t_mcp_b"]
        assert elapsed < 0.45  # serial discovery would take >= 0.5s

    def test_bindings_ordered_by_created_at_then_id(self, isolated_db):
        from datetime import datetime

        from api.models import ProductMcpServerORM

        _seed_registry(isolated_db)
        same_ts = datetime(2026, 1, 1, 0, 0, 0)
        with isolated_db.SessionLocal() as db:
            db.get(ProductMcpServerORM, "pmb_a").created_at = same_ts
            db.get(ProductMcpServerORM, "pmb_b").created_at = same_ts
            db.commit()

        m = manager_mod.McpToolManager()
        rows = m._load_bindings("prod_1", isolated_db.SessionLocal)
        # equal created_at -> id tiebreak gives a deterministic order
        assert [b.id for b, _server in rows] == ["pmb_a", "pmb_b"]

    def test_db_error_returns_empty(self, isolated_db, monkeypatch):
        m = manager_mod.McpToolManager()

        def _boom(product_id, session_factory=None):
            raise RuntimeError("db down")

        monkeypatch.setattr(m, "_load_bindings", _boom)
        assert asyncio.run(m.get_tools_for_product("prod_x")) == []

    def test_no_bindings_returns_empty(self, isolated_db):
        from api.models import ProductORM

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_9", name="P9"))
            db.commit()

        m = manager_mod.McpToolManager()
        assert (
            asyncio.run(
                m.get_tools_for_product("prod_9", session_factory=isolated_db.SessionLocal)
            )
            == []
        )


class TestGatherNeverRaises:
    def test_manager_construction_failure_yields_empty(self, monkeypatch):
        def _boom():
            raise RuntimeError("no manager for you")

        monkeypatch.setattr(manager_mod, "get_mcp_manager", _boom)
        assert asyncio.run(manager_mod.gather_mcp_agent_tools("prod_1")) == []

    def test_delegates_to_manager(self, monkeypatch):
        m = manager_mod.McpToolManager()

        async def _fake(pid, session_factory=None):
            return [_FakeTool("ok")]

        monkeypatch.setattr(m, "get_tools_for_product", _fake)
        monkeypatch.setattr(manager_mod, "get_mcp_manager", lambda: m)
        out = asyncio.run(manager_mod.gather_mcp_agent_tools("prod_1"))
        assert [t.name for t in out] == ["ok"]
