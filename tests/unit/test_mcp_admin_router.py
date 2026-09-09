#!/usr/bin/env python3
"""Unit tests for ``api.routers.mcp_admin`` (Wave C contract 1).

Admin CRUD + health-check + cached tool list over an isolated SQLite DB:

- create (http + stdio): secrets encrypted at rest, masked in responses.
- validation 400s: bad URL scheme, shell-metachar/``..``/whitespace commands,
  bogus transport, cross-field mismatches, oversize args/secrets, empty name.
- duplicate name -> 409; transport immutable on PUT -> 400.
- PUT secret-dict presence semantics (absent keeps, present replaces).
- DELETE -> 200, bindings cascade, second DELETE -> 404.
- POST /test persists status via a faked manager; GET /tools serves the cache
  without ever connecting.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _admin_user_orm():
    from api.models import UserORM

    return UserORM(
        id="user_admin1",
        username="admin",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


class _FakeManager:
    """Stands in for the outbound manager: no network, records calls."""

    def __init__(self):
        self.invalidated = []
        self.tools_meta = {}
        self.health_calls = 0
        self.health = (True, None, [{"name": "echo", "description": "Echo tool"}])

    def invalidate(self, server_id=None):
        self.invalidated.append(server_id)

    def cached_tools_meta(self, server_id):
        return list(self.tools_meta.get(server_id, []))

    async def health_check(self, server, use_cache=False):
        self.health_calls += 1
        return self.health


def _build_client(db_mod, mcp_admin_mod, monkeypatch, *, fake=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.auth import deps as auth_deps

    fake = fake if fake is not None else _FakeManager()
    monkeypatch.setattr(mcp_admin_mod, "get_mcp_manager", lambda: fake)

    app = FastAPI()
    app.include_router(mcp_admin_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[mcp_admin_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.require_admin] = _admin_user_orm
    return app, TestClient(app), fake


def _mod():
    from api.routers import mcp_admin

    return mcp_admin


def _stored_row(db_mod, server_id):
    from api.models import McpServerORM

    with db_mod.SessionLocal() as db:
        return db.query(McpServerORM).filter(McpServerORM.id == server_id).one()


_HTTP_BODY = {
    "name": "alpha",
    "transport": "http",
    "url": "http://localhost:9000/mcp",
    "headers": {"Authorization": "Bearer s3cret-value"},
}


# --- create ------------------------------------------------------------------
class TestCreateMcpServer:
    def test_create_http_server(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        resp = client.post("/api/admin/mcp/servers", json=_HTTP_BODY)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"].startswith("mcp_")
        assert len(body["id"]) == len("mcp_") + 32  # token_hex(16)
        assert body["transport"] == "http"
        assert body["url"] == "http://localhost:9000/mcp"
        assert body["enabled"] is True
        assert body["status"] == "unknown"
        assert body["status_checked_at"] is None

    def test_secrets_encrypted_at_rest_and_masked_in_response(
        self, isolated_db, monkeypatch
    ):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        body = client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()
        # Response: masked key-only view, never the value.
        assert body["headers_masked"] == {"Authorization": "***"}
        assert "s3cret-value" not in body

        row = _stored_row(isolated_db, body["id"])
        stored = row.headers
        assert "s3cret-value" not in str(stored)  # encrypted at rest
        from api.mcp.secrets import decrypt_secret_dict

        assert decrypt_secret_dict(stored) == {"Authorization": "Bearer s3cret-value"}

    def test_create_stdio_server(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        resp = client.post(
            "/api/admin/mcp/servers",
            json={
                "name": "beta",
                "transport": "stdio",
                "command": "/usr/local/bin/mcp-server",
                "args": ["--verbose", "--port", "9000"],
                "env": {"API_TOKEN": "tok"},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["command"] == "/usr/local/bin/mcp-server"
        assert body["args"] == ["--verbose", "--port", "9000"]
        assert body["url"] is None
        assert body["env_masked"] == {"API_TOKEN": "***"}

    def test_duplicate_name_409(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        assert client.post("/api/admin/mcp/servers", json=_HTTP_BODY).status_code == 200
        resp = client.post(
            "/api/admin/mcp/servers",
            json={**_HTTP_BODY, "url": "http://other:1/mcp"},
        )
        assert resp.status_code == 409

    def test_list_returns_created_servers(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        client.post("/api/admin/mcp/servers", json=_HTTP_BODY)
        client.post(
            "/api/admin/mcp/servers",
            json={"name": "beta", "transport": "stdio", "command": "/bin/srv"},
        )
        resp = client.get("/api/admin/mcp/servers")
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()}
        assert names == {"alpha", "beta"}


# --- validation ---------------------------------------------------------------
class TestCreateValidation:
    def _post(self, client, payload):
        return client.post("/api/admin/mcp/servers", json=payload)

    def test_bad_url_scheme(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(client, {"name": "x", "transport": "http", "url": "ftp://h/mcp"})
        assert r.status_code == 400

    def test_missing_url_for_http(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(client, {"name": "x", "transport": "http"})
        assert r.status_code == 400

    def test_bogus_transport(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client, {"name": "x", "transport": "websocket", "url": "http://h/mcp"}
        )
        assert r.status_code == 400

    def test_http_rejects_command_and_args(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        base = {"name": "x", "transport": "http", "url": "http://h/mcp"}
        assert self._post(client, {**base, "command": "/bin/srv"}).status_code == 400
        assert self._post(client, {**base, "args": ["a"]}).status_code == 400

    def test_stdio_rejects_url(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {"name": "x", "transport": "stdio", "command": "/bin/srv",
             "url": "http://h/mcp"},
        )
        assert r.status_code == 400

    @pytest.mark.parametrize(
        "command",
        [
            "sh -c run.sh",       # whitespace smuggles arguments
            "/bin/srv;rm -rf /",  # shell metacharacters
            "../escape/bin",      # path traversal
            "cmd`id`",            # backtick substitution
            "cmd$x",              # variable expansion
            "cmd>out",            # redirect
            "",                   # empty
        ],
    )
    def test_stdio_command_rejected(self, isolated_db, monkeypatch, command):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(client, {"name": "x", "transport": "stdio", "command": command})
        assert r.status_code == 400, command

    @pytest.mark.parametrize(
        "command",
        [
            "/bin/sh",               # shell: args become code via -c
            "/usr/bin/env",           # trampoline
            "/usr/bin/python3.12",     # interpreter (version suffix)
            "npx",                    # package runner
            "/usr/bin/osascript",     # macOS automation
        ],
    )
    def test_stdio_interpreter_commands_rejected(
        self, isolated_db, monkeypatch, command
    ):
        """Shells/interpreters/runners turn validated args into code exec."""
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {"name": "x", "transport": "stdio", "command": command,
             "args": ["-c", "id"]},
        )
        assert r.status_code == 400, command

    @pytest.mark.parametrize(
        "env",
        [
            {"LD_PRELOAD": "/tmp/x.so"},      # loader hijack
            {"LD_LIBRARY_PATH": "/tmp"},        # loader hijack
            {"PATH": "/tmp"},                   # binary substitution
            {"PYTHONPATH": "/tmp"},             # interpreter injection
            {"NODE_OPTIONS": "--require /x"},   # interpreter injection
        ],
    )
    def test_stdio_forbidden_env_rejected(self, isolated_db, monkeypatch, env):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {"name": "x", "transport": "stdio",
             "command": "/usr/local/bin/mcp-server", "env": env},
        )
        assert r.status_code == 400, env

    def test_stdio_benign_env_still_accepted(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {"name": "x", "transport": "stdio",
             "command": "/usr/local/bin/mcp-server",
             "env": {"API_TOKEN": "tok"}},
        )
        assert r.status_code == 200, r.text

    def test_args_entry_too_long(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {"name": "x", "transport": "stdio", "command": "/bin/srv",
             "args": ["a" * 600]},
        )
        assert r.status_code == 400

    def test_headers_value_too_long(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {**_HTTP_BODY, "headers": {"Authorization": "v" * 3000}},
        )
        assert r.status_code == 400

    def test_header_key_control_chars_rejected(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(client, {**_HTTP_BODY, "headers": {"Bad\nKey": "v"}})
        assert r.status_code == 400

    def test_whitespace_only_name_rejected(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(client, {**_HTTP_BODY, "name": "   "})
        assert r.status_code == 400


# --- update ---------------------------------------------------------------------
class TestUpdateMcpServer:
    def _create(self, client):
        return client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()

    def test_update_url_and_invalidate(self, isolated_db, monkeypatch):
        app, client, fake = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)
        resp = client.put(
            f"/api/admin/mcp/servers/{created['id']}",
            json={"url": "http://new-host:9001/mcp"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["url"] == "http://new-host:9001/mcp"
        assert created["id"] in fake.invalidated

    def test_transport_immutable(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)
        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}",
            json={"transport": "stdio", "command": "/bin/x"},
        )
        assert r.status_code == 400

    def test_absent_headers_keeps_stored_ciphertext(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)
        before = _stored_row(isolated_db, created["id"]).headers

        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}", json={"enabled": False}
        )
        assert r.status_code == 200
        assert r.json()["enabled"] is False

        after = _stored_row(isolated_db, created["id"]).headers
        assert after == before  # untouched when the field is absent

    def test_present_headers_replaces_wholesale(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)

        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}",
            json={"headers": {"X-Other": "v2"}},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["headers_masked"] == {"X-Other": "***"}

        from api.mcp.secrets import decrypt_secret_dict

        stored = _stored_row(isolated_db, created["id"]).headers
        assert decrypt_secret_dict(stored) == {"X-Other": "v2"}

    def test_rename_conflict_409(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        first = self._create(client)
        client.post(
            "/api/admin/mcp/servers",
            json={**_HTTP_BODY, "name": "second", "url": "http://s:1/mcp"},
        )
        r = client.put(
            f"/api/admin/mcp/servers/{first['id']}", json={"name": "second"}
        )
        assert r.status_code == 409

    def test_update_unknown_404(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        assert (
            client.put("/api/admin/mcp/servers/mcp_nope", json={"enabled": False}).status_code
            == 404
        )

    def test_all_masked_headers_echo_keeps_secret(self, isolated_db, monkeypatch):
        """A UI echoing the masked view back must not wipe the stored secret."""
        from api.mcp.secrets import decrypt_secret_dict

        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)
        before = decrypt_secret_dict(_stored_row(isolated_db, created["id"]).headers)

        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}",
            json={"headers": {"Authorization": "***"}},
        )
        assert r.status_code == 200, r.text
        after = decrypt_secret_dict(_stored_row(isolated_db, created["id"]).headers)
        assert after == before  # no-op, secret intact
        assert r.json()["headers_masked"] == {"Authorization": "***"}

    def test_mixed_masked_headers_replaces_with_real_entries(
        self, isolated_db, monkeypatch
    ):
        from api.mcp.secrets import decrypt_secret_dict

        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)

        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}",
            json={"headers": {"Authorization": "***", "X-New": "v2"}},
        )
        assert r.status_code == 200, r.text
        # Masked entries are dropped; the payload replaces wholesale with the
        # real entries only.
        stored = decrypt_secret_dict(_stored_row(isolated_db, created["id"]).headers)
        assert stored == {"X-New": "v2"}

    def test_empty_headers_clears(self, isolated_db, monkeypatch):
        from api.mcp.secrets import decrypt_secret_dict

        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)

        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}", json={"headers": {}}
        )
        assert r.status_code == 200
        assert decrypt_secret_dict(_stored_row(isolated_db, created["id"]).headers) == {}

    def test_url_change_resets_status(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)
        assert client.post(f"/api/admin/mcp/servers/{created['id']}/test").status_code == 200
        row = _stored_row(isolated_db, created["id"])
        assert row.status == "ok"

        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}",
            json={"url": "http://new-host:9002/mcp"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "unknown"
        assert body["status_error"] is None
        assert body["status_checked_at"] is None

    def test_enabled_toggle_keeps_status(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = self._create(client)
        assert client.post(f"/api/admin/mcp/servers/{created['id']}/test").status_code == 200

        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}", json={"enabled": False}
        )
        assert r.status_code == 200
        assert r.json()["status"] == "ok"  # config unchanged -> verdict kept

    def test_stdio_put_rejects_forbidden_env(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post(
            "/api/admin/mcp/servers",
            json={"name": "s", "transport": "stdio",
                  "command": "/usr/local/bin/mcp-server"},
        ).json()

        r = client.put(
            f"/api/admin/mcp/servers/{created['id']}",
            json={"env": {"LD_PRELOAD": "/tmp/x.so"}},
        )
        assert r.status_code == 400


class TestPresetServerRowsReadOnly:
    """System-managed preset rows (databases preset flow): GET/test work,
    PUT/DELETE are 400 — their lifecycle belongs to the owning database."""

    def _seed(self, db_mod, server_id="mcp_preset1"):
        from api.models import McpServerORM
        from api.mcp.secrets import encrypt_secret_dict

        with db_mod.SessionLocal() as db:
            db.add(McpServerORM(
                id=server_id,
                name="preset-postgresql-dbp_1",
                preset_key="postgresql",
                transport="stdio",
                command="dbhub",
                args=["--transport", "stdio"],
                env=encrypt_secret_dict({
                    "DSN": "postgresql://app:pw@db.internal:5432/x",
                    "READONLY": "true",
                }),
                enabled=True,
                status="ok",
            ))
            db.commit()
        return server_id

    def test_list_serves_preset_key(self, isolated_db, monkeypatch):
        server_id = self._seed(isolated_db)
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        rows = client.get("/api/admin/mcp/servers").json()
        row = next(r for r in rows if r["id"] == server_id)
        assert row["preset_key"] == "postgresql"
        # The encrypted DSN is masked in the admin view.
        assert "db.internal" not in str(row.get("env_masked"))

    def test_put_rejected_400_row_untouched(self, isolated_db, monkeypatch):
        server_id = self._seed(isolated_db)
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = client.put(
            f"/api/admin/mcp/servers/{server_id}", json={"enabled": False}
        )
        assert r.status_code == 400
        assert "preset" in r.json()["detail"].lower()
        assert _stored_row(isolated_db, server_id).enabled is True

    def test_delete_rejected_400_row_still_there(self, isolated_db, monkeypatch):
        server_id = self._seed(isolated_db)
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        r = client.delete(f"/api/admin/mcp/servers/{server_id}")
        assert r.status_code == 400
        assert _stored_row(isolated_db, server_id) is not None


# --- delete ---------------------------------------------------------------------
class TestDeleteMcpServer:
    def test_delete_200_then_404(self, isolated_db, monkeypatch):
        app, client, fake = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()

        r = client.delete(f"/api/admin/mcp/servers/{created['id']}")
        assert r.status_code == 200
        assert "message" in r.json()
        assert created["id"] in fake.invalidated
        assert client.delete(f"/api/admin/mcp/servers/{created['id']}").status_code == 404

    def test_delete_cascades_bindings(self, isolated_db, monkeypatch):
        from api.models import ProductMcpServerORM, ProductORM

        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P1"))
            db.add(
                ProductMcpServerORM(
                    id="pmb_1", product_id="prod_1",
                    mcp_server_id=created["id"], enabled=True,
                )
            )
            db.commit()

        assert client.delete(f"/api/admin/mcp/servers/{created['id']}").status_code == 200

        with isolated_db.SessionLocal() as db:
            remaining = db.query(ProductMcpServerORM).count()
        assert remaining == 0


# --- health-check + cached tools ----------------------------------------------------
class TestTestAndToolsEndpoints:
    def test_endpoint_ok_persists_status(self, isolated_db, monkeypatch):
        app, client, fake = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()

        r = client.post(f"/api/admin/mcp/servers/{created['id']}/test")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["detail"] is None
        assert body["tools"] == [{"name": "echo", "description": "Echo tool"}]
        assert fake.health_calls == 1

        row = _stored_row(isolated_db, created["id"])
        assert row.status == "ok"
        assert row.status_checked_at is not None
        assert row.status_error is None

    def test_endpoint_failure_persists_error(self, isolated_db, monkeypatch):
        fake = _FakeManager()
        fake.health = (False, "timeout", [])
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch, fake=fake)
        created = client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()

        r = client.post(f"/api/admin/mcp/servers/{created['id']}/test")
        assert r.status_code == 200
        assert r.json()["ok"] is False
        assert r.json()["detail"] == "timeout"

        row = _stored_row(isolated_db, created["id"])
        assert row.status == "error"
        assert row.status_error == "timeout"

    def test_test_unknown_server_404(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        assert client.post("/api/admin/mcp/servers/mcp_nope/test").status_code == 404

    def test_tools_served_from_cache_without_connecting(self, isolated_db, monkeypatch):
        fake = _FakeManager()
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch, fake=fake)
        created = client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()
        fake.tools_meta[created["id"]] = [{"name": "echo", "description": None}]

        r = client.get(f"/api/admin/mcp/servers/{created['id']}/tools")
        assert r.status_code == 200
        assert r.json() == [{"name": "echo", "description": None}]
        assert fake.health_calls == 0  # never reconnected

    def test_tools_empty_cache_returns_empty_list(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()
        r = client.get(f"/api/admin/mcp/servers/{created['id']}/tools")
        assert r.status_code == 200
        assert r.json() == []

    def test_tools_unknown_server_404(self, isolated_db, monkeypatch):
        app, client, _ = _build_client(isolated_db, _mod(), monkeypatch)
        assert client.get("/api/admin/mcp/servers/mcp_nope/tools").status_code == 404

    def test_second_test_within_window_429(self, isolated_db, monkeypatch):
        """Rate limit: /test doubles as a connectivity oracle — throttle it."""
        app, client, fake = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/mcp/servers", json=_HTTP_BODY).json()

        first = client.post(f"/api/admin/mcp/servers/{created['id']}/test")
        assert first.status_code == 200
        second = client.post(f"/api/admin/mcp/servers/{created['id']}/test")
        assert second.status_code == 429
        assert fake.health_calls == 1  # the 429 didn't dial
