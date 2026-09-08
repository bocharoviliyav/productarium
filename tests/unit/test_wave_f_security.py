"""Wave F security hardening tests.

Covers the fixes landed in Wave F:
- Router-level auth on ``products`` / ``docgen`` / ``databases`` (401 without a
  session, pass with one) while ``AUTH_PROVIDER=none`` stays a no-op.
- ``CORS_ORIGINS`` env parsing in ``api.api`` (default allowlist, ``*``
  wildcard disabling credentials, explicit multi-origin list).
- ``COOKIE_SECURE`` env flag in ``api.auth.router`` (module flag + the Set-Cookie
  header actually carrying ``Secure``).
- ``_validate_url`` rejecting embedded userinfo (``user:pass@host``) on the
  admin MCP server registry.
- ``api.utils.fs.open_read_nofollow`` (O_NOFOLLOW on the final component) plus
  the unchanged symlink semantics of the repo tools (symlink-to-inside reads
  the confined target; symlink-to-outside is rejected by confinement).
- Generic client-facing error details: public ``ask`` SSE stream, public
  ``push``, integrations ``list_spaces`` / pull — exception text (which may
  embed internal URLs and credentials) must stay in the server log only.
"""

from __future__ import annotations

import errno
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --- shared helpers -----------------------------------------------------------
def _get_test_db_factory(db_mod):
    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    return _get_test_db


def _admin_user_orm():
    from api.models import UserORM

    return UserORM(
        id="user_admin1",
        username="admin",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


def _api_token_orm():
    from api.models import ApiTokenORM

    return ApiTokenORM(
        id="tok_fixed",
        user_id="user_admin1",
        token_hash="x" * 64,
        name="fixed",
        created_at=datetime.utcnow(),
    )


def _seed_product(db_mod, product_id="prod_1", name="Acme"):
    from api.models import ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name=name))
        db.commit()
    return product_id


# --- A. Router-level auth (products / docgen / databases) ---------------------
class TestRouterLevelAuthz:
    def _build_client(self, isolated_db, monkeypatch):
        """App over the three hardened routers with get_db overridden but NO
        auth override — requests must 401 while AUTH_PROVIDER=local."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth import deps as auth_deps
        from api.routers import databases, docgen, products

        app = FastAPI()
        for mod in (products, docgen, databases):
            app.include_router(mod.router)

        _get_test_db = _get_test_db_factory(isolated_db)
        app.dependency_overrides[auth_deps.get_db] = _get_test_db
        # Guarantee local-auth mode regardless of import order / developer env.
        monkeypatch.setattr(auth_deps, "AUTH_PROVIDER", "local")
        return app, TestClient(app), auth_deps

    def test_products_router_requires_session(self, isolated_db, monkeypatch):
        app, client, auth_deps = self._build_client(isolated_db, monkeypatch)

        r = client.get("/api/products")
        assert r.status_code == 401
        r = client.post("/api/products", json={"id": "prod_1", "name": "Acme"})
        assert r.status_code == 401

        # With a resolved user the same calls pass (empty list / create OK).
        app.dependency_overrides[auth_deps.get_current_user] = _admin_user_orm
        r = client.get("/api/products")
        assert r.status_code == 200
        assert r.json() == []
        r = client.post("/api/products", json={"id": "prod_1", "name": "Acme"})
        assert r.status_code == 200

    def test_docgen_router_requires_session(self, isolated_db, monkeypatch):
        app, client, auth_deps = self._build_client(isolated_db, monkeypatch)

        status_url = (
            "/api/products/prod_1/codebases/cb_1/generate/status?job_id=job_x"
        )
        r = client.get(status_url)
        assert r.status_code == 401
        r = client.post(
            "/api/products/prod_1/codebases/cb_1/generate", json={"language": "ru"}
        )
        assert r.status_code == 401

        # Authenticated: the status poll proceeds to the (missing) job lookup.
        app.dependency_overrides[auth_deps.get_current_user] = _admin_user_orm
        r = client.get(status_url)
        assert r.status_code == 404  # job not found — auth passed

    def test_databases_router_requires_session(self, isolated_db, monkeypatch):
        app, client, auth_deps = self._build_client(isolated_db, monkeypatch)
        _seed_product(isolated_db)

        payload = {
            "id": "db_1",
            "name": "pg-main",
            "dsn": "postgresql://user:secretpw@localhost:5432/app",
        }
        r = client.post("/api/products/prod_1/databases", json=payload)
        assert r.status_code == 401

        app.dependency_overrides[auth_deps.get_current_user] = _admin_user_orm
        r = client.post("/api/products/prod_1/databases", json=payload)
        assert r.status_code == 200
        # Bonus invariant (Wave E): the raw DSN never round-trips.
        assert "secretpw" not in r.text


# --- B. CORS_ORIGINS env parsing ----------------------------------------------
class TestCorsConfig:
    def _reload(self, monkeypatch, value):
        import api.api as api_mod

        if value is None:
            monkeypatch.delenv("CORS_ORIGINS", raising=False)
        else:
            monkeypatch.setenv("CORS_ORIGINS", value)
        return api_mod, importlib.reload(api_mod)

    def _restore(self, api_mod, monkeypatch):
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        importlib.reload(api_mod)

    def test_default_allowlist(self, monkeypatch):
        api_mod, mod = self._reload(monkeypatch, None)
        try:
            assert mod._cors_allow_all is False
            assert mod._cors_origins == ["http://localhost:3000"]

            from fastapi.testclient import TestClient

            r = TestClient(mod.app).options(
                "/api/products",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert r.headers.get("access-control-allow-origin") == (
                "http://localhost:3000"
            )
            assert r.headers.get("access-control-allow-credentials") == "true"
        finally:
            self._restore(api_mod, monkeypatch)

    def test_wildcard_disables_credentials(self, monkeypatch):
        api_mod, mod = self._reload(monkeypatch, "*")
        try:
            assert mod._cors_allow_all is True
            assert mod._cors_origins == ["*"]

            from fastapi.testclient import TestClient

            r = TestClient(mod.app).options(
                "/api/products",
                headers={
                    "Origin": "http://anything.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert r.headers.get("access-control-allow-origin") == "*"
            # Wildcard origin + credentials is invalid per the CORS spec.
            assert r.headers.get("access-control-allow-credentials") != "true"
        finally:
            self._restore(api_mod, monkeypatch)

    def test_mixed_wildcard_disables_credentials(self, monkeypatch):
        """A "*" mixed with explicit origins must not smuggle a literal "*"
        member into allow_origins — Starlette would then reflect arbitrary
        origins even with credentials enabled, bypassing the allowlist
        (review #5)."""
        api_mod, mod = self._reload(monkeypatch, "*,http://foo.example")
        try:
            assert mod._cors_allow_all is True
            assert mod._cors_origins == ["*"]

            from fastapi.testclient import TestClient

            r = TestClient(mod.app).options(
                "/api/products",
                headers={
                    "Origin": "http://anything.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert r.headers.get("access-control-allow-origin") == "*"
            # Wildcard origin + credentials is invalid per the CORS spec.
            assert r.headers.get("access-control-allow-credentials") != "true"
        finally:
            self._restore(api_mod, monkeypatch)

    def test_explicit_origin_list(self, monkeypatch):
        api_mod, mod = self._reload(
            monkeypatch, "https://a.example, https://b.example"
        )
        try:
            assert mod._cors_allow_all is False
            assert mod._cors_origins == ["https://a.example", "https://b.example"]
        finally:
            self._restore(api_mod, monkeypatch)


# --- C. COOKIE_SECURE ---------------------------------------------------------
class TestCookieSecure:
    def _reload(self, monkeypatch, value):
        import api.auth.router as router_mod

        if value is None:
            monkeypatch.delenv("COOKIE_SECURE", raising=False)
        else:
            monkeypatch.setenv("COOKIE_SECURE", value)
        return router_mod, importlib.reload(router_mod)

    def _restore(self, router_mod, monkeypatch):
        monkeypatch.delenv("COOKIE_SECURE", raising=False)
        importlib.reload(router_mod)

    def test_flag_parsing(self, monkeypatch):
        router_mod, _ = self._reload(monkeypatch, None)
        try:
            for truthy in ("1", "true", "TRUE", "yes", "on"):
                mod = importlib.reload(router_mod)
                monkeypatch.setenv("COOKIE_SECURE", truthy)
                mod = importlib.reload(router_mod)
                assert mod._COOKIE_SECURE is True, truthy
                assert mod._COOKIE_KWARGS["secure"] is True, truthy
            for falsy in ("", "0", "false", "off"):
                monkeypatch.setenv("COOKIE_SECURE", falsy)
                mod = importlib.reload(router_mod)
                assert mod._COOKIE_SECURE is False, falsy
            monkeypatch.delenv("COOKIE_SECURE", raising=False)
            mod = importlib.reload(router_mod)
            assert mod._COOKIE_SECURE is False
        finally:
            self._restore(router_mod, monkeypatch)

    def _login_client(self, router_mod, isolated_db, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth.local import hash_password
        from api.models import UserORM

        with isolated_db.SessionLocal() as db:
            db.add(
                UserORM(
                    id="u1",
                    username="alice",
                    password_hash=hash_password("pw12345"),
                    role="user",
                    provider="local",
                )
            )
            db.commit()

        app = FastAPI()
        app.include_router(router_mod.router)
        app.dependency_overrides[router_mod.get_db] = _get_test_db_factory(
            isolated_db
        )
        monkeypatch.setattr(router_mod, "AUTH_PROVIDER", "local")
        return TestClient(app)

    def test_login_cookie_carries_secure_when_enabled(
        self, isolated_db, monkeypatch
    ):
        router_mod, mod = self._reload(monkeypatch, "true")
        try:
            client = self._login_client(mod, isolated_db, monkeypatch)
            r = client.post(
                "/api/auth/login", json={"username": "alice", "password": "pw12345"}
            )
            assert r.status_code == 200
            cookie = ",".join(r.headers.get_list("set-cookie")).lower()
            assert "secure" in cookie
            assert "httponly" in cookie
            assert "samesite=lax" in cookie
        finally:
            self._restore(router_mod, monkeypatch)

    def test_login_cookie_no_secure_by_default(self, isolated_db, monkeypatch):
        router_mod, mod = self._reload(monkeypatch, None)
        try:
            client = self._login_client(mod, isolated_db, monkeypatch)
            r = client.post(
                "/api/auth/login", json={"username": "alice", "password": "pw12345"}
            )
            assert r.status_code == 200
            cookie = ",".join(r.headers.get_list("set-cookie")).lower()
            assert "secure" not in cookie
            assert "httponly" in cookie
        finally:
            self._restore(router_mod, monkeypatch)


# --- D. MCP admin URL validation ----------------------------------------------
class TestMcpAdminUrlUserinfo:
    def test_validate_url_unit(self):
        from fastapi import HTTPException

        from api.routers.mcp_admin import _validate_url

        # Clean URLs pass (private hosts are intentionally allowed).
        assert _validate_url("https://mcp.internal:9000/mcp") == (
            "https://mcp.internal:9000/mcp"
        )
        assert _validate_url("  http://localhost:9000/sse  ") == (
            "http://localhost:9000/sse"
        )
        # Embedded userinfo is rejected — auth belongs in (encrypted) headers.
        for bad in (
            "http://user:pass@localhost:9000/mcp",
            "https://token@host.example/mcp",
            "http://u:p@10.0.0.5/",
        ):
            with pytest.raises(HTTPException) as ei:
                _validate_url(bad)
            assert ei.value.status_code == 400
            assert "credentials" in ei.value.detail

    def test_create_server_with_credentials_rejected(
        self, isolated_db, monkeypatch
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth import deps as auth_deps
        from api.routers import mcp_admin as mcp_mod

        app = FastAPI()
        app.include_router(mcp_mod.router)
        _get_test_db = _get_test_db_factory(isolated_db)
        app.dependency_overrides[mcp_mod.get_db] = _get_test_db
        app.dependency_overrides[auth_deps.get_db] = _get_test_db
        app.dependency_overrides[auth_deps.require_admin] = _admin_user_orm
        client = TestClient(app)

        r = client.post(
            "/api/admin/mcp/servers",
            json={
                "name": "leaky",
                "transport": "http",
                "url": "http://user:pass@localhost:9000/mcp",
            },
        )
        assert r.status_code == 400
        assert "credentials" in r.json()["detail"]

        r2 = client.post(
            "/api/admin/mcp/servers",
            json={
                "name": "clean",
                "transport": "http",
                "url": "http://localhost:9000/mcp",
            },
        )
        assert r2.status_code in (200, 201)


# --- E. open_read_nofollow + repo tool semantics ------------------------------
class TestOpenReadNofollow:
    def test_regular_file_text_and_binary(self, tmp_path):
        from api.utils.fs import open_read_nofollow

        f = tmp_path / "plain.txt"
        f.write_text("hello", encoding="utf-8")
        with open_read_nofollow(str(f)) as fh:
            assert fh.read() == "hello"
        with open_read_nofollow(str(f), binary=True) as fh:
            assert fh.read() == b"hello"

    def test_errors_replace(self, tmp_path):
        from api.utils.fs import open_read_nofollow

        f = tmp_path / "bytes.bin"
        f.write_bytes(b"\xff\xfeok")
        with open_read_nofollow(str(f), errors="replace") as fh:
            text = fh.read()
        assert text.endswith("ok")
        assert "\ufffd" in text  # invalid bytes replaced, not raised

    def test_symlink_final_component_raises(self, tmp_path):
        from api.utils.fs import open_read_nofollow

        target = tmp_path / "target.txt"
        target.write_text("data", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        with pytest.raises(OSError) as ei:
            open_read_nofollow(str(link))
        if hasattr(os, "O_NOFOLLOW"):
            assert ei.value.errno == errno.ELOOP

    def test_missing_file_raises(self, tmp_path):
        from api.utils.fs import open_read_nofollow

        with pytest.raises(OSError):
            open_read_nofollow(str(tmp_path / "nope.txt"))

    def test_repo_read_file_symlink_semantics(self, tmp_path):
        """repo_read_file keeps its confinement contract under O_NOFOLLOW.

        A symlink pointing INSIDE the clone resolves to a confined regular
        file and reads the target (realpath resolves before the open); a
        symlink pointing OUTSIDE is rejected by ``_confined_path``.
        """
        from api.docgen.codebase import build_repo_tools

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "real.txt").write_text("confined content", encoding="utf-8")
        (repo / "inside.txt").symlink_to(repo / "real.txt")
        (repo / "outside.txt").symlink_to(tmp_path / "outside-target.txt")
        (tmp_path / "outside-target.txt").write_text("outside secret", encoding="utf-8")

        tools = {t.name: t for t in build_repo_tools(str(repo))}
        assert tools["repo_read_file"].invoke({"path": "real.txt"}) == (
            "confined content"
        )
        # Symlink-to-inside: realpath resolves to the confined target.
        assert tools["repo_read_file"].invoke({"path": "inside.txt"}) == (
            "confined content"
        )
        # Symlink-to-outside: confinement rejects before any open.
        out = tools["repo_read_file"].invoke({"path": "outside.txt"})
        assert out.startswith("ERROR: file not found inside the repository")
        assert "outside secret" not in out


# --- F. Generic client-facing error details -----------------------------------
def _public_client(isolated_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.auth import deps as auth_deps
    from api.routers import public as public_mod

    app = FastAPI()
    app.include_router(public_mod.router)
    _get_test_db = _get_test_db_factory(isolated_db)
    app.dependency_overrides[public_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.require_api_token] = _api_token_orm
    return TestClient(app)


class TestGenericErrorDetails:
    def test_public_ask_sse_generic_error(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        client = _public_client(isolated_db)

        import api.expert.chat as chat_mod

        async def _boom(**kwargs):
            raise RuntimeError(
                "secret internal detail http://10.0.0.5:9/x token=XYZ"
            )
            yield  # pragma: no cover - async generator shape

        monkeypatch.setattr(chat_mod, "run_expert_chat", _boom)

        r = client.post("/api/public/products/prod_1/ask", json={"query": "hi"})
        assert r.status_code == 200
        assert "internal error while streaming the answer" in r.text
        assert "XYZ" not in r.text
        assert "10.0.0.5" not in r.text

    def test_public_push_generic_502(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        client = _public_client(isolated_db)

        from api.integrations import registry as registry_mod

        class _FakeConnector:
            def push(self, payload):
                raise RuntimeError(
                    "git remote http://user:SECRETPUSH@github.internal/x"
                )

        monkeypatch.setattr(
            registry_mod, "get_connector", lambda name: _FakeConnector()
        )

        r = client.post(
            "/api/public/products/prod_1/push", json={"target": "confluence"}
        )
        assert r.status_code == 502
        assert r.json()["detail"] == "Push to 'confluence' failed"
        assert "SECRETPUSH" not in r.text
        assert "github.internal" not in r.text

    def test_integrations_list_spaces_generic_502(self, isolated_db, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth import deps as auth_deps
        from api.routers import integrations as integ_mod

        app = FastAPI()
        app.include_router(integ_mod.router)
        _get_test_db = _get_test_db_factory(isolated_db)
        app.dependency_overrides[integ_mod.get_db] = _get_test_db
        app.dependency_overrides[auth_deps.get_db] = _get_test_db
        app.dependency_overrides[auth_deps.get_current_user] = _admin_user_orm

        def _spaces():
            raise RuntimeError(
                "confluence http://user:SECRETTOKEN@confluence.internal/x"
            )

        fake = SimpleNamespace(list_spaces=_spaces)
        monkeypatch.setattr(integ_mod, "get_connector", lambda name: fake)

        r = TestClient(app).get("/api/integrations/confluence/spaces")
        assert r.status_code == 502
        assert r.json()["detail"] == (
            "Failed to list sources for connector 'confluence'"
        )
        assert "SECRETTOKEN" not in r.text
        assert "confluence.internal" not in r.text

    def test_integrations_pull_generic_502(self, isolated_db, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth import deps as auth_deps
        from api.routers import integrations as integ_mod

        _seed_product(isolated_db)
        app = FastAPI()
        app.include_router(integ_mod.router)
        _get_test_db = _get_test_db_factory(isolated_db)
        app.dependency_overrides[integ_mod.get_db] = _get_test_db
        app.dependency_overrides[auth_deps.get_db] = _get_test_db
        app.dependency_overrides[auth_deps.get_current_user] = _admin_user_orm

        def _pull(source_id, opts=None):
            raise RuntimeError("clone failed: http://x-token:SECRET@host/repo")

        fake = SimpleNamespace(pull=_pull)
        monkeypatch.setattr(integ_mod, "get_connector", lambda name: fake)

        r = TestClient(app).post(
            "/api/products/prod_1/codebases/from-integration",
            json={"connector": "github", "source_id": "octocat/hello"},
        )
        assert r.status_code == 502
        assert r.json()["detail"] == "Pull failed; see server logs for details"
        assert "SECRET" not in r.text


# --- H. DOCGEN SYMLINK HARDENING (review #5) ----------------------------------
class TestDocgenSymlinkHardening:
    """Assembly-time readers must be as symlink-safe as the agent tools.

    Git preserves symlinks, so a malicious repo can plant
    ``README.md -> /etc/passwd`` or ``evil.py -> ~/secret.py``; the docgen
    context readers must never follow them outside the clone.
    """

    def test_read_all_documents_skips_symlinks(self, tmp_path):
        from api.repositories.documents import read_all_documents

        (tmp_path / "main.py").write_text("print('ok')\n")
        secret = tmp_path.parent / "secret_outside.py"
        secret.write_text("ROOT_SECRET = 'abc123'\n")
        os.symlink(secret, tmp_path / "evil.py")
        # A symlink pointing INSIDE the clone is skipped by collection too
        # (stricter than the agent tools, which read confined targets).
        os.symlink(tmp_path / "main.py", tmp_path / "alias.py")

        docs = read_all_documents(str(tmp_path))
        paths = {d.meta_data["file_path"] for d in docs}
        assert "main.py" in paths
        assert "evil.py" not in paths
        assert "alias.py" not in paths
        assert all("ROOT_SECRET" not in (d.text or "") for d in docs)

    def test_read_readme_symlink_outside_rejected(self, tmp_path):
        from api.docgen.codebase import _read_readme

        (tmp_path / "main.py").write_text("x = 1\n")
        secret = tmp_path.parent / "host_secret.md"
        secret.write_text("# host secret\n")
        os.symlink(secret, tmp_path / "README.md")
        assert _read_readme(str(tmp_path)) == ""

    def test_read_readme_plain_file_ok(self, tmp_path):
        from api.docgen.codebase import _read_readme

        (tmp_path / "README.md").write_text("# hello\n")
        assert _read_readme(str(tmp_path)) == "# hello\n"

    def test_git_connector_find_readme_symlink_rejected(self, tmp_path):
        from api.integrations._git_base import GitConnector

        secret = tmp_path.parent / "host_readme.md"
        secret.write_text("# host secret\n")
        os.symlink(secret, tmp_path / "README.md")
        assert GitConnector._find_readme(str(tmp_path)) is None

    def test_git_connector_find_readme_plain_ok(self, tmp_path):
        from api.integrations._git_base import GitConnector

        (tmp_path / "README.md").write_text("# hello\n")
        assert GitConnector._find_readme(str(tmp_path)) == "# hello\n"
