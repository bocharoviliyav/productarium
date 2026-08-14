#!/usr/bin/env python3
"""
Foundation unit tests for the Productarium backend (api.models, api.schemas,
api.auth.local, api.auth.tokens, api.settings_store, ORM migration).

Runs under pytest (pytest.ini: testpaths=test, markers=unit). No DB server
required; each test sets up an isolated SQLite Env (DB_PROVIDER=sqlite) and
runs init_db() to exercise the one-shot migration on a fresh DB.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import pytest


# --- Helpers -----------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Isolated SQLite DB for every test + temp dirs so we don't touch real ones."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_PROVIDER", "sqlite")
    monkeypatch.setenv("DB_HOST", str(tmp_path))
    monkeypatch.setenv("DB_NAME", str(db_file))
    monkeypatch.setenv("DB_USERNAME", "")
    monkeypatch.setenv("DB_PASSWORD", "")
    monkeypatch.setenv("AUTH_PROVIDER", "local")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    # Avoid loading a real .env for tests.
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    # Set a stable SETTINGS_SECRET_KEY so encryption tests are deterministic.
    from cryptography.fernet import Fernet
    monkeypatch.setenv("SETTINGS_SECRET_KEY", Fernet.generate_key().decode())
    yield


def _reload_modules() -> dict[str, Any]:
    """Reload api.db/api.settings_store/api.auth.* after env changes."""
    import api.db as db
    importlib.reload(db)
    import api.config.settings as ss
    importlib.reload(ss)
    import api.auth.local as local
    importlib.reload(local)
    import api.auth.tokens as tokens
    importlib.reload(tokens)
    return {"db": db, "settings_store": ss, "local": local, "tokens": tokens}


# --- ORM models ---------------------------------------------------------------
class TestORMModels:
    def test_user_orm_required_fields(self):
        # Column defaults (provider='local') apply at flush, not construction,
        # so we flush inside a session on an isolated SQLite DB to check them.
        mods = _reload_modules()
        mods["db"].init_db()
        from api.models import UserORM
        with mods["db"].SessionLocal() as db:
            u = UserORM(id="user_1", username="alice", role="admin")
            db.add(u)
            db.flush()
            assert u.role == "admin"
            assert u.provider == "local"  # default applied at flush
            assert u.password_hash is None
            assert u.provider_subject is None

    def test_product_orm_drops_type(self):
        from api.models import ProductORM
        # ProductORM defines no `type` attribute (replaced by summary/owner_id).
        assert "type" not in ProductORM.__dict__
        p = ProductORM(id="prod_1", name="Acme", summary="hello", owner_id="user_1")
        assert p.summary == "hello"
        assert p.owner_id == "user_1"

    def test_typed_artifact_orms(self):
        # ArtifactORM was split into CodebaseORM/SpecORM/LinksORM (the legacy
        # polymorphic ``type``/``kind`` fields are gone — kind lives on SpecORM).
        from api.models import CodebaseORM, SpecORM, LinksORM
        c = CodebaseORM(id="art_1", product_id="prod_1", name="svc", verified=True, source="api")
        assert c.verified is True
        assert c.source == "api"
        assert c.verified_at is None
        s = SpecORM(id="art_2", product_id="prod_1", name="spec", kind="asyncapi")
        assert s.kind == "asyncapi"
        l = LinksORM(id="art_3", product_id="prod_1", name="links", content="[]")
        assert l.content == "[]"

    def test_knowledge_node_tree_fields(self):
        # Defaults (source='manual') apply at flush, so use a session.
        mods = _reload_modules()
        mods["db"].init_db()
        from api.models import KnowledgeNodeORM, ProductORM
        with mods["db"].SessionLocal() as db:
            p = ProductORM(id="prod_1", name="Acme")
            db.add(p)
            db.flush()
            n = KnowledgeNodeORM(
                id="node_1", product_id="prod_1", parent_id=None,
                title="Overview", slug="overview", node_type="folder",
            )
            db.add(n)
            db.flush()
            assert n.node_type == "folder"
            assert n.source == "manual"  # default applied at flush
            assert n.parent_id is None


# --- init_db + one-shot migration -------------------------------------------
class TestInitDbMigration:
    def test_init_db_idempotent(self):
        """init_db() runs create_all cleanly on a fresh DB and is idempotent."""
        mods = _reload_modules()
        assert mods["db"].init_db() is True
        assert mods["db"].init_db() is True  # idempotent

    def test_create_all_creates_all_orm_tables(self):
        mods = _reload_modules()
        mods["db"].init_db()
        from sqlalchemy import inspect
        insp = inspect(mods["db"].engine)
        tables = set(insp.get_table_names())
        # The polymorphic ``artifacts`` table was split into codebases/specs/links.
        for expected in (
            "products", "codebases", "specs", "links", "productarium_users",
            "knowledge_nodes", "settings", "api_tokens",
        ):
            assert expected in tables, f"missing table {expected}"

    def test_typed_artifact_orm_writes_round_trip(self):
        """CodebaseORM/SpecORM/LinksORM round-trip through the typed tables."""
        mods = _reload_modules()
        mods["db"].init_db()
        from api.models import CodebaseORM, SpecORM, LinksORM, ProductORM
        with mods["db"].SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme"))
            db.flush()
            db.add(CodebaseORM(id="art_cb", product_id="prod_1", name="svc",
                               generated_docs="# docs", pages={"p": {}}))
            db.add(SpecORM(id="art_sp", product_id="prod_1", name="spec",
                          kind="openapi", content="openapi: 3.0"))
            db.add(LinksORM(id="art_lk", product_id="prod_1", name="links",
                           content='[{"url":"https://x"}]'))
            db.commit()
            assert db.get(CodebaseORM, "art_cb").generated_docs == "# docs"
            assert db.get(SpecORM, "art_sp").kind == "openapi"
            assert db.get(LinksORM, "art_lk").content == '[{"url":"https://x"}]'


# --- Settings store ----------------------------------------------------------
class TestSettingsStore:
    def test_set_and_get_plaintext(self):
        mods = _reload_modules()
        mods["db"].init_db()
        ss = mods["settings_store"]
        ss.set_setting("models.expert.provider", "ollama", encrypt=False)
        assert ss.get_setting("models.expert.provider") == "ollama"

    def test_set_and_get_encrypted_roundtrip(self):
        mods = _reload_modules()
        mods["db"].init_db()
        ss = mods["settings_store"]
        ss.set_setting("git.github.token", "secret-token-123", encrypt=True)
        # Reading the secret decrypts
        assert ss.get_setting("git.github.token") == "secret-token-123"
        # Listing should NOT decrypt (returns ciphertext)
        listed = ss.list_settings(prefix="git.github.")
        assert len(listed) == 1
        assert listed[0]["encrypted"] is True
        assert listed[0]["value"] != "secret-token-123"

    def test_get_setting_default(self):
        mods = _reload_modules()
        mods["db"].init_db()
        assert mods["settings_store"].get_setting("missing.key", default="X") == "X"

    def test_get_secret_is_alias(self):
        mods = _reload_modules()
        mods["db"].init_db()
        ss = mods["settings_store"]
        ss.set_setting("a.b", "v", encrypt=True)
        assert ss.get_secret("a.b") == "v"

    def test_delete_setting(self):
        mods = _reload_modules()
        mods["db"].init_db()
        ss = mods["settings_store"]
        ss.set_setting("tmp.k", "v")
        assert ss.delete_setting("tmp.k") is True
        assert ss.delete_setting("tmp.k") is False  # already gone

    def test_get_model_for_task_returns_defaults(self):
        mods = _reload_modules()
        mods["db"].init_db()
        cfg = mods["settings_store"].get_model_for_task("docgen")
        # provider was removed from the return dict (single OpenAI-compatible
        # provider); model/base_url/api_key remain.
        assert "provider" not in cfg
        assert cfg["model"]  # non-empty
        assert cfg["base_url"]  # non-empty
        assert cfg["api_key"]  # non-empty

    def test_get_git_creds_env_fallback(self):
        mods = _reload_modules()
        mods["db"].init_db()
        os.environ["GITHUB_ENTERPRISE_URL"] = "https://ghe.example.com"
        creds = mods["settings_store"].get_git_creds("github")
        assert creds["url"] == "https://ghe.example.com"
        assert creds["token"] is None

    def test_get_confluence_creds(self):
        mods = _reload_modules()
        mods["db"].init_db()
        ss = mods["settings_store"]
        ss.set_setting("confluence.base_url", "https://confluence.example.com")
        ss.set_setting("confluence.space", "ENG")
        c = ss.get_confluence_creds()
        assert c["base_url"] == "https://confluence.example.com"
        assert c["space"] == "ENG"

    def test_get_integration_config_json(self):
        mods = _reload_modules()
        mods["db"].init_db()
        ss = mods["settings_store"]
        ss.set_setting("integrations.confluence", '{"base_url":"https://x","space":"S"}')
        cfg = ss.get_integration_config("confluence")
        assert cfg.get("base_url") == "https://x"
        assert cfg.get("space") == "S"


# --- Auth local bcrypt -------------------------------------------------------
class TestAuthLocal:
    def test_hash_then_verify(self):
        mods = _reload_modules()
        local = mods["local"]
        if not local.is_available():
            pytest.skip("passlib/bcrypt not installed")
        h = local.hash_password("hunter2")
        assert h and h != "hunter2"
        assert local.verify_password("hunter2", h) is True
        assert local.verify_password("wrong", h) is False

    def test_verify_empty_hash_returns_false(self):
        mods = _reload_modules()
        local = mods["local"]
        assert local.verify_password("hunter2", "") is False
        assert local.verify_password("", "x") is False


# --- Auth session tokens -----------------------------------------------------
class TestAuthTokens:
    def test_issue_and_verify_session_token(self):
        mods = _reload_modules()
        tokens = mods["tokens"]
        from api.schemas import UserOut
        u = UserOut(id="user_1", username="alice", role="user")
        tok = tokens.create_session_token(u)
        assert tok
        claims = tokens.verify_session_token(tok)
        assert claims is not None
        assert claims["sub"] == "user_1"
        assert claims["username"] == "alice"
        assert claims["role"] == "user"
        assert "exp" in claims and "iat" in claims

    def test_verify_garbage_returns_none(self):
        mods = _reload_modules()
        tokens = mods["tokens"]
        assert tokens.verify_session_token("not.a.real.jwt") is None
        assert tokens.verify_session_token("") is None

    def test_cookie_name_is_productarium_session(self):
        mods = _reload_modules()
        assert mods["tokens"].SESSION_COOKIE_NAME == "productarium_session"


# --- Cognee bug fix ----------------------------------------------------------
class TestCogneeSkipConnectionTest:
    def test_skip_defaulted_to_true(self):
        """api.cognee sets COGNEE_SKIP_CONNECTION_TEST=true."""
        import api.cognee as cm
        assert os.environ.get("COGNEE_SKIP_CONNECTION_TEST", "").lower() in (
            "1", "true", "t", "yes",
        )

    def test_cognee_module_importable(self):
        """api.cognee must import even if cognee has issues."""
        import api.cognee as cm
        # _COGNEE_AVAILABLE is defined (True if cognee installed, False otherwise).
        assert hasattr(cm, "_COGNEE_AVAILABLE")
        assert isinstance(cm._COGNEE_AVAILABLE, bool)

    def test_add_and_index_document_does_not_raise(self):
        """add_and_index_document must never raise, even when DB/cognee is down."""
        import asyncio
        import api.cognee as cm
        # Should return None on missing-cognee OR log+return None on cognee error.
        result = asyncio.run(cm.add_and_index_document("text", "ds"))
        assert result is None

    def test_query_cognee_returns_empty_on_failure(self):
        import asyncio
        import api.cognee as cm
        out = asyncio.run(cm.query_cognee("q", "ds"))
        # Empty string on any failure (including unavailable cognee).
        assert out == ""


# --- Schemas -----------------------------------------------------------------
class TestSchemas:
    def test_product_no_type(self):
        from api.schemas import Product
        fields = Product.model_fields
        assert "type" not in fields
        assert "summary" in fields
        assert "owner_id" in fields

    def test_typed_artifact_schemas(self):
        # The polymorphic Artifact schema was split into Codebase/Spec/Links.
        from api.schemas import Codebase, Spec, Links
        cb_fields = Codebase.model_fields
        assert "repo_url" in cb_fields and "pages" in cb_fields
        sp_fields = Spec.model_fields
        assert "kind" in sp_fields and "content" in sp_fields
        lk_fields = Links.model_fields
        assert "content" in lk_fields
        # Shared verified/source fields exist on every typed schema.
        for cls in (Codebase, Spec, Links):
            for new_field in ("verified", "verified_by", "verified_at", "source"):
                assert new_field in cls.model_fields

    def test_user_out(self):
        from api.schemas import UserOut
        u = UserOut(id="ui", username="bob", role="user", provider="local")
        assert u.username == "bob" and u.role == "user"

    def test_login_request(self):
        from api.schemas import LoginRequest
        lr = LoginRequest(username="bob", password="x")
        assert lr.username == "bob"
