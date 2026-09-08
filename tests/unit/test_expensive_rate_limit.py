#!/usr/bin/env python3
"""Unit tests for the per-user rate limits on expensive endpoints (P1-17).

docgen generate / expert ask (+ ask/doc) / public ask share the token-bucket
infrastructure from P0-7 (``api.utils.rate_limit``). Each endpoint family has
its own settings key + env fallback; beyond the limit the endpoint answers 429
with a ``Retry-After`` header, and buckets are per user (per token owner for
the public API — tokens without a linked user key on the token id).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture(autouse=True)
def _reset_limits():
    from api.utils.rate_limit import reset_rate_limits

    reset_rate_limits()
    yield
    reset_rate_limits()


def _user(uid: str = "user_1"):
    from api.models import UserORM

    return UserORM(
        id=uid, username=uid, role="admin", provider="local",
        created_at=datetime.utcnow(),
    )


def _api_token(uid: str = "user_1", tid: str = "tok_1"):
    from api.models import ApiTokenORM

    return ApiTokenORM(
        id=tid, user_id=uid, token_hash="x" * 64, name="t",
        created_at=datetime.utcnow(),
    )


def _db_override(db_mod):
    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    return _get_test_db


# --- docgen generate ----------------------------------------------------------
class TestDocgenRateLimit:
    @pytest.fixture
    def client(self, isolated_db, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth import deps as auth_deps
        from api.models import CodebaseORM, ProductORM
        from api.routers import docgen as docgen_mod

        monkeypatch.setenv("RATE_DOCGEN_PER_USER_HOUR", "2")

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme"))
            db.flush()
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="svc", source="manual",
            ))
            db.commit()

        # Never actually run a job in these tests.
        monkeypatch.setattr(docgen_mod, "submit_job", lambda *a, **k: None)

        app = FastAPI()
        app.include_router(docgen_mod.router)
        ov = _db_override(isolated_db)
        # require_product_access resolves get_current_user/get_db from
        # api.auth.deps — the same function objects the router imported.
        app.dependency_overrides[docgen_mod.get_db] = ov
        app.dependency_overrides[auth_deps.get_db] = ov
        app.dependency_overrides[auth_deps.get_current_user] = _user
        return TestClient(app)

    def test_429_after_limit_with_retry_after(self, client):
        codes = [
            client.post(
                "/api/products/prod_1/codebases/cb_1/generate", json={}
            ).status_code
            for _ in range(3)
        ]
        assert codes[:2] == [202, 202]
        assert codes[2] == 429

        r = client.post("/api/products/prod_1/codebases/cb_1/generate", json={})
        assert r.status_code == 429
        assert int(r.headers["Retry-After"]) >= 1
        assert "Too many requests" in r.json()["detail"]

    def test_buckets_are_per_user(self, client, monkeypatch):
        from api.auth import deps as auth_deps

        codes = [
            client.post(
                "/api/products/prod_1/codebases/cb_1/generate", json={}
            ).status_code
            for _ in range(2)
        ]
        assert codes == [202, 202]

        # A different user starts with a fresh bucket.
        client.app.dependency_overrides[auth_deps.get_current_user] = lambda: _user("user_2")
        r = client.post("/api/products/prod_1/codebases/cb_1/generate", json={})
        assert r.status_code == 202

    def test_spec_generate_shares_the_docgen_budget(self, client, isolated_db, monkeypatch):
        from api.models import ProductORM, SpecORM

        with isolated_db.SessionLocal() as db:
            db.add(SpecORM(
                id="spec_1", product_id="prod_1", name="s",
                kind="openapi", source="manual",
            ))
            db.commit()

        # Two codebase generates exhaust the hourly budget…
        for _ in range(2):
            assert client.post(
                "/api/products/prod_1/codebases/cb_1/generate", json={}
            ).status_code == 202
        # …so the spec generate is limited by the SAME bucket.
        r = client.post("/api/products/prod_1/specs/spec_1/generate", json={})
        assert r.status_code == 429


# --- expert ask / ask/doc ------------------------------------------------------
class TestExpertRateLimit:
    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth import deps as auth_deps
        from api.routers import expert as expert_mod
        from api.expert.types import EVENT_CONTENT, ExpertStreamEvent

        monkeypatch.setenv("RATE_EXPERT_PER_USER_MINUTE", "2")

        def _fake_chat(*args, **kw):
            async def gen():
                yield ExpertStreamEvent(EVENT_CONTENT, "ok")

            return gen()

        async def _fake_doc(*args, **kw):
            return "# doc"

        monkeypatch.setattr(expert_mod, "run_agent_chat_stream", _fake_chat)
        monkeypatch.setattr(expert_mod, "run_agent_doc", _fake_doc)

        app = FastAPI()
        app.include_router(expert_mod.router)
        app.dependency_overrides[auth_deps.get_current_user] = _user
        return TestClient(app)

    def test_ask_429_after_limit_with_retry_after(self, client):
        codes = [
            client.post("/api/products/prod_1/ask", json={"query": "q"}).status_code
            for _ in range(3)
        ]
        assert codes[:2] == [200, 200]
        assert codes[2] == 429
        r = client.post("/api/products/prod_1/ask", json={"query": "q"})
        assert int(r.headers["Retry-After"]) >= 1

    def test_ask_doc_shares_the_expert_budget(self, client):
        assert client.post("/api/products/prod_1/ask", json={"query": "q"}).status_code == 200
        assert client.post(
            "/api/products/prod_1/ask/doc", json={"query": "q"}
        ).status_code == 200
        # Both endpoints draw from the same rate.expert.per_user_min bucket.
        assert client.post(
            "/api/products/prod_1/ask/doc", json={"query": "q"}
        ).status_code == 429


# --- public ask ----------------------------------------------------------------
class TestPublicAskRateLimit:
    @pytest.fixture
    def client(self, isolated_db, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth import deps as auth_deps
        from api.models import ProductORM
        from api.routers import public as public_mod

        monkeypatch.setenv("RATE_PUBLIC_PER_USER_MINUTE", "2")

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme"))
            db.commit()

        async def _fake_stream(*args, **kwargs):
            yield "chunk"

        import api.expert.chat as expert_chat_mod

        monkeypatch.setattr(expert_chat_mod, "run_expert_chat", _fake_stream)

        app = FastAPI()
        app.include_router(public_mod.router)
        ov = _db_override(isolated_db)
        app.dependency_overrides[public_mod.get_db] = ov
        app.dependency_overrides[auth_deps.get_db] = ov
        app.dependency_overrides[auth_deps.require_api_token] = _api_token
        return TestClient(app)

    def test_429_after_limit(self, client):
        codes = [
            client.post("/api/public/products/prod_1/ask", json={"query": "q"}).status_code
            for _ in range(3)
        ]
        assert codes[:2] == [200, 200]
        assert codes[2] == 429
        r = client.post("/api/public/products/prod_1/ask", json={"query": "q"})
        assert int(r.headers["Retry-After"]) >= 1

    def test_bucket_keyed_by_token_owner(self, client, monkeypatch):
        from api.auth import deps as auth_deps

        for _ in range(2):
            assert client.post(
                "/api/public/products/prod_1/ask", json={"query": "q"}
            ).status_code == 200
        assert client.post(
            "/api/public/products/prod_1/ask", json={"query": "q"}
        ).status_code == 429

        # A token of a DIFFERENT user has its own bucket…
        client.app.dependency_overrides[auth_deps.require_api_token] = lambda: _api_token("user_2", "tok_2")
        assert client.post(
            "/api/public/products/prod_1/ask", json={"query": "q"}
        ).status_code == 200
        # …while two tokens of the SAME user share one.
        client.app.dependency_overrides[auth_deps.require_api_token] = lambda: _api_token("user_1", "tok_3")
        assert client.post(
            "/api/public/products/prod_1/ask", json={"query": "q"}
        ).status_code == 429

    def test_token_without_user_keys_on_token_id(self, client):
        from api.auth import deps as auth_deps

        # Legacy token with no linked user: keyed by the token id.
        client.app.dependency_overrides[auth_deps.require_api_token] = lambda: _api_token(None, "tok_anon")
        assert client.post(
            "/api/public/products/prod_1/ask", json={"query": "q"}
        ).status_code == 200
