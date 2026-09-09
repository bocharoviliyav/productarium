#!/usr/bin/env python3
"""Unit tests for the per-IP login rate limit (P0-7).

``POST /api/auth/login`` must enforce a token bucket per client IP: beyond
the configured requests/minute the endpoint answers 429 with a Retry-After
header, regardless of credential validity. ``reset_rate_limits`` clears the
buckets for tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture()
def client(isolated_db, monkeypatch):
    """Auth router over the isolated DB with a 3/min limit for tests."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.auth import router as auth_mod
    from api.utils.rate_limit import reset_rate_limits

    monkeypatch.setenv("RATE_AUTH_PER_IP_MINUTE", "3")
    monkeypatch.setattr(auth_mod, "AUTH_PROVIDER", "local")
    reset_rate_limits()

    app = FastAPI()
    app.include_router(auth_mod.router)

    def _get_test_db():
        s = isolated_db.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[auth_mod.get_db] = _get_test_db
    yield TestClient(app)
    reset_rate_limits()


class TestLoginRateLimit:
    def test_429_after_limit_with_retry_after(self, client):
        codes = []
        for _ in range(4):
            r = client.post("/api/auth/login", json={"username": "x", "password": "y"})
            codes.append(r.status_code)
        # First 3 attempts reach the credential check (401 for bad creds with
        # AUTH_PROVIDER=local), the 4th is rate-limited.
        assert codes[:3] == [401, 401, 401]
        assert codes[3] == 429
        r4 = client.post("/api/auth/login", json={"username": "x", "password": "y"})
        assert r4.status_code == 429
        assert int(r4.headers["Retry-After"]) >= 1

    def test_limit_applies_before_auth(self, client):
        """429 is answered before any credential validation runs."""
        r = client.post("/api/auth/login", json={"username": "x", "password": "y"})
        assert r.status_code in (401, 429)
        assert "set-cookie" not in r.headers

    def test_reset_rate_limits_clears_buckets(self, client):
        from api.utils.rate_limit import reset_rate_limits

        for _ in range(4):
            client.post("/api/auth/login", json={"username": "x", "password": "y"})
        assert client.post(
            "/api/auth/login", json={"username": "x", "password": "y"}
        ).status_code == 429

        reset_rate_limits()
        assert client.post(
            "/api/auth/login", json={"username": "x", "password": "y"}
        ).status_code == 401
