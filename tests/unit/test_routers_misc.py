#!/usr/bin/env python3
"""Unit tests for the misc router (health + lang/config endpoints)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_health_check():
    from api.routers import misc

    app, client = _build_client(misc)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["service"] == "productarium-api"
    assert "timestamp" in body


def test_lang_config():
    from api.routers import misc

    app, client = _build_client(misc)
    resp = client.get("/lang/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "supported_languages" in body
    assert "default" in body


def test_health_check_is_async_endpoint():
    """The endpoint is async; verify it returns a plain JSON body (no streaming)."""
    from api.routers import misc

    app, client = _build_client(misc)
    resp = client.get("/health")
    assert resp.headers["content-type"].startswith("application/json")


# --- helper -----------------------------------------------------------------
def _build_client(misc_mod):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(misc_mod.router)
    return app, TestClient(app)
