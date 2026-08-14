#!/usr/bin/env python3
"""Unit tests for the rlm router (POST /api/rlm/run)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_rlm_run_success(monkeypatch):
    """A successful run_rlm_task returns its result dict directly."""
    from api.routers import rlm as rlm_mod

    async def _fake_run(query, model):
        assert query == "hello"
        assert model == "qwen/test"
        return {"answer": "result text", "model": model}

    monkeypatch.setattr(
        "api.rlm.runner.run_rlm_task", _fake_run
    )
    app, client = _build_client(rlm_mod)
    resp = client.post("/api/rlm/run", json={"query": "hello", "model": "qwen/test"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "result text", "model": "qwen/test"}


def test_rlm_run_default_model_none(monkeypatch):
    """When model is omitted it is passed as None to run_rlm_task."""
    from api.routers import rlm as rlm_mod

    captured = {}

    async def _fake_run(query, model):
        captured["query"] = query
        captured["model"] = model
        return {"answer": "ok"}

    monkeypatch.setattr("api.rlm.runner.run_rlm_task", _fake_run)
    app, client = _build_client(rlm_mod)
    resp = client.post("/api/rlm/run", json={"query": "q"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "ok"}
    assert captured["model"] is None
    assert captured["query"] == "q"


def test_rlm_run_exception_returns_500(monkeypatch):
    """An exception from run_rlm_task surfaces as HTTP 500."""
    from api.routers import rlm as rlm_mod

    async def _fake_run(query, model):
        raise RuntimeError("boom")

    monkeypatch.setattr("api.rlm.runner.run_rlm_task", _fake_run)
    app, client = _build_client(rlm_mod)
    resp = client.post("/api/rlm/run", json={"query": "q"})
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


# --- helper -----------------------------------------------------------------
def _build_client(rlm_mod):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(rlm_mod.router)
    return app, TestClient(app)
