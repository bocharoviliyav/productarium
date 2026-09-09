#!/usr/bin/env python3
"""Unit tests for the markitdown upload size cap (P0-11).

Uploads are capped at ``limits.upload_max_bytes`` setting >
``UPLOAD_MAX_BYTES`` env > 50 MiB default; both a Content-Length pre-check
and a bounded read enforce it (413).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _admin():
    from api.models import UserORM

    return UserORM(
        id="user_admin1", username="admin", role="admin",
        provider="local", created_at=datetime.utcnow(),
    )


def _build_client(db_mod):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import knowledge as knowledge_mod

    app = FastAPI()
    app.include_router(knowledge_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[knowledge_mod.get_db] = _get_test_db
    app.dependency_overrides[knowledge_mod.get_current_user] = _admin
    return app, TestClient(app)


def _seed(db_mod):
    from api.models import KnowledgeNodeORM, ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id="prod_1", name="P1", description="d"))
        db.add(KnowledgeNodeORM(id="node_1", product_id="prod_1",
                                title="N", slug="n"))
        db.commit()


class TestUploadMaxBytesResolution:
    def test_default_50mib(self, monkeypatch):
        from api.routers import knowledge as knowledge_mod

        monkeypatch.delenv("UPLOAD_MAX_BYTES", raising=False)
        assert knowledge_mod._upload_max_bytes() == 50 * 1024 * 1024

    def test_env_override(self, monkeypatch):
        from api.routers import knowledge as knowledge_mod

        monkeypatch.setenv("UPLOAD_MAX_BYTES", "123")
        assert knowledge_mod._upload_max_bytes() == 123

    def test_invalid_env_falls_back(self, monkeypatch):
        from api.routers import knowledge as knowledge_mod

        monkeypatch.setenv("UPLOAD_MAX_BYTES", "not-a-number")
        assert knowledge_mod._upload_max_bytes() == 50 * 1024 * 1024


class TestUploadEnforcement:
    def test_oversized_upload_413(self, isolated_db, monkeypatch):
        from api.routers import knowledge as knowledge_mod

        _seed(isolated_db)
        monkeypatch.setattr(knowledge_mod, "_upload_max_bytes", lambda: 16)
        app, client = _build_client(isolated_db)

        resp = client.post(
            "/api/products/prod_1/knowledge/nodes/node_1/upload",
            files={"file": ("big.txt", b"x" * 64, "text/plain")},
        )
        assert resp.status_code == 413
        assert "exceeds" in resp.json()["detail"].lower()

    def test_small_upload_passes(self, isolated_db, monkeypatch):
        from api.routers import knowledge as knowledge_mod

        _seed(isolated_db)
        monkeypatch.setattr(knowledge_mod, "_upload_max_bytes", lambda: 10_000_000)
        app, client = _build_client(isolated_db)

        resp = client.post(
            "/api/products/prod_1/knowledge/nodes/node_1/upload",
            files={"file": ("small.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 200, resp.text
