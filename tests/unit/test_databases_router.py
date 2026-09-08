"""Unit tests for ``api.routers.databases`` (Wave E database artifacts).

Hermetic (in-memory SQLite via ``isolated_db`` + ``build_test_client``):
no job worker thread (``submit_job`` is stubbed), no indexing, no LLM.

Covers:
- POST   /api/products/{id}/databases                    (add + DSN masking
  + leak guard + MCP pin validation 404/400 + product 404)
- DELETE /api/products/{id}/databases/{db_id}            (delete + 404s)
- PUT    /api/products/{id}/databases/{db_id}            (doc-edit shapes:
  pages / page_id+content / generated_docs; meta shape: name / dsn / pin
  set+clear; 400/404s)
- POST   .../verify                                      (owner / admin /
  403 non-owner / 404s)
- POST   .../generate + GET .../generate/status          (202 + poll contract,
  wrong job/product 404)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.models import (
    DatabaseORM,
    McpServerORM,
    ProductMcpServerORM,
    ProductORM,
    UserORM,
)
from api.routers import databases as databases_router_module
from tests.conftest import build_test_client


RAW_DSN = "postgresql://app:sup3rs3cret@db.internal:5432/prod"
MASKED_DSN = "postgresql://app:***REDACTED***@db.internal:5432/prod"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _database_payload(dbid: str = "db_1", **overrides) -> dict:
    payload = {
        "id": dbid,
        "name": "Main DB",
        "dsn": RAW_DSN,
        "mcp_server_id": None,
        "generated_docs": None,
        "pages": None,
        "verified": False,
        "verified_by": None,
        "verified_at": None,
        "source": "manual",
    }
    payload.update(overrides)
    return payload


def _seed_product(db_mod, pid: str = "prod_1", owner_id: Optional[str] = None):
    with db_mod.SessionLocal() as s:
        s.add(ProductORM(id=pid, name="Widget", description="d", owner_id=owner_id))
        s.commit()


def _seed_database(db_mod, pid: str = "prod_1", dbid: str = "db_1", **overrides):
    with db_mod.SessionLocal() as s:
        s.add(DatabaseORM(
            id=dbid, product_id=pid, name="Main DB",
            dsn_masked=MASKED_DSN, source="manual", **overrides,
        ))
        s.commit()


def _seed_mcp_server(db_mod, *, server_id="mcp_1", enabled=True,
                     bound=True, binding_enabled=True):
    with db_mod.SessionLocal() as s:
        s.add(McpServerORM(
            id=server_id, name=f"srv-{server_id}", transport="http",
            url="http://localhost:9000/sse", enabled=enabled,
        ))
        if bound:
            s.add(ProductMcpServerORM(
                id=f"pmb_{server_id}", product_id="prod_1",
                mcp_server_id=server_id, enabled=binding_enabled,
            ))
        s.commit()


def _make_client(db_mod):
    return build_test_client(db_mod, [databases_router_module], auth_none=True)


def _client_with_user(db_mod, user):
    app, client = build_test_client(db_mod, [databases_router_module], auth_none=True)
    app.dependency_overrides[databases_router_module.get_current_user] = lambda: user
    return app, client


def _disable_reindex(monkeypatch):
    monkeypatch.setattr(databases_router_module, "_reindex", lambda *a, **kw: None)


def _stub_submit_job(monkeypatch):
    """No worker thread: record submit calls, leave the job queued."""
    calls = []

    def fake_submit(job_id, product_id, entity_type, entity_id, model, language):
        calls.append({
            "job_id": job_id, "product_id": product_id,
            "entity_type": entity_type, "entity_id": entity_id,
            "model": model, "language": language,
        })

    monkeypatch.setattr(databases_router_module, "submit_job", fake_submit)
    return calls


def _regular_user(uid: str = "user_regular") -> UserORM:
    return UserORM(
        id=uid, username=uid, role="user", provider="local",
        created_at=datetime.utcnow(),
    )


@pytest.fixture(autouse=True)
def _clear_docgen_registry():
    """Реестр docgen-jobs — глобальное состояние модуля: с дедупликацией
    (create_or_get_job) зависший queued-джоб из прошлого теста «съедал» бы
    POST следующего теста для той же сущности. Чистим между тестами."""
    import api.docgen.jobs as jobs_mod
    jobs_mod._docgen_jobs.clear()
    jobs_mod._ENTITY_LOCKS.clear()
    yield
    jobs_mod._docgen_jobs.clear()
    jobs_mod._ENTITY_LOCKS.clear()


# --------------------------------------------------------------------------- #
# POST /api/products/{id}/databases
# --------------------------------------------------------------------------- #
class TestAddDatabase:
    def test_add_masks_dsn_in_response(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/prod_1/databases", json=_database_payload())
        assert r.status_code == 200
        body = r.json()
        assert len(body["databases"]) == 1
        db = body["databases"][0]
        assert db["dsn_masked"] == MASKED_DSN
        # Input-only field: never echoed back.
        assert "dsn" not in db or db.get("dsn") is None
        # Raw DSN must not appear anywhere in the response payload.
        assert "sup3rs3cret" not in r.text

    def test_add_persists_only_masked_dsn(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        client.post("/api/products/prod_1/databases", json=_database_payload())
        with isolated_db.SessionLocal() as s:
            row = s.get(DatabaseORM, "db_1")
            assert row is not None
            assert row.dsn_masked == MASKED_DSN
            # The ORM has no raw-dsn column at all.
            assert not hasattr(row, "dsn") or getattr(row, "dsn", None) is None

    def test_add_roundtrip_dsn_masked_accepted_safely(self, isolated_db):
        """A client echoing dsn_masked back (GET→PUT-style) cannot smuggle
        credentials through the dsn_masked field either."""
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        payload = _database_payload(dsn=None, dsn_masked="postgresql://app:hunter2@db/x")
        r = client.post("/api/products/prod_1/databases", json=payload)
        assert r.status_code == 200
        assert r.json()["databases"][0]["dsn_masked"] == (
            "postgresql://app:***REDACTED***@db/x"
        )

    def test_add_passwordless_dsn_not_500(self, isolated_db):
        # Review #4 HIGH-functional: a DSN with no secret legitimately masks
        # to itself — the leak guard must NOT turn that into a 500.
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases",
            json=_database_payload(dsn="postgres://localhost:5432/db"),
        )
        assert r.status_code == 200
        assert r.json()["databases"][0]["dsn_masked"] == "postgres://localhost:5432/db"

    def test_add_password_with_special_chars_masked(self, isolated_db):
        # Review #4 HIGH: '/' and '@' inside the password must be masked.
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases",
            json=_database_payload(dsn="postgresql://app:p@ss/w0rd@db:5432/prod"),
        )
        assert r.status_code == 200
        assert r.json()["databases"][0]["dsn_masked"] == (
            "postgresql://app:***REDACTED***@db:5432/prod"
        )
        assert "p@ss/w0rd" not in r.text

    def test_add_ignores_client_verified(self, isolated_db):
        # Review #4: verification is server-owned — POSTing verified=true
        # must not grant it (only the owner/admin verify endpoint can).
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases",
            json=_database_payload(verified=True, verified_by="user_evil"),
        )
        assert r.status_code == 200
        db = r.json()["databases"][0]
        assert db["verified"] is False
        assert db["verified_by"] is None
        with isolated_db.SessionLocal() as s:
            row = s.get(DatabaseORM, "db_1")
            assert row.verified is False
            assert row.verified_by is None

    def test_add_missing_product_404(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/missing/databases", json=_database_payload())
        assert r.status_code == 404

    def test_add_unknown_mcp_pin_404(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        payload = _database_payload(mcp_server_id="mcp_ghost")
        r = client.post("/api/products/prod_1/databases", json=payload)
        assert r.status_code == 404
        assert "MCP server not found" in r.json()["detail"]

    def test_add_unbound_mcp_pin_400(self, isolated_db):
        _seed_product(isolated_db)
        _seed_mcp_server(isolated_db, bound=False)
        app, client = _make_client(isolated_db)
        payload = _database_payload(mcp_server_id="mcp_1")
        r = client.post("/api/products/prod_1/databases", json=payload)
        assert r.status_code == 400
        assert "bound" in r.json()["detail"]

    def test_add_disabled_binding_mcp_pin_400(self, isolated_db):
        _seed_product(isolated_db)
        _seed_mcp_server(isolated_db, binding_enabled=False)
        app, client = _make_client(isolated_db)
        payload = _database_payload(mcp_server_id="mcp_1")
        r = client.post("/api/products/prod_1/databases", json=payload)
        assert r.status_code == 400

    def test_add_disabled_server_mcp_pin_400(self, isolated_db):
        _seed_product(isolated_db)
        _seed_mcp_server(isolated_db, enabled=False)
        app, client = _make_client(isolated_db)
        payload = _database_payload(mcp_server_id="mcp_1")
        r = client.post("/api/products/prod_1/databases", json=payload)
        assert r.status_code == 400

    def test_add_valid_mcp_pin_echoed(self, isolated_db):
        _seed_product(isolated_db)
        _seed_mcp_server(isolated_db)
        app, client = _make_client(isolated_db)
        payload = _database_payload(mcp_server_id="mcp_1")
        r = client.post("/api/products/prod_1/databases", json=payload)
        assert r.status_code == 200
        assert r.json()["databases"][0]["mcp_server_id"] == "mcp_1"


# --------------------------------------------------------------------------- #
# DELETE
# --------------------------------------------------------------------------- #
class TestDeleteDatabase:
    def test_delete(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/prod_1/databases/db_1")
        assert r.status_code == 200
        assert r.json()["databases"] == []

    def test_delete_missing_product_404(self, isolated_db):
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/missing/databases/db_1")
        assert r.status_code == 404

    def test_delete_missing_database_is_200_noop(self, isolated_db):
        # Same contract as the codebases router: deleting an absent child is
        # idempotent (the product is returned without it).
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/prod_1/databases/nope")
        assert r.status_code == 200
        assert r.json()["databases"] == []


# --------------------------------------------------------------------------- #
# PUT — doc-edit shapes (artifact viewer editor)
# --------------------------------------------------------------------------- #
class TestUpdateDatabaseDocs:
    def test_update_single_page_content(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_database(isolated_db, pages={
            "page_overview": {
                "id": "page_overview", "title": "Overview",
                "content": "old", "filePaths": [], "importance": "high",
                "relatedPages": [],
            },
        })
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"page_id": "page_overview", "content": "new content"},
        )
        assert r.status_code == 200
        pages = r.json()["databases"][0]["pages"]
        assert pages["page_overview"]["content"] == "new content"

    def test_update_creates_missing_page(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"page_id": "page_new", "content": "created"},
        )
        assert r.status_code == 200
        pages = r.json()["databases"][0]["pages"]
        assert pages["page_new"]["content"] == "created"
        assert pages["page_new"]["title"] == "page_new"

    def test_update_pages_wholesale(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_database(isolated_db, pages={"old": {"content": "x"}})
        new_pages = {"page_a": {"id": "page_a", "title": "A", "content": "a"}}
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1", json={"pages": new_pages}
        )
        assert r.status_code == 200
        assert set(r.json()["databases"][0]["pages"]) == {"page_a"}

    def test_update_generated_docs(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"generated_docs": "# Edited docs"},
        )
        assert r.status_code == 200
        assert r.json()["databases"][0]["generated_docs"] == "# Edited docs"

    def test_update_missing_database_404(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/nope",
            json={"generated_docs": "x"},
        )
        assert r.status_code == 404

    def test_update_missing_product_404(self, isolated_db, monkeypatch):
        _disable_reindex(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/missing/databases/db_1",
            json={"generated_docs": "x"},
        )
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# PUT — metadata shape (product page settings form)
# --------------------------------------------------------------------------- #
class TestUpdateDatabaseMeta:
    def test_update_name(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/databases/db_1", json={"name": "Renamed"})
        assert r.status_code == 200
        assert r.json()["databases"][0]["name"] == "Renamed"

    def test_update_dsn_remasked(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"dsn": "postgresql://app:newpass@other:5432/other"},
        )
        assert r.status_code == 200
        db = r.json()["databases"][0]
        assert db["dsn_masked"] == "postgresql://app:***REDACTED***@other:5432/other"
        assert "newpass" not in r.text

    def test_update_dsn_empty_clears(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/databases/db_1", json={"dsn": ""})
        assert r.status_code == 200
        assert r.json()["databases"][0]["dsn_masked"] is None

    def test_update_dsn_passwordless_not_500(self, isolated_db):
        # Review #4: password-less DSN on the meta PUT must not trip the
        # leak guard either.
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"dsn": "postgres://app@host:5432/db"},
        )
        assert r.status_code == 200
        assert r.json()["databases"][0]["dsn_masked"] == "postgres://app@host:5432/db"

    def test_update_pin_valid(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        _seed_mcp_server(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"mcp_server_id": "mcp_1"},
        )
        assert r.status_code == 200
        assert r.json()["databases"][0]["mcp_server_id"] == "mcp_1"

    def test_update_pin_empty_clears(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db, mcp_server_id="mcp_1")
        _seed_mcp_server(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1", json={"mcp_server_id": ""}
        )
        assert r.status_code == 200
        assert r.json()["databases"][0]["mcp_server_id"] is None

    def test_update_pin_unknown_404(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"mcp_server_id": "mcp_ghost"},
        )
        assert r.status_code == 404

    def test_update_pin_unbound_400(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        _seed_mcp_server(isolated_db, bound=False)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"mcp_server_id": "mcp_1"},
        )
        assert r.status_code == 400

    def test_update_meta_missing_database_404(self, isolated_db):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/databases/nope", json={"name": "x"})
        assert r.status_code == 404

    def test_update_empty_body_is_noop_200(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put("/api/products/prod_1/databases/db_1", json={})
        assert r.status_code == 200
        assert r.json()["databases"][0]["name"] == "Main DB"


# --------------------------------------------------------------------------- #
# Verify (owner or admin)
# --------------------------------------------------------------------------- #
class TestVerifyDatabase:
    def test_owner_can_verify(self, isolated_db, admin_user):
        _seed_product(isolated_db, owner_id="user_admin1")
        _seed_database(isolated_db)
        admin_user.id = "user_admin1"
        app, client = _client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/prod_1/databases/db_1/verify")
        assert r.status_code == 200
        db = r.json()["databases"][0]
        assert db["verified"] is True
        assert db["verified_by"] == "user_admin1"
        assert db["verified_at"] is not None

    def test_admin_can_verify_any(self, isolated_db, admin_user):
        _seed_product(isolated_db, owner_id="someone_else")
        _seed_database(isolated_db)
        app, client = _client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/prod_1/databases/db_1/verify")
        assert r.status_code == 200
        assert r.json()["databases"][0]["verified"] is True

    def test_non_owner_403(self, isolated_db):
        _seed_product(isolated_db, owner_id="someone_else")
        _seed_database(isolated_db)
        app, client = _client_with_user(isolated_db, _regular_user())
        r = client.post("/api/products/prod_1/databases/db_1/verify")
        assert r.status_code == 403

    def test_verify_missing_product_404(self, isolated_db, admin_user):
        app, client = _client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/missing/databases/db_1/verify")
        assert r.status_code == 404

    def test_verify_missing_database_404(self, isolated_db, admin_user):
        _seed_product(isolated_db)
        app, client = _client_with_user(isolated_db, admin_user)
        r = client.post("/api/products/prod_1/databases/nope/verify")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Generate (202 + poll)
# --------------------------------------------------------------------------- #
class TestGenerateDatabaseDocs:
    def test_generate_returns_202_job_contract(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        calls = _stub_submit_job(monkeypatch)
        app, client = _make_client(isolated_db)

        r = client.post("/api/products/prod_1/databases/db_1/generate", json={})
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "queued"
        assert body["entity_type"] == "database"
        assert body["entity_id"] == "db_1"
        assert body["job_id"]

        assert len(calls) == 1
        assert calls[0]["entity_type"] == "database"
        assert calls[0]["entity_id"] == "db_1"
        assert calls[0]["product_id"] == "prod_1"
        assert calls[0]["language"] == "ru"

        # The status endpoint mirrors the codebases contract.
        s = client.get(
            "/api/products/prod_1/databases/db_1/generate/status",
            params={"job_id": body["job_id"]},
        )
        assert s.status_code == 200
        status = s.json()
        assert status["job_id"] == body["job_id"]
        assert status["status"] == "queued"
        assert status["indexing_status"] == "idle"
        # Progress block: initial snapshot for a queued job.
        assert status["progress"]["phase"] == "queued"
        assert status["progress"]["sections_done"] == 0
        assert set(status) == {
            "job_id", "status", "progress", "indexing_status",
            "indexing_message", "error", "created_at", "started_at",
            "finished_at", "docs_chars",
        }

    def test_generate_passes_model_and_language(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        calls = _stub_submit_job(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases/db_1/generate",
            json={"model": "qwen/test", "language": "en"},
        )
        assert r.status_code == 202
        assert calls[0]["model"] == "qwen/test"
        assert calls[0]["language"] == "en"

    def test_generate_missing_database_404(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _stub_submit_job(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/prod_1/databases/nope/generate", json={})
        assert r.status_code == 404

    def test_generate_missing_product_404(self, isolated_db, monkeypatch):
        _stub_submit_job(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post("/api/products/missing/databases/db_1/generate", json={})
        assert r.status_code == 404

    def test_status_unknown_job_404(self, isolated_db):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.get(
            "/api/products/prod_1/databases/db_1/generate/status",
            params={"job_id": "no-such-job"},
        )
        assert r.status_code == 404
        assert "Docgen job not found" in r.json()["detail"]

    def test_status_wrong_product_404(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        calls = _stub_submit_job(monkeypatch)
        app, client = _make_client(isolated_db)
        created = client.post(
            "/api/products/prod_1/databases/db_1/generate", json={}
        ).json()

        r = client.get(
            "/api/products/other/databases/db_1/generate/status",
            params={"job_id": created["job_id"]},
        )
        assert r.status_code == 404

    def test_status_wrong_entity_type_404(self, isolated_db, monkeypatch):
        """A codebase job id must not resolve through the databases status
        endpoint (entity scoping)."""
        from api.docgen.jobs import create_job as create_job_real

        _seed_product(isolated_db)
        _seed_database(isolated_db)
        _stub_submit_job(monkeypatch)
        job_id = create_job_real("prod_1", "codebase", "db_1")
        app, client = _make_client(isolated_db)
        r = client.get(
            "/api/products/prod_1/databases/db_1/generate/status",
            params={"job_id": job_id},
        )
        assert r.status_code == 404
