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
MASKED_DSN = "postgresql://***REDACTED***@db.internal:5432/prod"


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
            "postgresql://***REDACTED***@db/x"
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
            "postgresql://***REDACTED***@db:5432/prod"
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
        assert db["dsn_masked"] == "postgresql://***REDACTED***@other:5432/other"
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
        assert r.json()["databases"][0]["dsn_masked"] == "postgres://***REDACTED***@host:5432/db"

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
        # ``language`` is a deprecated no-op request field: when not sent, the
        # router forwards None and the WORKER resolves the effective language
        # from the admin ``generation.language`` setting at job start
        # (api.docgen.jobs._run_docgen_job_async).
        assert calls[0]["language"] is None

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


# --------------------------------------------------------------------------- #
# Preset flow (db_type: validate -> real MCP check -> server+binding+DB)
# --------------------------------------------------------------------------- #
PRESET_DSN = "postgresql://app:hunter2@db.internal:5432/prod"
ORACLE_PRESET_DSN = "app/orapw@db.internal:1521/XEPDB1"


def _preset_payload(dbid: str = "dbp_1", db_type: str = "postgresql",
                    dsn: str = PRESET_DSN, **overrides) -> dict:
    payload = {
        "id": dbid,
        "name": "Preset DB",
        "db_type": db_type,
        "dsn": dsn,
        "source": "manual",
    }
    payload.update(overrides)
    return payload


def _mock_preset_ok(monkeypatch, error: Optional[BaseException] = None):
    """Stub the REAL MCP connection check + launcher choice (hermetic).

    Returns the list of (preset_key, dsn) pairs the check received."""
    from api.mcp.presets import Launcher

    calls = []

    async def fake_check(spec, dsn):
        calls.append((spec.key, dsn))
        if error is not None:
            raise error
        return {"server_name": spec.server_name, "tools": [spec.probe_tool]}

    monkeypatch.setattr(databases_router_module, "check_preset_connection", fake_check)
    monkeypatch.setattr(
        databases_router_module,
        "choose_launcher",
        lambda spec, dsn: Launcher("baked", "dbhub", ("--transport", "stdio")),
    )
    return calls


class TestAddPresetDatabase:
    def test_preset_add_creates_server_binding_and_db(self, isolated_db, monkeypatch):
        from api.mcp.secrets import decrypt_secret_dict

        _seed_product(isolated_db)
        calls = _mock_preset_ok(monkeypatch)
        app, client = _make_client(isolated_db)

        r = client.post(
            "/api/products/prod_1/databases", json=_preset_payload()
        )
        assert r.status_code == 200, r.text
        # The check ran with the validated DSN.
        assert calls == [("postgresql", PRESET_DSN)]

        # Response: preset row, NO DSN anywhere (raw or masked).
        db = r.json()["databases"][0]
        assert db["db_type"] == "postgresql"
        assert db["dsn_masked"] is None
        assert db["source"] == "preset"
        assert "hunter2" not in r.text

        with isolated_db.SessionLocal() as s:
            row = s.get(DatabaseORM, "dbp_1")
            assert row is not None
            assert row.db_type == "postgresql"
            assert row.dsn_masked is None
            assert row.source == "preset"
            assert row.mcp_server_id

            server = s.get(McpServerORM, row.mcp_server_id)
            assert server is not None
            assert server.name == "preset-postgresql-dbp_1"
            assert server.preset_key == "postgresql"
            assert server.transport == "stdio"
            assert server.command == "dbhub"
            assert server.args == ["--transport", "stdio"]
            assert server.enabled is True
            assert server.status == "ok"
            assert server.status_checked_at is not None
            # The DSN lives ONLY in the Fernet-encrypted env ciphertext.
            assert "hunter2" not in (server.env or "")
            stored_env = decrypt_secret_dict(server.env)
            assert stored_env.get("DSN") == PRESET_DSN
            assert stored_env.get("READONLY") == "true"

            binding = (
                s.query(ProductMcpServerORM)
                .filter_by(product_id="prod_1", mcp_server_id=server.id)
                .one_or_none()
            )
            assert binding is not None and binding.enabled is True

    def test_preset_add_oracle(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _mock_preset_ok(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases",
            json=_preset_payload(db_type="oracle", dsn=ORACLE_PRESET_DSN),
        )
        assert r.status_code == 200, r.text
        db = r.json()["databases"][0]
        assert db["db_type"] == "oracle"
        assert db["dsn_masked"] is None
        assert "orapw" not in r.text

    def test_preset_add_unknown_type_400(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        calls = _mock_preset_ok(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases",
            json=_preset_payload(db_type="neo4j"),
        )
        assert r.status_code == 400
        assert "Unknown database type" in r.json()["detail"]
        assert calls == []
        with isolated_db.SessionLocal() as s:
            assert s.query(DatabaseORM).count() == 0

    def test_preset_add_invalid_dsn_400_no_rows(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        calls = _mock_preset_ok(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases",
            json=_preset_payload(dsn="mysql://app:pw@db:3306/x"),
        )
        assert r.status_code == 400
        assert calls == []  # validation rejects BEFORE the connection check
        with isolated_db.SessionLocal() as s:
            assert s.query(DatabaseORM).count() == 0
            assert s.query(McpServerORM).count() == 0

    def test_preset_add_connection_check_failed_400_no_rows(
        self, isolated_db, monkeypatch
    ):
        from api.mcp.presets import PresetConnectionError

        _seed_product(isolated_db)
        _mock_preset_ok(
            monkeypatch,
            error=PresetConnectionError("Connection check failed (RuntimeError: boom)"),
        )
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases", json=_preset_payload()
        )
        assert r.status_code == 400
        assert "Connection check failed" in r.json()["detail"]
        assert "hunter2" not in r.text
        with isolated_db.SessionLocal() as s:
            assert s.query(DatabaseORM).count() == 0
            assert s.query(McpServerORM).count() == 0
            assert s.query(ProductMcpServerORM).count() == 0

    def test_preset_add_missing_product_404(self, isolated_db, monkeypatch):
        _mock_preset_ok(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/missing/databases", json=_preset_payload()
        )
        assert r.status_code == 404

    def test_preset_repost_same_id_replaces_server(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _mock_preset_ok(monkeypatch)
        app, client = _make_client(isolated_db)
        r1 = client.post(
            "/api/products/prod_1/databases", json=_preset_payload()
        )
        assert r1.status_code == 200
        r2 = client.post(
            "/api/products/prod_1/databases",
            json=_preset_payload(dsn="postgresql://app:newpw@db.internal:5432/prod"),
        )
        assert r2.status_code == 200, r2.text
        body = r2.json()
        assert len(body["databases"]) == 1
        assert "newpw" not in r2.text
        with isolated_db.SessionLocal() as s:
            assert s.query(DatabaseORM).filter_by(id="dbp_1").count() == 1
            # Exactly one dedicated server row remains (the old one dropped).
            servers = (
                s.query(McpServerORM).filter_by(preset_key="postgresql").all()
            )
            assert len(servers) == 1
            bindings = (
                s.query(ProductMcpServerORM)
                .filter_by(mcp_server_id=servers[0].id)
                .all()
            )
            assert len(bindings) == 1

    def test_legacy_add_skips_connection_check(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        calls = _mock_preset_ok(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases", json=_database_payload()
        )
        assert r.status_code == 200
        assert calls == []  # the legacy (no db_type) flow never spawns MCP


class TestPresetDatabaseUpdates:
    def _add_preset(self, isolated_db, monkeypatch, client):
        _mock_preset_ok(monkeypatch)
        r = client.post(
            "/api/products/prod_1/databases", json=_preset_payload()
        )
        assert r.status_code == 200, r.text

    def test_put_dsn_rejected_400(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        self._add_preset(isolated_db, monkeypatch, client)
        r = client.put(
            "/api/products/prod_1/databases/dbp_1",
            json={"dsn": "postgresql://x:y@h/db"},
        )
        assert r.status_code == 400
        assert "immutable" in r.json()["detail"]

    def test_put_mcp_pin_rejected_400(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        self._add_preset(isolated_db, monkeypatch, client)
        r = client.put(
            "/api/products/prod_1/databases/dbp_1",
            json={"mcp_server_id": "mcp_9"},
        )
        assert r.status_code == 400
        assert "immutable" in r.json()["detail"]

    def test_put_name_still_allowed(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db)
        self._add_preset(isolated_db, monkeypatch, client)
        r = client.put(
            "/api/products/prod_1/databases/dbp_1",
            json={"name": "Renamed"},
        )
        assert r.status_code == 200
        assert r.json()["databases"][0]["name"] == "Renamed"

    def test_put_doc_edit_still_allowed(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _disable_reindex(monkeypatch)
        app, client = _make_client(isolated_db)
        self._add_preset(isolated_db, monkeypatch, client)
        r = client.put(
            "/api/products/prod_1/databases/dbp_1",
            json={"generated_docs": "# hello"},
        )
        assert r.status_code == 200
        assert r.json()["databases"][0]["generated_docs"] == "# hello"

    def test_legacy_dsn_put_still_works(self, isolated_db, monkeypatch):
        """The immutability guard must not touch LEGACY (non-preset) rows."""
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.put(
            "/api/products/prod_1/databases/db_1",
            json={"dsn": "postgresql://app:next@db.internal:5432/prod"},
        )
        assert r.status_code == 200


class TestPresetDatabaseDelete:
    def test_delete_drops_dedicated_server_and_binding(self, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        _mock_preset_ok(monkeypatch)
        app, client = _make_client(isolated_db)
        r = client.post(
            "/api/products/prod_1/databases", json=_preset_payload()
        )
        assert r.status_code == 200
        with isolated_db.SessionLocal() as s:
            server_id = s.get(DatabaseORM, "dbp_1").mcp_server_id

        r = client.delete("/api/products/prod_1/databases/dbp_1")
        assert r.status_code == 200
        assert r.json()["databases"] == []
        with isolated_db.SessionLocal() as s:
            assert s.get(DatabaseORM, "dbp_1") is None
            assert s.get(McpServerORM, server_id) is None
            assert (
                s.query(ProductMcpServerORM)
                .filter_by(mcp_server_id=server_id)
                .count()
                == 0
            )

    def test_delete_legacy_keeps_registry_servers(self, isolated_db):
        """Legacy rows: DELETE must NOT delete registry MCP server rows."""
        _seed_product(isolated_db)
        _seed_database(isolated_db)
        _seed_mcp_server(isolated_db)
        app, client = _make_client(isolated_db)
        r = client.delete("/api/products/prod_1/databases/db_1")
        assert r.status_code == 200
        with isolated_db.SessionLocal() as s:
            assert s.get(McpServerORM, "mcp_1") is not None
