"""Unit tests for ``api.routers.hld`` and the ``hlds`` segment routes.

Covers: POST /hld/generate (202 + auto-create + dedup reuse), generate
status, page content (current + archived version), versions list/detail/
restore, verify (entity + page, owner/admin only) and the light payload.
Hermetic: ``submit_job`` stubbed on the docgen router module (no worker).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.jobs as jobs_mod
from api.models import CodebaseORM, HldORM, ProductORM, UserORM
from api.repositories import doc_version_repo as dvr
from api.routers import docgen as docgen_router_module
from api.routers import doc_versions as doc_versions_router_module
from api.routers import hld as hld_router_module
from tests.conftest import build_test_client


@pytest.fixture(autouse=True)
def _clear_registry():
    jobs_mod._docgen_jobs.clear()
    jobs_mod._ENTITY_LOCKS.clear()
    from api.utils.rate_limit import reset_rate_limits

    reset_rate_limits()
    yield
    jobs_mod._docgen_jobs.clear()
    jobs_mod._ENTITY_LOCKS.clear()


def _seed(db_mod, *, with_hld=True):
    """Product + codebase (context source) + optionally the HLD row with v1 pages."""
    with db_mod.SessionLocal() as s:
        s.add(ProductORM(id="prod_1", name="Acme"))
        s.flush()
        s.add(CodebaseORM(
            id="cb_1", product_id="prod_1", name="svc",
            generated_docs="codebase docs",
        ))
        if with_hld:
            hld = HldORM(
                id="hld_prod_1", product_id="prod_1",
                generated_docs="## Обзор продукта\n\nv1 body",
                pages={"hld_overview": {
                    "id": "hld_overview", "title": "Обзор продукта",
                    "content": "v1 body",
                }},
            )
            s.add(hld)
        s.commit()


def _client(db_mod, monkeypatch):
    monkeypatch.setattr(docgen_router_module, "submit_job", MagicMock())
    app, client = build_test_client(
        db_mod,
        [hld_router_module, docgen_router_module, doc_versions_router_module],
    )
    return app, client


class TestHldGenerate:
    def test_generate_auto_creates_row_and_returns_202(self, isolated_db, monkeypatch):
        _seed(isolated_db, with_hld=False)
        _, client = _client(isolated_db, monkeypatch)

        r = client.post("/api/products/prod_1/hld/generate", json={})
        assert r.status_code == 202
        body = r.json()
        assert body["entity_type"] == "hld"
        assert body["entity_id"] == "hld_prod_1"
        assert body["reused"] is False

        with isolated_db.SessionLocal() as s:  # row auto-created BEFORE the job
            assert s.get(HldORM, "hld_prod_1") is not None

    def test_repeat_post_reuses_inflight_job(self, isolated_db, monkeypatch):
        _seed(isolated_db, with_hld=False)
        _, client = _client(isolated_db, monkeypatch)

        b1 = client.post("/api/products/prod_1/hld/generate", json={}).json()
        b2 = client.post("/api/products/prod_1/hld/generate", json={}).json()
        assert b2["job_id"] == b1["job_id"]
        assert b2["reused"] is True
        assert docgen_router_module.submit_job.call_count == 1

    def test_status_endpoint(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        _, client = _client(isolated_db, monkeypatch)
        b = client.post("/api/products/prod_1/hld/generate", json={}).json()

        r = client.get(
            f"/api/products/prod_1/hld/hld_prod_1/generate/status?job_id={b['job_id']}"
        )
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        # wrong job id -> 404
        assert client.get(
            "/api/products/prod_1/hld/hld_prod_1/generate/status?job_id=nope"
        ).status_code == 404


class TestHldPagesAndVersions:
    def _seed_with_versions(self, db_mod):
        _seed(db_mod)
        with db_mod.SessionLocal() as s:
            hld = s.get(HldORM, "hld_prod_1")
            dvr.append_version(s, "hld", hld, source="edit")  # v1
            hld.pages = {"hld_overview": {
                "id": "hld_overview", "title": "Обзор продукта",
                "content": "v2 body",
            }}
            hld.generated_docs = "## Обзор продукта\n\nv2 body"
            dvr.append_version(s, "hld", hld, source="edit")  # v2
            s.commit()

    def test_page_content_current_and_archive(self, isolated_db, monkeypatch):
        self._seed_with_versions(isolated_db)
        _, client = _client(isolated_db, monkeypatch)
        base = "/api/products/prod_1/hlds/hld_prod_1/pages/hld_overview"

        r = client.get(base)
        assert r.status_code == 200
        assert r.json() == {
            "page": {"id": "hld_overview", "title": "Обзор продукта",
                     "content": "v2 body"},
            "source": "current",
        }
        r = client.get(f"{base}?version=1")
        assert r.status_code == 200
        assert r.json()["source"] == "archive"
        assert r.json()["page"]["content"] == "v1 body"

    def test_singular_segment_alias(self, isolated_db, monkeypatch):
        """The UI builds sub-resource URLs with entityPath("hld") — the
        singular segment must work on cancel, page-content and versions."""
        self._seed_with_versions(isolated_db)
        _, client = _client(isolated_db, monkeypatch)

        job = client.post("/api/products/prod_1/hld/generate", json={}).json()
        assert client.post(
            "/api/products/prod_1/hld/hld_prod_1/generate/cancel"
        ).json() == {"cancelled": True, "job_id": job["job_id"]}

        page = client.get("/api/products/prod_1/hld/hld_prod_1/pages/hld_overview")
        assert page.status_code == 200
        assert page.json()["page"]["content"] == "v2 body"

        versions = client.get("/api/products/prod_1/hld/hld_prod_1/versions")
        assert versions.status_code == 200
        assert versions.json()["entity_type"] == "hld"

    def test_archive_page_is_product_scoped(self, isolated_db, monkeypatch):
        """?version=N must not leak another product's rows: version lookups
        key on (entity_type, entity_id) alone, so a DIFFERENT product's
        hld_{product_id} id must 404 even for an admin with ro access."""
        self._seed_with_versions(isolated_db)
        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_2", name="Other"))
            s.commit()
        _, client = _client(isolated_db, monkeypatch)

        r = client.get(
            "/api/products/prod_2/hlds/hld_prod_1/pages/hld_overview?version=1"
        )
        assert r.status_code == 404

    def test_versions_list_detail_restore(self, isolated_db, monkeypatch):
        self._seed_with_versions(isolated_db)
        _, client = _client(isolated_db, monkeypatch)
        base = "/api/products/prod_1/hlds/hld_prod_1"

        lst = client.get(f"{base}/versions").json()
        assert lst["entity_type"] == "hld"
        assert lst["current_version"] == 2
        assert [v["version"] for v in lst["versions"]] == [2, 1]

        detail = client.get(f"{base}/versions/1").json()
        assert detail["generated_docs"] == "## Обзор продукта\n\nv1 body"
        assert detail["pages"]["hld_overview"]["content"] == "v1 body"

        r = client.post(f"{base}/versions/1/restore")
        assert r.status_code == 200
        product = r.json()
        assert product["hld"]["pages"]["hld_overview"]["content"] == "v1 body"
        assert product["hld"]["current_version"] == 3  # rollback appended


class TestHldVerify:
    def test_verify_entity_light_payload(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        _, client = _client(isolated_db, monkeypatch)

        r = client.post("/api/products/prod_1/hld/hld_prod_1/verify?light=true")
        assert r.status_code == 200
        hld = r.json()["hld"]
        assert hld["verified"] is True
        assert hld["verified_by"] == "user_admin1"
        # light: page meta only, no bodies, no blob
        assert "content" not in hld["pages"]["hld_overview"]
        assert hld["generated_docs"] is None

    def test_verify_page_flags(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        _, client = _client(isolated_db, monkeypatch)

        r = client.post(
            "/api/products/prod_1/hld/hld_prod_1/pages/hld_overview/verify"
        )
        assert r.status_code == 200
        page = r.json()["hld"]["pages"]["hld_overview"]
        assert page["verified"] is True
        assert page["content"] == "v1 body"  # non-light keeps bodies

        assert client.post(  # unknown page
            "/api/products/prod_1/hld/hld_prod_1/pages/ghost/verify"
        ).status_code == 404
        assert client.post(  # unknown hld
            "/api/products/prod_1/hld/ghost/verify"
        ).status_code == 404

    def test_non_owner_forbidden(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        app, client = _client(isolated_db, monkeypatch)
        regular = UserORM(
            id="user_u2", username="u2", role="user", provider="local",
        )
        app.dependency_overrides[hld_router_module.get_current_user] = lambda: regular

        assert client.post(
            "/api/products/prod_1/hld/hld_prod_1/verify"
        ).status_code == 403
        assert client.post(
            "/api/products/prod_1/hld/hld_prod_1/pages/hld_overview/verify"
        ).status_code == 403
