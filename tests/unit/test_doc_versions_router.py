"""Unit tests for ``api.routers.doc_versions`` (list / detail / restore).

Hermetic: isolated in-memory SQLite + ``build_test_client`` (admin override);
``_reindex`` is patched to a recorder so no memory backend is touched.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import CodebaseORM, ProductORM, SpecORM
from api.repositories import doc_version_repo as dvr
from api.routers import doc_versions as doc_versions_router
from tests.conftest import build_test_client


def _seed(db_mod):
    with db_mod.SessionLocal() as s:
        s.add(ProductORM(id="prod_1", name="Acme"))
        s.flush()
        s.add(CodebaseORM(
            id="art_1", product_id="prod_1", name="svc", source="manual",
            generated_docs="# V1", pages={"page_overview": {"content": "v1"}},
        ))
        s.add(SpecORM(id="spec_1", product_id="prod_1", name="api",
                      kind="openapi", content="# SPEC V1"))
        s.flush()
        cb = s.get(CodebaseORM, "art_1")
        dvr.append_version(s, "codebase", cb, source="edit")
        cb.generated_docs = "# V2"
        dvr.append_version(s, "codebase", cb, source="generate", model="m1")
        spec = s.get(SpecORM, "spec_1")
        dvr.append_version(s, "spec", spec, source="edit")
        spec.content = "# SPEC V2"
        dvr.append_version(s, "spec", spec, source="edit")
        s.commit()


def _client(db_mod):
    return build_test_client(db_mod, [doc_versions_router], auth_none=True)


class TestDocVersionsRouter:
    def test_list_desc_with_current_flag(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        r = client.get("/api/products/prod_1/codebases/art_1/versions")
        assert r.status_code == 200
        body = r.json()
        assert body["entity_type"] == "codebase"
        assert body["current_version"] == 2
        assert [v["version"] for v in body["versions"]] == [2, 1]
        assert [v["is_current"] for v in body["versions"]] == [True, False]
        assert body["versions"][0]["model"] == "m1"
        # No payload columns on the list view.
        assert "generated_docs" not in body["versions"][0]

    def test_detail_returns_snapshot_payload(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        r = client.get("/api/products/prod_1/codebases/art_1/versions/1")
        assert r.status_code == 200
        body = r.json()
        assert body["generated_docs"] == "# V1"
        assert body["pages"] == {"page_overview": {"content": "v1"}}
        assert body["is_current"] is False
        assert client.get(
            "/api/products/prod_1/codebases/art_1/versions/99"
        ).status_code == 404

    def test_restore_appends_rollback_and_reindexes(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        reindexed = []
        monkeypatch.setattr(
            doc_versions_router, "_reindex",
            lambda *a, **kw: reindexed.append((a, kw)),
        )

        r = client.post("/api/products/prod_1/codebases/art_1/versions/1/restore")
        assert r.status_code == 200
        # Response is the full Product with the rolled-back artifact.
        cb = next(c for c in r.json()["codebases"] if c["id"] == "art_1")
        assert cb["generated_docs"] == "# V1"
        assert cb["current_version"] == 3

        # The restored text was re-indexed with the artifact as its source.
        assert reindexed == [(
            ("prod_1", "# V1", "art_1"), {"source_type": "codebase"},
        )]
        with isolated_db.SessionLocal() as s:
            rows = dvr.list_versions(s, "codebase", "art_1")
            assert [v.source for v in rows] == ["rollback", "generate", "edit"]

    def test_restore_unknown_version_404(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        assert client.post(
            "/api/products/prod_1/codebases/art_1/versions/99/restore"
        ).status_code == 404

    def test_spec_segment_restores_content(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        reindexed = []
        monkeypatch.setattr(
            doc_versions_router, "_reindex",
            lambda *a, **kw: reindexed.append((a, kw)),
        )

        r = client.get("/api/products/prod_1/specs/spec_1/versions")
        assert [v["version"] for v in r.json()["versions"]] == [2, 1]

        r = client.post("/api/products/prod_1/specs/spec_1/versions/1/restore")
        assert r.status_code == 200
        spec = next(s for s in r.json()["specs"] if s["id"] == "spec_1")
        assert spec["content"] == "# SPEC V1"
        assert reindexed[0][0][1] == "# SPEC V1"
        assert reindexed[0][1] == {"source_type": "spec"}

    def test_404s(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        assert client.get("/api/products/ghost/codebases/art_1/versions").status_code == 404
        assert client.get("/api/products/prod_1/links/art_1/versions").status_code == 404
        assert client.get("/api/products/prod_1/codebases/ghost/versions").status_code == 404
