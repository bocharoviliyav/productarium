"""Unit tests for the per-page content endpoint in ``api.routers.docgen``.

``GET /api/products/{pid}/{segment}/{id}/pages/{page_id}[?version=N]`` is the
companion of the ``?light=1`` product payloads: the viewer lazy-loads one
page body at a time (current entity or an archived doc version).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import CodebaseORM, DatabaseORM, ProductORM
from api.repositories import doc_version_repo as dvr
from api.routers import docgen as docgen_router
from tests.conftest import build_test_client

_OLD_PAGE = {"id": "page_overview", "title": "Overview", "content": "# v1 body"}
_NEW_PAGE = {"id": "page_overview", "title": "Overview", "content": "# v2 body"}


def _seed(db_mod):
    with db_mod.SessionLocal() as s:
        s.add(ProductORM(id="prod_1", name="Acme"))
        s.flush()
        cb = CodebaseORM(
            id="art_1", product_id="prod_1", name="svc", source="manual",
            generated_docs="# stored", pages={"page_overview": _OLD_PAGE},
        )
        s.add(cb)
        s.add(DatabaseORM(
            id="db_1", product_id="prod_1", name="DB", source="manual",
            pages={"page_tables": {"id": "page_tables", "title": "Tables", "content": "t"}},
        ))
        s.flush()
        dvr.append_version(s, "codebase", cb, source="edit")  # v1 = old body
        cb.pages = {"page_overview": _NEW_PAGE}
        s.commit()


def _client(db_mod):
    return build_test_client(db_mod, [docgen_router], auth_none=True)


class TestGetDocgenPage:
    def test_current_page_full_body(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        r = client.get("/api/products/prod_1/codebases/art_1/pages/page_overview")
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "current"
        assert body["page"]["content"] == "# v2 body"
        assert body["page"]["id"] == "page_overview"

    def test_archived_version_body(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        r = client.get(
            "/api/products/prod_1/codebases/art_1/pages/page_overview?version=1"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "archive"
        assert body["page"]["content"] == "# v1 body"

    def test_database_segment(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        r = client.get("/api/products/prod_1/databases/db_1/pages/page_tables")
        assert r.status_code == 200
        assert r.json()["page"]["content"] == "t"

    def test_specs_segment_rejected(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        assert client.get(
            "/api/products/prod_1/specs/art_1/pages/page_overview"
        ).status_code == 400

    def test_404s(self, isolated_db):
        _seed(isolated_db)
        _app, client = _client(isolated_db)
        base = "/api/products/prod_1/codebases/art_1/pages"
        assert client.get(f"{base}/nope").status_code == 404  # page
        assert client.get(f"{base}/page_overview?version=99").status_code == 404  # version
        assert client.get(  # entity
            "/api/products/prod_1/codebases/ghost/pages/page_overview"
        ).status_code == 404
        assert client.get(  # product
            "/api/products/ghost/codebases/art_1/pages/page_overview"
        ).status_code == 404
