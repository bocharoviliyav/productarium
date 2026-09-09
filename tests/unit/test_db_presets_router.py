"""Unit tests for ``api.routers.db_presets`` — the ``GET /api/db-presets``
catalog endpoint that powers the add-database type selector.

Hermetic: static payload from the preset registry, no network, no secrets.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.mcp.presets import presets_public_view
from api.routers import db_presets as db_presets_router
from tests.conftest import build_test_client

EXPECTED_KEYS = {"postgresql", "mysql", "mariadb", "sqlserver", "sqlite", "oracle"}


class TestListDbPresets:
    def test_catalog_matches_registry(self, isolated_db):
        app, client = build_test_client(isolated_db, [db_presets_router])
        r = client.get("/api/db-presets")
        assert r.status_code == 200
        assert r.json() == presets_public_view()

    def test_catalog_shape(self, isolated_db):
        app, client = build_test_client(isolated_db, [db_presets_router])
        body = client.get("/api/db-presets").json()
        assert {e["key"] for e in body} == EXPECTED_KEYS
        for entry in body:
            assert entry["label"]
            assert entry["dsn_example"]
            assert entry["dsn_hint"]
            assert set(entry["server"]) == {
                "name", "homepage", "license", "license_notice",
            }
            assert entry["server"]["license"] == "MIT"

    def test_catalog_carries_no_ciphertext(self, isolated_db):
        """The catalog is a static shape description — it never carries
        encrypted payloads (or any user-supplied DSN: it has none)."""
        app, client = build_test_client(isolated_db, [db_presets_router])
        text = client.get("/api/db-presets").text
        assert "gAAAA" not in text  # no ciphertext leaks

    def test_requires_auth(self, isolated_db):
        app, client = build_test_client(
            isolated_db, [db_presets_router],
            auth_none=False, default_admin_auth=False,
        )
        r = client.get("/api/db-presets")
        assert r.status_code == 401
