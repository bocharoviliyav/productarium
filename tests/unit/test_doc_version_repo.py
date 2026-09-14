"""Unit tests for ``api.repositories.doc_version_repo`` (Vault KV-v2 snapshots).

Hermetic: real ORM rows over the isolated in-memory SQLite (``isolated_db``).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import CodebaseORM, ProductORM, SpecORM
from api.repositories import doc_version_repo as dvr


def _seed(db_mod, **codebase_overrides):
    with db_mod.SessionLocal() as s:
        s.add(ProductORM(id="prod_1", name="Acme"))
        s.flush()
        s.add(CodebaseORM(
            id="art_1", product_id="prod_1", name="svc", source="manual",
            **codebase_overrides,
        ))
        s.commit()


class TestAppendVersion:
    def test_append_increments_and_points_current(self, isolated_db):
        _seed(isolated_db, generated_docs="# V1")
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            v1 = dvr.append_version(s, "codebase", cb, source="edit")
            cb.generated_docs = "# V2"
            v2 = dvr.append_version(
                s, "codebase", cb, source="generate", model="m", job_id="job_1",
            )
            s.commit()
        assert (v1, v2) == (1, 2)

        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert cb.current_version == 2
            # Snapshots are immutable: v1 keeps the payload it was taken with.
            rows = dvr.list_versions(s, "codebase", "art_1")
            assert [r.version for r in rows] == [2, 1]  # newest first
            assert rows[1].generated_docs == "# V1"
            assert rows[1].source == "edit"
            assert rows[0].source == "generate"
            assert rows[0].model == "m"
            assert rows[0].job_id == "job_1"
            assert dvr.get_version(s, "codebase", "art_1", 1).pages is None

    def test_spec_snapshots_content_only(self, isolated_db):
        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_1", name="Acme"))
            s.flush()
            spec = SpecORM(id="spec_1", product_id="prod_1", name="api",
                           kind="openapi", content="# SPEC")
            s.add(spec)
            s.flush()
            assert dvr.append_version(s, "spec", spec, source="edit") == 1
            s.commit()
        with isolated_db.SessionLocal() as s:
            row = dvr.get_version(s, "spec", "spec_1", 1)
            assert row.content == "# SPEC"
            assert row.generated_docs is None and row.pages is None


class TestEnsureBaselineVersion:
    def test_bootstraps_legacy_docs_as_v1(self, isolated_db):
        _seed(isolated_db, generated_docs="# legacy")
        with isolated_db.SessionLocal() as s:
            assert dvr.ensure_baseline_version(s, "codebase", s.get(CodebaseORM, "art_1")) == 1
            s.commit()
        with isolated_db.SessionLocal() as s:
            rows = dvr.list_versions(s, "codebase", "art_1")
            assert len(rows) == 1
            assert rows[0].source == "baseline"
            assert rows[0].generated_docs == "# legacy"

    def test_noop_when_versions_exist(self, isolated_db):
        _seed(isolated_db, generated_docs="# legacy")
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="edit")
            assert dvr.ensure_baseline_version(s, "codebase", cb) is None
            s.commit()
        with isolated_db.SessionLocal() as s:
            assert len(dvr.list_versions(s, "codebase", "art_1")) == 1

    def test_skips_empty_artifact(self, isolated_db):
        _seed(isolated_db)
        with isolated_db.SessionLocal() as s:
            assert dvr.ensure_baseline_version(s, "codebase", s.get(CodebaseORM, "art_1")) is None
            s.commit()
        with isolated_db.SessionLocal() as s:
            assert dvr.list_versions(s, "codebase", "art_1") == []

    def test_baseline_plus_edit_in_one_transaction(self, isolated_db):
        """Flush regression: autoflush=False sessions must still see the
        baseline row in max(version) when the edit appends in the SAME tx."""
        _seed(isolated_db, generated_docs="# legacy")
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert dvr.ensure_baseline_version(s, "codebase", cb) == 1
            cb.generated_docs = "# edited"
            assert dvr.append_version(s, "codebase", cb, source="edit") == 2
            s.commit()
        with isolated_db.SessionLocal() as s:
            rows = dvr.list_versions(s, "codebase", "art_1")
            assert [r.version for r in rows] == [2, 1]
            assert rows[1].source == "baseline"
            assert rows[1].generated_docs == "# legacy"
            assert s.get(CodebaseORM, "art_1").current_version == 2


class TestRestoreVersion:
    def test_restore_appends_rollback_and_returns_text(self, isolated_db):
        _seed(isolated_db, generated_docs="# V1", pages={"page_overview": {}})
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="edit", model="m0")
            cb.generated_docs, cb.pages = "# V2", None
            dvr.append_version(s, "codebase", cb, source="generate", model="m1")
            s.commit()
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            indexed = dvr.restore_version(s, "codebase", cb, 1)
            s.commit()
        assert indexed == "# V1"
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            # Artifact rolled back; the rollback itself is the new v3.
            assert cb.generated_docs == "# V1"
            assert cb.pages == {"page_overview": {}}
            assert cb.current_version == 3
            rows = dvr.list_versions(s, "codebase", "art_1")
            assert [r.source for r in rows] == ["rollback", "generate", "edit"]
            assert rows[0].model == "m0"  # inherited from the restored row
            assert rows[1].generated_docs == "# V2"  # history untouched

    def test_restore_unknown_version_returns_none(self, isolated_db):
        _seed(isolated_db, generated_docs="# V1")
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert dvr.restore_version(s, "codebase", cb, 99) is None
            s.commit()

    def test_spec_restore_writes_content(self, isolated_db):
        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_1", name="Acme"))
            s.flush()
            spec = SpecORM(id="spec_1", product_id="prod_1", name="api",
                           kind="openapi", content="# S1")
            s.add(spec)
            s.flush()
            dvr.append_version(s, "spec", spec, source="edit")
            spec.content = "# S2"
            dvr.append_version(s, "spec", spec, source="edit")
            assert dvr.restore_version(s, "spec", spec, 1) == "# S1"
            s.commit()
        with isolated_db.SessionLocal() as s:
            spec = s.get(SpecORM, "spec_1")
            assert spec.content == "# S1"
            assert spec.current_version == 3


class TestRestoreEntityFromCurrentVersion:
    def test_restores_row_payload(self, isolated_db):
        _seed(isolated_db, generated_docs="# v1", pages={"page_overview": {}})
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="generate")
            s.commit()
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            # Simulate a mid-run checkpoint that dirtied the row.
            cb.generated_docs, cb.pages = "# partial", None
            assert dvr.restore_entity_from_current_version(s, "codebase", cb) is True
            s.commit()
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert cb.generated_docs == "# v1"
            assert cb.pages == {"page_overview": {}}

    def test_clears_docs_when_no_versions(self, isolated_db):
        _seed(isolated_db, generated_docs="# junk first-ever run")
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert dvr.restore_entity_from_current_version(s, "codebase", cb) is False
            s.commit()
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert cb.generated_docs is None and cb.pages is None

    def test_spec_clear_is_noop(self, isolated_db):
        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_1", name="Acme"))
            s.flush()
            spec = SpecORM(id="spec_1", product_id="prod_1", name="api",
                           kind="openapi", content="# draft")
            s.add(spec)
            s.flush()
            assert dvr.restore_entity_from_current_version(s, "spec", spec) is False
            assert spec.content == "# draft"  # spec rewrites content only at the final node
            s.commit()
