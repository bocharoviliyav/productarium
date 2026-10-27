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


def _pages(n, marker="v"):
    return {
        f"page_{i:03d}": {"id": f"page_{i:03d}", "title": f"P{i}", "content": f"{marker} {i}"}
        for i in range(n)
    }


class TestDeltaVersions:
    def test_page_scoped_regen_stores_single_override(self, isolated_db):
        """SPEC-012: regenerating 1 page of a 200-page document must NOT copy
        the other 199 pages into the new snapshot row."""
        _seed(isolated_db, generated_docs="# V1", pages=_pages(200))
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="generate")  # v1 full
            changed = dict(cb.pages)
            changed["page_007"] = {**changed["page_007"], "content": "regenerated"}
            cb.pages = changed
            assert dvr.append_version(
                s, "codebase", cb, source="generate", force_pages=["page_007"],
            ) == 2
            s.commit()
        with isolated_db.SessionLocal() as s:
            stored = dvr._raw_version(s, "codebase", "art_1", 2)
            assert stored.base_version == 1
            assert list(stored.pages) == ["page_007"]  # ONLY the override
            assert stored.generated_docs is None
            # Materialized read resolves the full 200-page document.
            mat = dvr.get_version(s, "codebase", "art_1", 2)
            assert len(mat.pages) == 200
            assert mat.pages["page_007"]["content"] == "regenerated"
            assert mat.pages["page_000"]["content"] == "v 0"
            assert mat.generated_docs == "# V1"  # blob resolves through the base

    def test_chain_materialization_base_delta_delta(self, isolated_db):
        _seed(isolated_db, generated_docs="# V1", pages=_pages(2))
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="generate")  # v1 full
            cb.pages = {**cb.pages, "page_000": {**cb.pages["page_000"], "content": "e1"}}
            dvr.append_version(s, "codebase", cb, source="edit")      # v2 delta
            cb.pages = {**cb.pages, "page_001": {**cb.pages["page_001"], "content": "e2"}}
            dvr.append_version(s, "codebase", cb, source="edit")      # v3 delta
            s.commit()
        with isolated_db.SessionLocal() as s:
            v2 = dvr._raw_version(s, "codebase", "art_1", 2)
            v3 = dvr._raw_version(s, "codebase", "art_1", 3)
            assert (v2.base_version, list(v2.pages)) == (1, ["page_000"])
            assert (v3.base_version, list(v3.pages)) == (2, ["page_001"])
            mat = dvr.get_version(s, "codebase", "art_1", 3)
            assert mat.pages["page_000"]["content"] == "e1"
            assert mat.pages["page_001"]["content"] == "e2"

    def test_chain_flattens_past_50_deltas(self, isolated_db):
        """Depth cap: at most 50 deltas per chain; the 51st consecutive page
        edit lands as a full snapshot again."""
        _seed(isolated_db, generated_docs="# V1", pages=_pages(1))
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="generate")  # v1 full
            last = 1
            for i in range(51):  # v2..v51 deltas, v52 must flatten
                cb.pages = {"page_000": {"content": f"edit {i}"}}
                last = dvr.append_version(s, "codebase", cb, source="edit")
            s.commit()
        assert last == 52
        with isolated_db.SessionLocal() as s:
            assert dvr._raw_version(s, "codebase", "art_1", 51).base_version == 50
            v52 = dvr._raw_version(s, "codebase", "art_1", 52)
            assert v52.base_version is None
            assert v52.pages == {"page_000": {"content": "edit 50"}}

    def test_page_deletion_and_blob_edit_stay_full(self, isolated_db):
        """Deltas cannot express page deletions; a diff with no changed pages
        (blob-only edit) is not worth a delta — both stay full snapshots."""
        _seed(isolated_db, generated_docs="# V1",
              pages={"a": {"content": "1"}, "b": {"content": "1"}})
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="generate")  # v1 full
            cb.pages = {"a": {"content": "1"}}  # page b deleted
            dvr.append_version(s, "codebase", cb, source="edit")      # v2 full
            cb.generated_docs = "# V2"         # pages untouched (blob-only)
            dvr.append_version(s, "codebase", cb, source="edit")      # v3 full
            s.commit()
        with isolated_db.SessionLocal() as s:
            for v in (2, 3):
                assert dvr._raw_version(s, "codebase", "art_1", v).base_version is None
            assert dvr._raw_version(s, "codebase", "art_1", 2).pages == {
                "a": {"content": "1"},
            }

    def test_restore_appends_delta_on_restored_base(self, isolated_db):
        """SPEC-012 rollback: the new version is a delta based on the restored
        target — overrides = pages the rollback changes vs the outgoing state;
        page deletions resolve naturally through the base."""
        _seed(isolated_db, generated_docs="# V1",
              pages={"a": {"content": "1"}, "b": {"content": "1"}})
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="generate")  # v1 full
            cb.pages = {"a": {"content": "2"}, "b": {"content": "1"},
                        "c": {"content": "9"}}
            dvr.append_version(s, "codebase", cb, source="edit")      # v2 delta {a,c}
            assert dvr.restore_version(s, "codebase", cb, 1) == "# V1"
            s.commit()
        with isolated_db.SessionLocal() as s:
            rows = {r.version: r for r in dvr.list_versions(s, "codebase", "art_1")}
            assert (rows[2].base_version, sorted(rows[2].pages)) == (1, ["a", "c"])
            assert (rows[3].base_version, rows[3].pages) == (1, {"a": {"content": "1"}})
            mat = dvr.get_version(s, "codebase", "art_1", 3)
            assert mat.pages == {"a": {"content": "1"}, "b": {"content": "1"}}  # c gone
            assert s.get(CodebaseORM, "art_1").pages == mat.pages


class TestDeleteVersion:
    def _seed_versions(self, isolated_db):
        _seed(isolated_db, generated_docs="# V1", pages={"a": {"content": "1"}})
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="generate")  # v1 full
            cb.generated_docs = "# V2"
            dvr.append_version(s, "codebase", cb, source="generate")  # v2 full
            cb.pages = {"a": {"content": "3"}}
            dvr.append_version(s, "codebase", cb, source="edit")      # v3 delta on v2
            s.commit()

    def test_refusal_reasons(self, isolated_db):
        self._seed_versions(isolated_db)
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert dvr.delete_version(s, "codebase", cb, 99) == "not_found"
            assert dvr.delete_version(s, "codebase", cb, 3) == "current"
            assert dvr.delete_version(s, "codebase", cb, 2) == "base"  # v3 builds on v2
            assert [r.version for r in dvr.list_versions(s, "codebase", "art_1")] == [3, 2, 1]

    def test_only_version_refused_when_pointer_unset(self, isolated_db):
        """Legacy state: a lone snapshot with no current_version pointer."""
        _seed(isolated_db, generated_docs="# V1")
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            dvr.append_version(s, "codebase", cb, source="edit")
            cb.current_version = None
            s.commit()
        with isolated_db.SessionLocal() as s:
            assert dvr.delete_version(s, "codebase", s.get(CodebaseORM, "art_1"), 1) == "single"

    def test_delete_old_version_removes_only_the_row(self, isolated_db):
        self._seed_versions(isolated_db)
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert dvr.delete_version(s, "codebase", cb, 1) is None
            s.commit()
        with isolated_db.SessionLocal() as s:
            cb = s.get(CodebaseORM, "art_1")
            assert [r.version for r in dvr.list_versions(s, "codebase", "art_1")] == [3, 2]
            # Artifact docs, current pointer and the remaining chain survive.
            assert cb.current_version == 3
            assert cb.generated_docs == "# V2"
            assert cb.pages == {"a": {"content": "3"}}
            assert dvr.get_version(s, "codebase", "art_1", 3).pages == {"a": {"content": "3"}}
