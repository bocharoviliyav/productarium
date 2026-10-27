"""Unit tests for SPEC-010 change detection (``api.docgen.incremental``).

Hermetic: the git client, the database introspection surface and the LLM are
monkeypatched — only the detection/impact planning logic runs for real.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.incremental as inc
from api.models import CodebaseORM, DatabaseORM, SpecORM

# Arbitrary product stand-in (only ``id`` is read).
PRODUCT = type("P", (), {"id": "prod_1"})()

BASELINE = "a" * 40
TIP = "b" * 40


def _page(pid: str, title: str, files=(), fingerprint=None):
    prov = {"source_files": list(files)}
    if fingerprint is not None:
        prov["object_fingerprint"] = fingerprint
    return {"id": pid, "title": title, "content": "body", "provenance": prov}


def _cb(pages=None, source_state=None, repo_url="https://github.com/o/r"):
    return CodebaseORM(
        id="cb_1", product_id="prod_1", name="r", repo_type="github",
        repo_url=repo_url, source_state=source_state, pages=pages or {},
    )


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Kill the impact-expansion LLM call unless a test overrides it."""
    async def _none(prompt, model, **kw):
        return None
    monkeypatch.setattr(inc, "_llm_or_none", _none)
    monkeypatch.setattr(inc, "_resolve_docgen_model", lambda m: ("m", "http://x", "k"))


@pytest.fixture
def git_repo(monkeypatch, tmp_path):
    """Managed clone dir exists; tip/diff are injected via the dict."""
    import api.repositories.documents as documents_mod

    monkeypatch.setattr(documents_mod, "managed_repo_dir", lambda *a, **k: str(tmp_path))

    import api.clients.git as git_mod

    state = {"tip": None, "diff": None}

    def _tip(repo_dir, repo_url, repo_type, token):
        if isinstance(state["tip"], Exception):
            raise state["tip"]
        return state["tip"]

    def _diff(repo_dir, base, tip):
        if isinstance(state["diff"], Exception):
            raise state["diff"]
        return state["diff"]

    monkeypatch.setattr(git_mod, "fetch_remote_tip", _tip)
    monkeypatch.setattr(git_mod, "diff_commit_names", _diff)
    return state


# ============================================================================
# Codebase detection
# ============================================================================
class TestDetectCodebase:
    def test_no_repo_url_is_unknown(self):
        res = asyncio.run(inc.detect_changes(_cb(repo_url=None), PRODUCT, "codebase"))
        assert res["changed"] is None
        assert res["impacted_pages"] == []

    def test_missing_clone_is_unknown(self, monkeypatch, tmp_path):
        import api.repositories.documents as documents_mod

        monkeypatch.setattr(documents_mod, "managed_repo_dir", lambda *a, **k: None)
        res = asyncio.run(inc.detect_changes(
            _cb(source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert res["changed"] is None
        assert "полную генерацию" in res["reason"]

    def test_no_baseline_is_unknown(self, git_repo):
        res = asyncio.run(inc.detect_changes(_cb(), PRODUCT, "codebase"))
        assert res["changed"] is None

    def test_same_tip_no_changes(self, git_repo):
        git_repo["tip"] = BASELINE
        res = asyncio.run(inc.detect_changes(
            _cb(source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert res["changed"] is False
        assert res["changed_files"] == []
        assert res["impacted_pages"] == []

    def test_changed_files_map_to_pages_via_provenance(self, git_repo):
        git_repo["tip"] = TIP
        git_repo["diff"] = ["src/payments/api.py", "README.md"]
        pages = {
            "page_pay": _page("page_pay", "Payments", ["src/payments/api.py"]),
            "page_auth": _page("page_auth", "Auth", ["src/auth.py"]),
        }
        res = asyncio.run(inc.detect_changes(
            _cb(pages=pages, source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert res["changed"] is True
        assert res["changed_files"] == ["src/payments/api.py", "README.md"]
        assert [p["id"] for p in res["impacted_pages"]] == ["page_pay"]
        assert res["impacted_pages"][0]["new"] is False

    def test_diff_failure_forces_full_regen(self, git_repo):
        git_repo["tip"] = TIP
        git_repo["diff"] = ValueError("no merge base")
        res = asyncio.run(inc.detect_changes(
            _cb(source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert res["changed"] is True
        assert "changed_files" not in res
        assert res["impacted_pages"] == []

    def test_fetch_failure_is_unknown(self, git_repo):
        git_repo["tip"] = ValueError("network down")
        res = asyncio.run(inc.detect_changes(
            _cb(source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert res["changed"] is None


# ============================================================================
# LLM impact expansion (one call, add-only, valid candidates only)
# ============================================================================
class TestImpactExpansion:
    @pytest.fixture
    def pages(self):
        return {
            "page_pay": _page("page_pay", "Payments", ["src/payments/api.py"]),
            "page_overview": _page("page_overview", "Overview", ["README.md"]),
            "page_auth": _page("page_auth", "Auth", ["src/auth.py"]),
        }

    def test_llm_adds_only_valid_candidates(self, git_repo, monkeypatch, pages):
        git_repo["tip"] = TIP
        git_repo["diff"] = ["src/payments/api.py"]

        async def fake_llm(prompt, model, **kw):
            return '["page_overview", "page_auth", "bogus_id"]'

        monkeypatch.setattr(inc, "_llm_or_none", fake_llm)
        res = asyncio.run(inc.detect_changes(
            _cb(pages=pages, source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        ids = [p["id"] for p in res["impacted_pages"]]
        assert ids == ["page_auth", "page_overview", "page_pay"]  # sorted; bogus filtered

    def test_llm_failure_keeps_deterministic_set(self, git_repo, pages):
        git_repo["tip"] = TIP
        git_repo["diff"] = ["src/payments/api.py"]
        res = asyncio.run(inc.detect_changes(
            _cb(pages=pages, source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert [p["id"] for p in res["impacted_pages"]] == ["page_pay"]

    def test_prompt_keeps_final_instruction_within_cap(self, git_repo, monkeypatch):
        """Q-5: on a huge impact plan the per-section caps keep the final
        JSON instruction inside the prompt — a whole-prompt cap would cut
        the tail and silently degrade the LLM answer to prose."""
        git_repo["tip"] = TIP
        git_repo["diff"] = [
            f"src/deeply/nested/package_{i}/very_long_module_name.py"
            for i in range(200)
        ]
        pages = {
            f"page_{i:03d}": _page(
                f"page_{i:03d}", f"Раздел документации номер {i}",
                [f"src/pkg_{j}/very_long_source_file_name.py" for j in range(4)],
            )
            for i in range(300)
        }
        captured = {}

        async def fake_llm(prompt, model, **kw):
            captured["prompt"] = prompt
            return "[]"

        monkeypatch.setattr(inc, "_llm_or_none", fake_llm)
        res = asyncio.run(inc.detect_changes(
            _cb(pages=pages, source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert res["changed"] is True
        assert len(captured["prompt"]) <= inc._IMPACT_PROMPT_CAP
        assert "Ответь ТОЛЬКО JSON-массивом" in captured["prompt"]


# ============================================================================
# Git subprocess timeouts (Q-3): a hung remote must fail the detect job
# (changed=None), never hold the entity lock forever.
# ============================================================================
class TestGitDetectTimeout:
    @staticmethod
    def _hanging_git(monkeypatch):
        import subprocess

        import api.clients.git as git_mod

        def hang(cmd, *a, **kw):
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout") or 0)

        monkeypatch.setattr(git_mod.subprocess, "run", hang)
        return git_mod

    def test_git_detect_key_registered(self, monkeypatch):
        from api.config.timeout import resolve_timeout

        monkeypatch.delenv("GIT_DETECT_TIMEOUT_SECONDS", raising=False)
        assert resolve_timeout("git_detect") == 120.0

    def test_fetch_timeout_raises_value_error(self, monkeypatch, tmp_path):
        git_mod = self._hanging_git(monkeypatch)
        with pytest.raises(ValueError, match="timed out"):
            git_mod.fetch_remote_tip(str(tmp_path), "https://github.com/o/r", "github")

    def test_diff_timeout_raises_value_error(self, monkeypatch):
        git_mod = self._hanging_git(monkeypatch)
        with pytest.raises(ValueError):
            git_mod.diff_commit_names("/repo", "a" * 40, "b" * 40)

    def test_fetch_timeout_detect_returns_unknown(self, monkeypatch, tmp_path):
        import api.repositories.documents as documents_mod

        monkeypatch.setattr(
            documents_mod, "managed_repo_dir", lambda *a, **k: str(tmp_path)
        )
        self._hanging_git(monkeypatch)
        res = asyncio.run(inc.detect_changes(
            _cb(source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert res["changed"] is None


# ============================================================================
# Database detection (catalog fingerprint comparison)
# ============================================================================
@pytest.fixture
def db_catalog(monkeypatch):
    import api.docgen.database as db_mod

    state = {"info": {"catalog": "v6"}, "baseline": "hash_live"}
    units = [
        ("table", "public.users", "tbl_public_users"),
        ("table", "public.orders", "tbl_public_orders"),
        ("table", "public.audit", "tbl_public_audit"),
    ]

    async def fake_info(entity, product_id):
        if isinstance(state["info"], Exception):
            raise state["info"]
        return state["info"]

    monkeypatch.setattr(db_mod, "fresh_database_info", fake_info)
    monkeypatch.setattr(
        db_mod, "database_source_baseline", lambda info: state["baseline"]
    )
    monkeypatch.setattr(db_mod, "_subpage_units", lambda info: units)
    monkeypatch.setattr(
        db_mod, "source_fingerprint", lambda info, cat, full: f"fp:{full}"
    )
    return state


def _db(pages=None, baseline="hash_live"):
    return DatabaseORM(
        id="db_1", product_id="prod_1", name="db",
        source_state={"kind": "database", "baseline": baseline} if baseline else None,
        pages=pages or {},
    )


class TestDetectDatabase:
    def test_same_baseline_no_changes(self, db_catalog):
        res = asyncio.run(inc.detect_changes(_db(), PRODUCT, "database"))
        assert res["changed"] is False
        assert res["impacted_pages"] == []

    def test_changed_and_new_objects_impact(self, db_catalog):
        db_catalog["baseline"] = "hash_moved"
        pages = {
            "tbl_public_users": _page(
                "tbl_public_users", "users", fingerprint="fp:public.users"
            ),
            "tbl_public_orders": _page(
                "tbl_public_orders", "orders", fingerprint="fp:STALE"
            ),
            # public.audit has NO page → new.
        }
        res = asyncio.run(inc.detect_changes(_db(pages=pages), PRODUCT, "database"))
        assert res["changed"] is True
        assert res["changed_objects"] == ["public.orders", "public.audit"]
        impacted = {p["id"]: p for p in res["impacted_pages"]}
        assert set(impacted) == {"tbl_public_orders", "tbl_public_audit"}
        assert impacted["tbl_public_orders"]["new"] is False
        assert impacted["tbl_public_audit"]["new"] is True
        assert impacted["tbl_public_audit"]["title"] == "public.audit"

    def test_introspection_failure_raises(self, db_catalog):
        db_catalog["info"] = ValueError("MCP surface unusable")
        with pytest.raises(ValueError, match="MCP surface unusable"):
            asyncio.run(inc.detect_changes(_db(), PRODUCT, "database"))


# ============================================================================
# Spec detection (content hash)
# ============================================================================
class TestDetectSpec:
    CONTENT = '{"openapi": "3.0.0"}'

    def _spec(self, baseline):
        return SpecORM(
            id="spec_1", product_id="prod_1", name="s", kind="openapi",
            content=self.CONTENT,
            source_state={"kind": "spec", "baseline": baseline} if baseline else None,
        )

    def test_unchanged(self):
        baseline = hashlib.sha256(self.CONTENT.encode()).hexdigest()[:16]
        res = asyncio.run(inc.detect_changes(self._spec(baseline), PRODUCT, "spec"))
        assert res["changed"] is False
        assert res["impacted_pages"] == []
        assert "changed_files" not in res
        assert "changed_objects" not in res

    def test_changed(self):
        res = asyncio.run(inc.detect_changes(self._spec("deadbeef"), PRODUCT, "spec"))
        assert res["changed"] is True
        assert res["impacted_pages"] == []

    def test_no_baseline_is_unknown(self):
        res = asyncio.run(inc.detect_changes(self._spec(None), PRODUCT, "spec"))
        assert res["changed"] is None


# ============================================================================
# checked_at bookkeeping + result shape
# ============================================================================
class TestDetectResultShape:
    def test_checked_at_updated_baseline_kept(self, git_repo):
        git_repo["tip"] = BASELINE
        entity = _cb(source_state={"kind": "codebase", "baseline": BASELINE})
        res = asyncio.run(inc.detect_changes(entity, PRODUCT, "codebase"))
        assert res["checked_at"]
        assert entity.source_state["baseline"] == BASELINE
        assert entity.source_state["checked_at"] == res["checked_at"]

    def test_result_envelope(self, git_repo):
        git_repo["tip"] = TIP
        git_repo["diff"] = ["src/x.py"]
        res = asyncio.run(inc.detect_changes(
            _cb(source_state={"kind": "codebase", "baseline": BASELINE}),
            PRODUCT, "codebase",
        ))
        assert res["entity_type"] == "codebase"
        assert res["entity_id"] == "cb_1"
        assert res["reason"]
        assert isinstance(res["impacted_pages"], list)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            asyncio.run(inc.detect_changes(_cb(), PRODUCT, "hld"))
