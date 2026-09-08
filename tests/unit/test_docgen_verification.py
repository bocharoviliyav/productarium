"""Unit tests for api.docgen.verification (Wave D verification pipeline).

Covers:
- ``mask_secrets``: token shapes, assignments, bearer, URL credentials,
  env-style values with the placeholder whitelist; clean text untouched.
- ``extract_citations`` / ``check_citations``: inline-code paths with spans,
  bare path:span, fenced code ignored, dedupe, resolution against repo files.
- ``check_section_structure``: ok / missing / empty / placeholder.
- ``hash_text`` / ``compute_file_hashes`` / ``section_fingerprint``:
  determinism and sensitivity (file content, prompt, model).
- ``plan_regeneration``: first-run, source-unchanged reuse, source-changed,
  no-provenance.
- ``diff_sections``: added / changed / unchanged.
- ``judge_section``: consistent / inconsistent JSON verdicts, tolerant
  parsing (fenced JSON), unparseable → skipped, LLM-build failure → skipped,
  empty draft → skipped.
- ``verify_section``: end-to-end masking + citations + fingerprint with the
  judge LLM mocked.
- ``build_section_provenance`` / ``attach_provenance`` /
  ``get_stored_provenance``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

import api.docgen.verification as v


# ===========================================================================
# mask_secrets
# ===========================================================================
class TestMaskSecrets:
    def test_clean_text_untouched(self):
        text = "The API uses OAuth2 with refresh tokens stored in the vault."
        masked, findings = v.mask_secrets(text)
        assert masked == text
        assert findings == []

    def test_empty(self):
        assert v.mask_secrets("") == ("", [])
        assert v.mask_secrets(None) == (None, [])

    def test_github_token(self):
        masked, findings = v.mask_secrets("token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234")
        assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234" not in masked
        assert v.SECRET_MASK in masked
        assert findings == ["github_token"]

    def test_gitlab_token(self):
        masked, findings = v.mask_secrets("glpat-AbCdEfGh1234567890XyZ")
        assert "glpat-AbCdEfGh1234567890XyZ" not in masked
        assert findings == ["gitlab_token"]

    def test_aws_key(self):
        masked, findings = v.mask_secrets("aws AKIAIOSFODNN7EXAMPLE here")
        assert "AKIAIOSFODNN7EXAMPLE" not in masked
        assert findings == ["aws_access_key"]

    def test_openai_style_key_masked(self):
        masked, findings = v.mask_secrets("key sk-AbCdEf1234567890XyZ stays")
        assert "sk-AbCdEf1234567890XyZ" not in masked
        assert v.SECRET_MASK in masked
        assert findings == ["openai_style_key"]

    def test_sk_needed_placeholder_not_masked(self):
        # The sk- pattern excludes "needed"-prefixed values so the project's
        # "not-needed"-style placeholders survive a standalone mention.
        text = "configured token sk-needed-AbCdEf1234567890XyZ in docs"
        masked, findings = v.mask_secrets(text)
        assert masked == text
        assert "openai_style_key" not in findings


    def test_secret_assignment(self):
        masked, findings = v.mask_secrets('config: api_key = "supersecretvalue123"')
        assert "supersecretvalue123" not in masked
        assert "secret_assignment" in findings

    def test_bearer_token(self):
        masked, findings = v.mask_secrets("Authorization: Bearer abcdef1234567890abcdef")
        assert "abcdef1234567890abcdef" not in masked
        assert "bearer_token" in findings

    def test_url_credentials(self):
        masked, findings = v.mask_secrets("db at https://user:passw0rd@example.com/db")
        assert "passw0rd" not in masked
        assert "url_credentials" in findings

    def test_env_assignment_masked(self):
        masked, findings = v.mask_secrets("GITHUB_TOKEN=gh-real-token-value-123")
        assert "gh-real-token-value-123" not in masked
        assert "env_secret" in findings
        # The variable NAME stays visible (useful context, not a secret).
        assert "GITHUB_TOKEN=" in masked

    def test_env_placeholder_whitelist(self):
        text = "LOCAL_OPENAI_API_KEY=not-needed"
        masked, findings = v.mask_secrets(text)
        assert masked == text
        assert findings == []

    def test_env_interpolation_not_masked(self):
        text = "DB_PASSWORD=${DB_PASSWORD}"
        masked, findings = v.mask_secrets(text)
        assert findings == []

    def test_json_quoted_assignment_masked(self):
        masked, findings = v.mask_secrets('{"password": "Hunter2SuperSecret"}')
        assert "Hunter2SuperSecret" not in masked
        assert "secret_assignment" in findings
        assert '"password": ***REDACTED***' in masked

    def test_yaml_unquoted_password_masked(self):
        masked, findings = v.mask_secrets("password: Hunter2XXX")
        assert "Hunter2XXX" not in masked
        assert "secret_assignment" in findings

    def test_hex_secret_after_key_masked(self):
        hexval = "3f9a7c1d5e8b2a4067f1c3d9e5a8b7c2"
        masked, findings = v.mask_secrets(f"secret_key: {hexval}")
        assert hexval not in masked
        assert "secret_assignment" in findings
        assert "secret_key:" in masked  # the key stays visible

    def test_gsk_xai_token(self):
        token = "gsk_ABCDEFGHIJKLMNOPQRSTUV"
        masked, findings = v.mask_secrets(f"uses {token} here")
        assert token not in masked
        assert findings == ["xai_api_key"]

    def test_hash_env_var_masked(self):
        masked, findings = v.mask_secrets("HEX_HASH=deadbeefcafebabe")
        assert "deadbeefcafebabe" not in masked
        assert findings == ["env_secret"]
        assert "HEX_HASH=" in masked  # the variable name stays visible

    def test_multiple_findings_labels_only(self):
        masked, findings = v.mask_secrets(
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234 and glpat-AbCdEfGh1234567890XyZ"
        )
        assert sorted(set(findings)) == ["github_token", "gitlab_token"]
        # No raw secret value leaks into the findings list.
        for f in findings:
            assert "ghp_" not in f and "glpat-" not in f


# ===========================================================================
# extract_citations + check_citations
# ===========================================================================
class TestExtractCitations:
    def test_empty(self):
        assert v.extract_citations("") == []
        assert v.extract_citations(None) == []

    def test_inline_code_path(self):
        cites = v.extract_citations("See `src/api.py` for details.")
        assert len(cites) == 1
        assert cites[0].path == "src/api.py"
        assert cites[0].line_span is None

    def test_inline_code_path_with_span(self):
        cites = v.extract_citations("Auth lives in `src/auth.py:42-58`.")
        assert cites[0].path == "src/auth.py"
        assert cites[0].line_span == "42-58"
        assert cites[0].key() == "src/auth.py:42-58"

    def test_bare_path_with_span(self):
        cites = v.extract_citations("Implemented in src/api.py:10 per the spec.")
        assert any(c.path == "src/api.py" and c.line_span == "10" for c in cites)

    def test_fenced_code_ignored(self):
        text = "```python\nx = 'src/api.py'\n```"
        assert v.extract_citations(text) == []

    def test_urls_ignored(self):
        assert v.extract_citations("Visit `https://example.com/api`") == []

    def test_deduplicated_in_order(self):
        text = "`src/a.py` then `src/b.py` then `src/a.py`"
        cites = v.extract_citations(text)
        assert [c.path for c in cites] == ["src/a.py", "src/b.py"]

    def test_single_word_not_a_path(self):
        # A bare backticked word without / or a known extension is not a citation.
        assert v.extract_citations("Use the `FastAPI` framework") == []

    def test_root_file_with_ext_is_citation(self):
        cites = v.extract_citations("Config in `docker-compose.yml`")
        assert cites and cites[0].path == "docker-compose.yml"


class TestCheckCitations:
    def test_split(self):
        citations = [
            v.Citation(path="src/api.py"),
            v.Citation(path="nope/missing.go"),
        ]
        resolved, unresolved = v.check_citations(citations, ["src/api.py", "README.md"])
        assert [c.path for c in resolved] == ["src/api.py"]
        assert [c.path for c in unresolved] == ["nope/missing.go"]

    def test_normalizes_dots_and_slashes(self):
        citations = [v.Citation(path="./src/../src/api.py")]
        resolved, _ = v.check_citations(citations, ["src/api.py"])
        assert resolved

    def test_empty_inputs(self):
        resolved, unresolved = v.check_citations([], [])
        assert resolved == [] and unresolved == []


# ===========================================================================
# check_section_structure
# ===========================================================================
class TestCheckSectionStructure:
    def test_ok(self):
        sections = {"overview": "text", "architecture": "text"}
        check = v.check_section_structure(sections, ["overview", "architecture"])
        assert check.ok
        assert check.missing == [] and check.empty == []

    def test_missing(self):
        check = v.check_section_structure({"overview": "x"}, ["overview", "qa"])
        assert not check.ok
        assert check.missing == ["qa"]

    def test_empty(self):
        check = v.check_section_structure({"overview": "   "}, ["overview"])
        assert check.empty == ["overview"]

    def test_placeholder_counts_as_empty(self):
        check = v.check_section_structure(
            {"overview": "PLACEHOLDER"}, ["overview"], placeholder="PLACEHOLDER"
        )
        assert check.empty == ["overview"]

    def test_placeholder_with_whitespace(self):
        check = v.check_section_structure(
            {"overview": " PLACEHOLDER \n"}, ["overview"], placeholder="PLACEHOLDER"
        )
        assert check.empty == ["overview"]


# ===========================================================================
# hashes + fingerprint
# ===========================================================================
class TestHashesAndFingerprint:
    def test_hash_text_stable(self):
        assert v.hash_text("abc") == v.hash_text("abc")
        assert v.hash_text("abc") != v.hash_text("abd")

    def test_hash_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("content")
        assert v.hash_file(str(p)) == v.hash_text("content")
        assert v.hash_file(str(tmp_path / "missing.txt")) is None

    def test_compute_file_hashes_confined(self, tmp_path):
        (tmp_path / "a.py").write_text("A")
        hashes = v.compute_file_hashes(str(tmp_path), ["a.py", "missing.py", "../outside.py"])
        assert hashes["a.py"] == v.hash_text("A")
        assert hashes["missing.py"] is None
        # Escaping paths are stored under their NORMALIZED form and hash to
        # None (confinement): "../outside.py" → "outside.py".
        assert hashes["outside.py"] is None

    def test_fingerprint_deterministic(self):
        fh = {"a.py": "hash_a", "b.py": "hash_b"}
        fp1 = v.section_fingerprint("overview", fh, prompt_hash="p", model="m")
        fp2 = v.section_fingerprint("overview", dict(reversed(list(fh.items()))), prompt_hash="p", model="m")
        assert fp1 == fp2

    def test_fingerprint_sensitivity(self):
        base = v.section_fingerprint("overview", {"a.py": "h1"}, prompt_hash="p", model="m")
        assert base != v.section_fingerprint("overview", {"a.py": "h2"}, prompt_hash="p", model="m")
        assert base != v.section_fingerprint("overview", {"a.py": "h1"}, prompt_hash="p2", model="m")
        assert base != v.section_fingerprint("overview", {"a.py": "h1"}, prompt_hash="p", model="m2")
        assert base != v.section_fingerprint("qa", {"a.py": "h1"}, prompt_hash="p", model="m")

    def test_hash_file_null_byte_returns_none(self):
        # open() raises ValueError ("embedded null byte") — caught, not raised.
        assert v.hash_file("main.py\x00.py") is None

    def test_compute_file_hashes_symlink_escape_hashes_none(self, tmp_path):
        (tmp_path / "a.py").write_text("A")
        outside = tmp_path.parent / "outside_escape.py"
        outside.write_text("OUTSIDE")
        os.symlink(outside, tmp_path / "link.py")
        hashes = v.compute_file_hashes(str(tmp_path), ["a.py", "link.py"])
        assert hashes["a.py"] == v.hash_text("A")
        # realpath resolution sends the symlink outside repo_dir -> None.
        assert hashes["link.py"] is None

    def test_fingerprint_sensitive_to_base_url_and_tree_hash(self):
        base = v.section_fingerprint("overview", {"a.py": "h1"}, prompt_hash="p", model="m")
        assert base != v.section_fingerprint(
            "overview", {"a.py": "h1"}, prompt_hash="p", model="m",
            base_url="http://gw:8080/v1",
        )
        with_t1 = v.section_fingerprint(
            "overview", {"a.py": "h1"}, prompt_hash="p", model="m", tree_hash="tree-1",
        )
        with_t2 = v.section_fingerprint(
            "overview", {"a.py": "h1"}, prompt_hash="p", model="m", tree_hash="tree-2",
        )
        assert base != with_t1 and with_t1 != with_t2

    def test_missing_file_hash_changes_fingerprint(self):
        # None → "missing" sentinel, still deterministic.
        fp1 = v.section_fingerprint("s", {"x": None})
        fp2 = v.section_fingerprint("s", {"x": None})
        assert fp1 == fp2
        assert fp1 != v.section_fingerprint("s", {"x": "real"})


# ===========================================================================
# plan_regeneration + diff_sections
# ===========================================================================
class TestPlanRegeneration:
    def _repo(self, tmp_path):
        (tmp_path / "main.py").write_text("print('v1')\n")
        return str(tmp_path)

    def _pages_with_provenance(self, tmp_path, model="m"):
        """Build pages whose fingerprints match the CURRENT repo content."""
        (tmp_path / "main.py").write_text("print('v1')\n")
        file_hashes = v.compute_file_hashes(str(tmp_path), ["main.py"])
        fp = v.section_fingerprint("overview", file_hashes, prompt_hash="ph", model=model)
        return {
            "page_overview": {
                "id": "page_overview",
                "title": "Overview",
                "content": "Old overview text",
                "provenance": {
                    "section_id": "overview",
                    "fingerprint": fp,
                    "source_files": ["main.py"],
                },
            }
        }

    def test_first_run_regenerates_all(self, tmp_path):
        plan = v.plan_regeneration({}, {}, self._repo(tmp_path), ["overview", "qa"])
        assert plan.regenerate == ["overview", "qa"]
        assert plan.reuse == {}
        assert plan.reasons == {"overview": "first-run", "qa": "first-run"}

    def test_source_unchanged_reuses(self, tmp_path):
        repo = self._repo(tmp_path)
        pages = self._pages_with_provenance(tmp_path)
        plan = v.plan_regeneration(
            pages, {"overview": "Old overview text"}, repo, ["overview"],
            prompt_hashes={"overview": "ph"}, model="m",
        )
        assert plan.reuse == {"overview": "Old overview text"}
        assert plan.regenerate == []
        assert plan.reasons["overview"] == "source-unchanged"

    def test_source_changed_regenerates(self, tmp_path):
        repo = self._repo(tmp_path)
        pages = self._pages_with_provenance(tmp_path)
        # Mutate the tracked source file AFTER the fingerprint was stored.
        (tmp_path / "main.py").write_text("print('v2 CHANGED')\n")
        plan = v.plan_regeneration(
            pages, {"overview": "Old overview text"}, repo, ["overview"],
            prompt_hashes={"overview": "ph"}, model="m",
        )
        assert plan.regenerate == ["overview"]
        assert plan.reasons["overview"] == "source-changed"

    def test_no_provenance_regenerates(self, tmp_path):
        pages = {"page_overview": {"content": "old text"}}  # no provenance key
        plan = v.plan_regeneration(
            pages, {"overview": "old text"}, self._repo(tmp_path), ["overview"],
        )
        assert plan.regenerate == ["overview"]
        assert plan.reasons["overview"] == "no-provenance"

    def test_tree_hash_change_regenerates(self, tmp_path):
        """Same file contents, different WHOLE-TREE hash -> regenerate (a
        breaking edit in a file the section never opened invalidates reuse)."""
        repo = self._repo(tmp_path)
        file_hashes = v.compute_file_hashes(repo, ["main.py"])
        fp = v.section_fingerprint(
            "overview", file_hashes, prompt_hash="ph", model="m", tree_hash="tree-1",
        )
        pages = {"page_overview": {
            "id": "page_overview", "content": "Old overview text",
            "provenance": {
                "section_id": "overview", "fingerprint": fp,
                "source_files": ["main.py"],
            },
        }}
        same_tree = v.plan_regeneration(
            pages, {"overview": "Old overview text"}, repo, ["overview"],
            prompt_hashes={"overview": "ph"}, model="m", tree_hash="tree-1",
        )
        assert same_tree.reuse == {"overview": "Old overview text"}
        new_tree = v.plan_regeneration(
            pages, {"overview": "Old overview text"}, repo, ["overview"],
            prompt_hashes={"overview": "ph"}, model="m", tree_hash="tree-2",
        )
        assert new_tree.regenerate == ["overview"]
        assert new_tree.reasons["overview"] == "source-changed"

    def test_empty_old_content_regenerates(self, tmp_path):
        pages = self._pages_with_provenance(tmp_path)
        plan = v.plan_regeneration(
            pages, {"overview": ""}, self._repo(tmp_path), ["overview"],
            prompt_hashes={"overview": "ph"}, model="m",
        )
        assert plan.regenerate == ["overview"]


class TestDiffSections:
    def test_statuses(self):
        old = {"a": "same", "b": "old", }
        new = {"a": "same", "b": "new", "c": "added"}
        assert v.diff_sections(old, new) == {"a": "unchanged", "b": "changed", "c": "added"}

    def test_whitespace_insensitive(self):
        assert v.diff_sections({"a": "x"}, {"a": " x \n"}) == {"a": "unchanged"}

    def test_empty(self):
        assert v.diff_sections({}, {}) == {}


# ===========================================================================
# judge_section
# ===========================================================================
class _FakeJudgeLLM:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.prompts: list = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._error:
            raise self._error
        return self._response


class TestJudgeSection:
    def _patch_llm(self, monkeypatch, fake):
        monkeypatch.setattr(v, "_build_judge_llm", lambda *a, **kw: fake)

    def test_consistent_verdict(self, monkeypatch):
        self._patch_llm(monkeypatch, _FakeJudgeLLM('{"consistent": true, "issues": []}'))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "consistent"
        assert verdict.issues == []

    def test_inconsistent_verdict_with_issues(self, monkeypatch):
        raw = '{"consistent": false, "issues": ["fabricated endpoint /x", "wrong version"]}'
        self._patch_llm(monkeypatch, _FakeJudgeLLM(raw))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "inconsistent"
        assert len(verdict.issues) == 2

    def test_fenced_json_parsed(self, monkeypatch):
        raw = '```json\n{"consistent": true, "issues": []}\n```'
        self._patch_llm(monkeypatch, _FakeJudgeLLM(raw))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "consistent"

    def test_prose_wrapped_json_parsed(self, monkeypatch):
        raw = 'Verdict: {"consistent": false, "issues": ["x"]} — see report.'
        self._patch_llm(monkeypatch, _FakeJudgeLLM(raw))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence")
)
        assert verdict.verdict == "inconsistent"

    def test_unparseable_skipped(self, monkeypatch):
        self._patch_llm(monkeypatch, _FakeJudgeLLM("not json at all"))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "skipped"

    def test_llm_error_skipped(self, monkeypatch):
        self._patch_llm(monkeypatch, _FakeJudgeLLM(error=RuntimeError("down")))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "skipped"

    def test_build_failure_skipped(self, monkeypatch):
        def _boom(*a, **kw):
            raise RuntimeError("no llm")
        monkeypatch.setattr(v, "_build_judge_llm", _boom)
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "skipped"

    def test_empty_draft_skipped(self, monkeypatch):
        self._patch_llm(monkeypatch, _FakeJudgeLLM('{"consistent": true}'))
        verdict = asyncio.run(v.judge_section("overview", "", "evidence"))
        assert verdict.verdict == "skipped"
        assert "empty draft" in verdict.issues

    def test_string_false_is_inconsistent(self, monkeypatch):
        """A JSON \"false\" STRING must not be truthy (strict parsing)."""
        self._patch_llm(monkeypatch, _FakeJudgeLLM('{"consistent": "false", "issues": ["x"]}'))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "inconsistent"

    def test_int_zero_is_inconsistent(self, monkeypatch):
        self._patch_llm(monkeypatch, _FakeJudgeLLM('{"consistent": 0, "issues": []}'))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "inconsistent"

    def test_int_one_is_consistent(self, monkeypatch):
        self._patch_llm(monkeypatch, _FakeJudgeLLM('{"consistent": 1, "issues": []}'))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "consistent"

    def test_missing_consistent_fail_open(self, monkeypatch):
        self._patch_llm(monkeypatch, _FakeJudgeLLM('{"issues": []}'))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert verdict.verdict == "consistent"

    def test_prompt_carries_draft_and_evidence(self, monkeypatch):
        fake = _FakeJudgeLLM('{"consistent": true, "issues": []}')
        self._patch_llm(monkeypatch, fake)
        asyncio.run(v.judge_section("overview", "DRAFT-XYZ", "EVIDENCE-ABC"))
        assert len(fake.prompts) == 1
        assert "DRAFT-XYZ" in fake.prompts[0]
        assert "EVIDENCE-ABC" in fake.prompts[0]

    def test_issues_capped_and_truncated(self, monkeypatch):
        issues = [f"issue {i} " + "x" * 500 for i in range(30)]
        raw = '{"consistent": false, "issues": ' + str(issues).replace("'", '"') + "}"
        self._patch_llm(monkeypatch, _FakeJudgeLLM(raw))
        verdict = asyncio.run(v.judge_section("overview", "draft", "evidence"))
        assert len(verdict.issues) == 20
        assert all(len(i) <= 300 for i in verdict.issues)


# ===========================================================================
# verify_section (end-to-end with the judge mocked)
# ===========================================================================
class TestVerifySection:
    def test_masking_citations_fingerprint(self, tmp_path, monkeypatch):
        (tmp_path / "main.py").write_text("import os\nimport sys\n")
        repo = str(tmp_path)
        monkeypatch.setattr(v, "_build_judge_llm", lambda *a, **kw: _FakeJudgeLLM(
            '{"consistent": true, "issues": []}'
        ))
        content = (
            "Uses `main.py:1-2` and `ghost/missing.py:5`. "
            "Token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234 leaked."
        )
        result = asyncio.run(v.verify_section(
            "overview", content,
            repo_dir=repo,
            repo_files=["main.py"],
            source_files=["main.py"],
            model="m",
            prompt_hash="ph",
            run_judge=True,
            judge_evidence="import os",
        ))
        assert "ghp_" not in result.masked_content
        assert v.SECRET_MASK in result.masked_content
        assert result.secrets_masked == 1
        # Citation guard (Wave 1.2): the resolvable span survives, the ghost
        # citation is REMOVED from the persisted text (not just warned).
        assert result.citations_resolved == ["main.py:1-2"]
        assert result.citations_unresolved == []
        assert result.citations_removed == ["`ghost/missing.py:5`"]
        assert "ghost/missing.py" not in result.masked_content
        assert result.fingerprint  # computed over main.py
        assert result.judge.verdict == "consistent"
        assert any("secret" in w for w in result.warnings)
        assert any("citation" in w for w in result.warnings)

    def test_judge_off(self, tmp_path, monkeypatch):
        (tmp_path / "a.py").write_text("x = 1\n")
        result = asyncio.run(v.verify_section(
            "qa", "plain content",
            repo_dir=str(tmp_path), repo_files=["a.py"], source_files=["a.py"],
            run_judge=False,
        ))
        assert result.judge is None

    def test_empty_source_files_fingerprint_still_computed(self, tmp_path):
        result = asyncio.run(v.verify_section(
            "cicd", "content",
            repo_dir=str(tmp_path), repo_files=[], source_files=[],
            run_judge=False,
        ))
        assert result.fingerprint


# ===========================================================================
# provenance build/attach/read
# ===========================================================================
class TestProvenance:
    def test_build_section_provenance_shape(self):
        prov = v.build_section_provenance(
            "overview",
            model="qwen/test",
            prompt_file="overview.md",
            source_files=["b.py", "a.py", "a.py"],
            fingerprint="fp",
            citations={"resolved": ["a.py:1"], "unresolved": []},
            judge=v.JudgeVerdict(verdict="inconsistent", issues=["x"]),
            regen="generated",
            mermaid_stats={"checked": 2, "repaired": 1},
            secrets_masked=3,
        )
        assert prov["section_id"] == "overview"
        assert prov["model"] == "qwen/test"
        assert prov["prompt_file"] == "overview.md"
        assert prov["source_files"] == ["a.py", "b.py"]  # sorted + deduped
        assert prov["fingerprint"] == "fp"
        assert prov["regen"] == "generated"
        assert prov["generator"] == "deepagents"
        assert prov["citations"] == {"resolved": ["a.py:1"], "unresolved": []}
        assert prov["judge"] == {"verdict": "inconsistent", "issues": ["x"]}
        assert prov["mermaid"] == {"checked": 2, "repaired": 1}
        assert prov["secrets_masked"] == 3
        assert prov["generated_at"]

    def test_generator_label_for_legacy_fallback(self):
        prov = v.build_section_provenance(
            "overview", model=None, prompt_file="overview.md", source_files=[],
            fingerprint=None, citations={"resolved": [], "unresolved": []},
            judge=None, regen="legacy-fallback",
        )
        assert prov["generator"] == "standard-llm"

    def test_reused_sections_have_no_judge_key(self):
        prov = v.build_section_provenance(
            "overview", model="m", prompt_file="overview.md", source_files=[],
            fingerprint="fp", citations={"resolved": [], "unresolved": []},
            judge=None, regen="reused-unchanged",
        )
        assert "judge" not in prov

    def test_attach_and_read(self):
        pages = {"page_overview": {"id": "page_overview", "content": "x"}}
        prov = v.build_section_provenance(
            "overview", model="m", prompt_file="overview.md", source_files=[],
            fingerprint="fp", citations={"resolved": [], "unresolved": []},
            judge=None, regen="generated",
        )
        v.attach_provenance(pages, "overview", prov)
        assert pages["page_overview"]["provenance"] == prov
        assert v.get_stored_provenance(pages["page_overview"]) == prov

    def test_attach_missing_page_noop(self):
        pages = {}
        v.attach_provenance(pages, "ghost", {"anything": 1})
        assert pages == {}

    def test_get_stored_provenance_tolerant(self):
        assert v.get_stored_provenance(None) == {}
        assert v.get_stored_provenance({}) == {}
        assert v.get_stored_provenance({"provenance": "not-a-dict"}) == {}

        class Obj:
            provenance = {"k": 1}

        assert v.get_stored_provenance(Obj()) == {"k": 1}
