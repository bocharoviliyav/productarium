"""Unit tests for api.docgen.citation_guard (Wave 1.2 port from the fork).

Covers:
- ``apply_outside_fences``: fenced blocks untouched, outside regions passed
  to the transform as ONE string.
- ``guard_citations``: removal of unresolvable citations (inline-code and
  bare-span forms), stripping of impossible line spans (with fail-closed
  unknown counts), verbatim keep of valid citations, fence protection,
  non-citation backticked words / URLs / markdown links untouched, exact
  report contents.
- ``count_file_lines``: real count, missing file, path escaping the repo
  root.
- ``verify_section`` integration: the guarded text is what the caller
  persists; removed/fixed land on the result and in the provenance payload;
  flows without a file set keep the warning-only behaviour.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import api.docgen.citation_guard as g
import api.docgen.verification as v


ALLOWED = {"src/main.py", "docs/readme.md"}


def _guard(markdown, allowed=ALLOWED, counts=None):
    return g.guard_citations(markdown, set(allowed), counts or {"src/main.py": 10, "docs/readme.md": 40})


# ===========================================================================
# apply_outside_fences
# ===========================================================================
class TestApplyOutsideFences:
    def test_fenced_block_untouched(self):
        md = "before\n```python\npath = 'src/gone.py:1-2'\n```\nafter"
        out = g.apply_outside_fences(md, lambda r: r.upper())
        assert out == "BEFORE\n```python\npath = 'src/gone.py:1-2'\n```\nAFTER"

    def test_region_passed_as_one_string(self):
        seen = []

        def transform(region):
            seen.append(region)
            return region

        md = "a\n```x\nf\n```\nb"
        g.apply_outside_fences(md, transform)
        assert seen == ["a", "b"]  # each outside region whole, not per line

    def test_text_without_fences(self):
        assert g.apply_outside_fences("hello", lambda r: r + "!") == "hello!"


# ===========================================================================
# guard_citations: removal
# ===========================================================================
class TestGuardRemoves:
    def test_unresolvable_inline_code_citation_removed(self):
        md = "The entry `src/gone.py` does work."
        cleaned, report = _guard(md)
        assert cleaned == "The entry does work."
        assert report.removed == ["`src/gone.py`"]
        assert report.fixed == []
        assert report.touched

    def test_unresolvable_bare_span_removed(self):
        md = "See also src/gone.py:5-9 for details."
        cleaned, report = _guard(md)
        assert "src/gone.py" not in cleaned
        assert "See also for details." in cleaned
        assert report.removed == ["src/gone.py:5-9"]

    def test_removal_at_sentence_end_keeps_punctuation(self):
        md = "Loads `src/gone.py`."
        cleaned, _ = _guard(md)
        assert cleaned == "Loads ."

    def test_empty_allowed_set_is_noop(self):
        md = "See `src/gone.py` now."
        cleaned, report = g.guard_citations(md, set(), {})
        assert cleaned == md
        assert not report.touched


# ===========================================================================
# guard_citations: span fixes
# ===========================================================================
class TestGuardFixesSpans:
    def test_valid_span_kept_verbatim(self):
        md = "Entry `src/main.py:2-5` handles boot."
        cleaned, report = _guard(md)
        assert cleaned == md
        assert not report.touched

    def test_span_beyond_file_stripped_path_kept(self):
        md = "Entry `src/main.py:5-99` handles boot."
        cleaned, report = _guard(md)
        assert cleaned == "Entry `src/main.py` handles boot."
        assert report.fixed == ["src/main.py (lines 5-99 stripped)"]

    def test_reversed_span_stripped(self):
        cleaned, report = _guard("x `src/main.py:9-2` y")
        assert cleaned == "x `src/main.py` y"
        assert len(report.fixed) == 1

    def test_zero_line_stripped(self):
        cleaned, report = _guard("x `src/main.py:0-3` y")
        assert cleaned == "x `src/main.py` y"

    def test_unknown_line_count_fails_closed(self):
        # Allowed file WITHOUT a line count: the span cannot be verified.
        cleaned, report = g.guard_citations(
            "x `docs/readme.md:1-2` y", {"docs/readme.md"}, {}
        )
        assert cleaned == "x `docs/readme.md` y"
        assert report.fixed == ["docs/readme.md (lines 1-2 stripped)"]

    def test_bare_span_stripped_to_bare_path(self):
        cleaned, report = _guard("See src/main.py:2-50 here.")
        assert cleaned == "See src/main.py here."
        assert report.fixed == ["src/main.py (lines 2-50 stripped)"]


# ===========================================================================
# guard_citations: what must NOT be touched
# ===========================================================================
class TestGuardLeaves:
    def test_citation_inside_fence_untouched(self):
        md = "text\n```\n`src/gone.py:1-2` and src/gone.py:3-4\n```\nmore"
        cleaned, report = _guard(md)
        assert cleaned == md
        assert not report.touched

    def test_plain_backticked_word_untouched(self):
        cleaned, report = _guard("Call `main` first, then `run`.")
        assert cleaned == "Call `main` first, then `run`."
        assert not report.touched

    def test_url_untouched(self):
        cleaned, report = _guard("Docs at `https://example.com/x.py` online.")
        assert cleaned == "Docs at `https://example.com/x.py` online."
        assert not report.touched

    def test_markdown_link_untouched(self):
        cleaned, report = _guard("See [the guide](https://x.example/gone.py).")
        assert cleaned == "See [the guide](https://x.example/gone.py)."
        assert not report.touched


# ===========================================================================
# count_file_lines
# ===========================================================================
class TestCountFileLines:
    def test_counts_real_file(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("one\ntwo\nthree\n")
        assert g.count_file_lines(str(tmp_path), "a.py") == 3

    def test_missing_file_is_zero(self, tmp_path):
        assert g.count_file_lines(str(tmp_path), "nope.py") == 0

    def test_escape_outside_root_is_zero(self, tmp_path, tmp_path_factory):
        # An ABSOLUTE rel_path pointing at a file OUTSIDE the repo root must
        # be refused (os.path.join keeps absolute second args verbatim).
        other = tmp_path_factory.mktemp("other")
        victim = other / "secret.py"
        victim.write_text("x\n")
        assert g.count_file_lines(str(tmp_path), str(victim)) == 0


# ===========================================================================
# verify_section integration
# ===========================================================================
class TestVerifySectionIntegration:
    def _repo(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("\n".join(f"line {i}" for i in range(1, 11)))
        return [str(tmp_path), ["src/main.py"]]

    def test_guard_rewrites_persisted_content(self, tmp_path):
        repo_dir, repo_files = self._repo(tmp_path)
        content = (
            "Boot lives in `src/main.py:1-3`.\n"
            "Broken span `src/main.py:5-999`.\n"
            "Fantasy `src/ghost.py` reference."
        )
        result = asyncio.run(
            v.verify_section(
                "overview", content,
                repo_dir=repo_dir, repo_files=repo_files,
                source_files=["src/main.py"], run_judge=False,
            )
        )
        assert result.masked_content == (
            "Boot lives in `src/main.py:1-3`.\n"
            "Broken span `src/main.py`.\n"
            "Fantasy reference."
        )
        assert result.citations_removed == ["`src/ghost.py`"]
        assert result.citations_fixed == ["src/main.py (lines 5-999 stripped)"]
        assert any("removed 1" in w for w in result.warnings)
        assert any("stripped invalid line span" in w for w in result.warnings)
        # The extractor ran on the CLEANED text: nothing unresolved remains.
        assert result.citations_unresolved == []

    def test_provenance_carries_guard_outcome(self, tmp_path):
        repo_dir, repo_files = self._repo(tmp_path)
        result = asyncio.run(
            v.verify_section(
                "overview", "See `src/ghost.py` and `src/main.py:0-2`.",
                repo_dir=repo_dir, repo_files=repo_files,
                source_files=["src/main.py"], run_judge=False,
            )
        )
        prov = v.build_section_provenance(
            "overview",
            model=None, prompt_file="docgen_sections.md",
            source_files=["src/main.py"], fingerprint=result.fingerprint,
            citations={
                "resolved": result.citations_resolved,
                "unresolved": result.citations_unresolved,
                "removed": result.citations_removed,
                "fixed": result.citations_fixed,
            },
            judge=None, regen="generated",
        )
        assert prov["citations"]["removed"] == ["`src/ghost.py`"]
        assert prov["citations"]["fixed"] == ["src/main.py (lines 0-2 stripped)"]

    def test_no_file_set_keeps_warn_only_behaviour(self):
        result = asyncio.run(
            v.verify_section(
                "overview", "See `src/ghost.py`.",
                repo_dir="", repo_files=[], source_files=[],
                run_judge=False,
            )
        )
        assert result.masked_content == "See `src/ghost.py`."
        assert result.citations_removed == []
        assert result.citations_fixed == []
        assert result.citations_unresolved == ["src/ghost.py"]
