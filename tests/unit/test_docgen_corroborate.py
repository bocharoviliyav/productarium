"""Unit tests for ``api.docgen.corroborate`` (3.1: anti-fabrication filter).

Hermetic: pure stdlib logic + tmp_path file scans; ``verify_section`` runs
with the judge disabled. No network, no DB.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import api.docgen.corroborate as corrob
import api.docgen.verification as v


# ============================================================================
# grounding_index (port of the fork's _grounding_index)
# ============================================================================
class TestGroundingIndex:
    def test_whole_token_and_dotted_segments(self):
        idx = corrob.grounding_index({"com.example.app"})
        assert "com.example.app" in idx
        assert "com" in idx and "example" in idx and "app" in idx

    def test_underscore_segments(self):
        idx = corrob.grounding_index({"CASH_CARD"})
        assert "cash_card" in idx
        assert "cash" in idx and "card" in idx

    def test_camel_segments(self):
        idx = corrob.grounding_index({"LoginController"})
        assert "logincontroller" in idx
        assert "login" in idx and "controller" in idx

    def test_short_segments_dropped(self):
        idx = corrob.grounding_index({"main.py"})
        assert "main" in idx
        assert "py" not in idx  # len <= 2 segments are noise

    def test_empty_tokens_skipped(self):
        assert "" not in corrob.grounding_index({"", None})  # type: ignore[arg-type]


# ============================================================================
# filter_ungrounded_prose
# ============================================================================
class TestFilterUngroundedProse:
    def test_drops_sentence_with_invented_identifier(self):
        text = (
            "The module prints hello. "
            "Ghost `GhostWidgetFactory` assembles the payload."
        )
        out, report = corrob.filter_ungrounded_prose(text, {"print", "hello"})
        assert "GhostWidgetFactory" not in out
        assert "module prints hello" in out
        assert report.sentences_removed == 1
        assert report.removed_identifiers == ["GhostWidgetFactory"]
        assert report.touched

    def test_plain_prose_without_code_tokens_untouched(self):
        text = "Обычная проза без идентификаторов. Second plain sentence."
        out, report = corrob.filter_ungrounded_prose(text, {"zzz"})
        assert out == text
        assert not report.touched

    def test_camel_segments_ground(self):
        out, report = corrob.filter_ungrounded_prose(
            "LoginController handles auth.", {"login", "controller"}
        )
        assert out == "LoginController handles auth."
        assert not report.touched

    def test_dotted_identifier_grounds_by_segments(self):
        out, _ = corrob.filter_ungrounded_prose(
            "The `public.users` table stores rows.", {"public", "users"}
        )
        assert "public.users" in out

    def test_screaming_snake_checked_but_single_caps_not(self):
        # ALLCAPS emphasis in Latin is NOT a code token (our docs are RU
        # prose where caps words are common)…
        out, _ = corrob.filter_ungrounded_prose(
            "RUN ONE section body.", {"zzz"}
        )
        assert out == "RUN ONE section body."
        # …while underscore-joined enums ARE and must ground.
        grounded, _ = corrob.filter_ungrounded_prose(
            "The `CASH_CARD` enum is used.", {"cash", "card"}
        )
        assert "CASH_CARD" in grounded
        dropped, report = corrob.filter_ungrounded_prose(
            "The `CASH_CARD` enum is used. Plain tail survives.", {"cash"}
        )
        assert "CASH_CARD" not in dropped
        assert "Plain tail survives." in dropped
        assert report.removed_identifiers == ["CASH_CARD"]

    def test_bullets_headings_quotes_tables_untouched(self):
        text = (
            "# Header mentions `GhostWidgetFactory`\n"
            "\n"
            "- bullet with `GhostWidgetFactory`\n"
            "\n"
            "> quoted `GhostWidgetFactory`\n"
            "\n"
            "| col | `GhostWidgetFactory` |\n"
            "\n"
            "Prose line. Sentence with `GhostWidgetFactory` dropped."
        )
        out, report = corrob.filter_ungrounded_prose(text, {"zzz"})
        assert "Header mentions" in out
        assert "- bullet with `GhostWidgetFactory`" in out
        assert "> quoted `GhostWidgetFactory`" in out
        assert "| col | `GhostWidgetFactory` |" in out
        assert "Prose line." in out
        assert "dropped" not in out
        assert report.sentences_removed == 1

    def test_fenced_code_untouched(self):
        text = (
            "Intro sentence.\n"
            "\n"
            "```python\n"
            "class GhostWidgetFactory:\n"
            "    pass\n"
            "```\n"
            "\n"
            "Tail mentions `GhostWidgetFactory` again."
        )
        out, _ = corrob.filter_ungrounded_prose(text, {"zzz"})
        assert "class GhostWidgetFactory:" in out
        assert "Tail mentions" not in out

    def test_paragraph_breaks_preserved(self):
        text = "First para. Solid.\n\nSecond para. Also solid."
        out, _ = corrob.filter_ungrounded_prose(text, {"zzz"})
        assert out == text

    def test_fully_dropped_paragraph_disappears(self):
        text = "Good para stays. `GhostA` vanishes.\n\nWhole para gone `GhostB` here."
        out, report = corrob.filter_ungrounded_prose(text, {"zzz"})
        assert out == "Good para stays."
        assert report.removed_identifiers == ["GhostA", "GhostB"]
        assert report.sentences_removed == 2

    def test_multiple_identifiers_in_one_sentence_counted_once(self):
        text = "`GhostA` and `GhostB` together."
        out, report = corrob.filter_ungrounded_prose(text, {"zzz"})
        # Whole doc would be empty → fail-open keeps the original.
        assert out == text
        text2 = "`GhostA` and `GhostB` together. Kept sentence."
        out2, report2 = corrob.filter_ungrounded_prose(text2, {"zzz"})
        assert out2 == "Kept sentence."
        assert report2.sentences_removed == 1
        assert report2.removed_identifiers == ["GhostA", "GhostB"]

    def test_fail_open_when_filter_empties_everything(self):
        text = "Only `GhostWidgetFactory` exists."
        out, report = corrob.filter_ungrounded_prose(text, {"zzz"})
        assert out == text
        assert report.emptied is True
        assert report.removed_identifiers == []
        assert not report.touched

    def test_empty_grounding_passthrough(self):
        text = "`GhostWidgetFactory` everywhere."
        out, report = corrob.filter_ungrounded_prose(text, set())
        assert out == text
        assert not report.touched and not report.emptied

    def test_empty_text_passthrough(self):
        out, report = corrob.filter_ungrounded_prose("", {"x"})
        assert out == ""
        assert not report.touched

    def test_always_grounded_vocabulary(self):
        # json/api are common vocabulary: JsonParser grounds via json+parser.
        out, _ = corrob.filter_ungrounded_prose(
            "Uses `JsonParser` and plain JSON.", {"parser"}
        )
        assert "JsonParser" in out
        # But a specific identifier still needs its specific segments.
        out2, report2 = corrob.filter_ungrounded_prose(
            "Uses `JsonParser`.", {"parser", "json"}
        )
        assert out2 == "Uses `JsonParser`."
        assert not report2.touched

    def test_removed_identifiers_sorted_and_unique(self):
        text = (
            "`BetaThing` first. `AlphaThing` second. `BetaThing` again. "
            "Plain tail."
        )
        out, report = corrob.filter_ungrounded_prose(text, {"zzz"})
        assert out == "Plain tail."
        assert report.removed_identifiers == ["AlphaThing", "BetaThing"]

    def test_cyrillic_prose_with_english_identifier(self):
        text = (
            "Система выводит привет. "
            "Несуществующий класс `GhostWidgetFactory` отвечает за сборку."
        )
        out, report = corrob.filter_ungrounded_prose(text, {"привет"})
        assert "Система выводит привет." in out
        assert "GhostWidgetFactory" not in out
        assert report.removed_identifiers == ["GhostWidgetFactory"]


# ============================================================================
# build_identifier_grounding (codebase entity)
# ============================================================================
class TestBuildIdentifierGrounding:
    def test_reads_source_files_and_paths(self, tmp_path):
        (tmp_path / "svc.py").write_text("class LoginController:\n    pass\n")
        (tmp_path / "other.py").write_text("ORDERS_ARCHIVE = 1\n")
        out = corrob.build_identifier_grounding(
            str(tmp_path), ["svc.py"], repo_files=["svc.py", "docs/guide.md"]
        )
        assert "LoginController" in out
        assert "svc.py" in out and "docs/guide.md" in out
        assert "docs.guide.md" in out  # dotted variant for module mentions
        assert "ORDERS_ARCHIVE" not in out  # only the section's files are read

    def test_missing_and_escaping_files_skipped(self, tmp_path):
        (tmp_path / "outside.py").write_text("SHOULD_NOT_APPEAR = 1\n")
        out = corrob.build_identifier_grounding(
            str(tmp_path), ["ghost.py", "../outside.py", ""]
        )
        assert not out

    def test_caps(self, tmp_path):
        (tmp_path / "big.py").write_text("VALUE = 'x' * 10\n")
        out = corrob.build_identifier_grounding(
            str(tmp_path), ["big.py"], per_file_chars=3
        )
        # Only the first 3 chars ("VAL") survive the per-file cap.
        assert out == {"VAL"}


# ============================================================================
# grounding_from_introspection (database entity)
# ============================================================================
class TestGroundingFromIntrospection:
    def test_names_and_definition_identifiers(self):
        info = {
            "schemas": ["public"],
            "tables": {
                "public.users": {
                    "schema": "public",
                    "table": "users",
                    "definition": (
                        "CREATE TABLE users (id integer PRIMARY KEY, "
                        "role user_role)"
                    ),
                },
                "orders": {"schema": None, "table": "orders", "definition": ""},
            },
        }
        out = corrob.grounding_from_introspection(info)
        assert {"public", "public.users", "users", "orders"} <= out
        assert {"CREATE", "TABLE", "integer", "PRIMARY", "user_role"} <= out

    def test_tolerates_garbage(self):
        assert corrob.grounding_from_introspection(None) == set()
        assert corrob.grounding_from_introspection({"tables": "nope"}) == set()


# ============================================================================
# verify_section integration (grounding param)
# ============================================================================
class TestVerifySectionGrounding:
    def test_filters_and_reports(self, tmp_path):
        (tmp_path / "main.py").write_text("import os\nprint('hello')\n")
        result = asyncio.run(v.verify_section(
            "overview",
            "The module prints hello. Ghost `GhostWidgetFactory` builds it.",
            repo_dir=str(tmp_path),
            repo_files=["main.py"],
            source_files=["main.py"],
            run_judge=False,
            grounding={"print", "hello"},
        ))
        assert "GhostWidgetFactory" not in result.masked_content
        assert "module prints hello" in result.masked_content
        assert result.corroborate_removed == ["GhostWidgetFactory"]
        assert any("corroborate" in w for w in result.warnings)

    def test_no_grounding_passthrough(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        result = asyncio.run(v.verify_section(
            "qa", "Mentions `GhostWidgetFactory` anyway.",
            repo_dir=str(tmp_path), repo_files=["a.py"], source_files=["a.py"],
            run_judge=False,
        ))
        assert "GhostWidgetFactory" in result.masked_content
        assert result.corroborate_removed == []

    def test_fail_open_keeps_content(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        content = "Only `GhostWidgetFactory` here."
        result = asyncio.run(v.verify_section(
            "cicd", content,
            repo_dir=str(tmp_path), repo_files=["a.py"], source_files=["a.py"],
            run_judge=False,
            grounding={"zzz"},
        ))
        assert result.masked_content == content
        assert result.corroborate_removed == []
        assert any("original kept" in w for w in result.warnings)

    def test_citations_extracted_from_filtered_text(self, tmp_path):
        (tmp_path / "main.py").write_text("import os\n")
        result = asyncio.run(v.verify_section(
            "overview",
            # The fabricated identifier and a LEGIT grounded citation share
            # a sentence: dropping the sentence must drop the citation too.
            "`GhostWidgetFactory` lives in `main.py` somewhere. "
            "Plain tail survives.",
            repo_dir=str(tmp_path), repo_files=["main.py"], source_files=["main.py"],
            run_judge=False,
            grounding={"import", "main.py"},
        ))
        assert result.masked_content == "Plain tail survives."
        assert result.citations_resolved == []


# ============================================================================
# build_section_provenance integration (ungrounded param)
# ============================================================================
class TestProvenanceUngrounded:
    def test_key_present_when_removed(self):
        prov = v.build_section_provenance(
            "overview", model="m", prompt_file="f.md", source_files=[],
            fingerprint=None, citations={"resolved": [], "unresolved": []},
            judge=None, regen="generated", ungrounded=["GhostWidgetFactory"],
        )
        assert prov["corroborate"] == {"removed": ["GhostWidgetFactory"]}

    def test_key_absent_without_removals(self):
        prov = v.build_section_provenance(
            "overview", model="m", prompt_file="f.md", source_files=[],
            fingerprint=None, citations={"resolved": [], "unresolved": []},
            judge=None, regen="generated",
        )
        assert "corroborate" not in prov
