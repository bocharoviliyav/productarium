"""Unit tests for api.docgen.fact_fold (Wave 1.4 port from the fork) + its
application in the database skeleton (tables) and spec renderers (schemas).

Covers:
- ``fold``: empty body → no block, title escaping, ``count`` override,
  ``</details>`` neutralised in the body (case-insensitive), the
  load-bearing blank line after ``</summary>``.
- ``rank_split`` / ``is_conserved``: the (visible, hidden) pair conserves
  every item with multiplicity; ``keep < 1`` rejected; exploding key
  degrades to input order.
- Database ``_render_skeleton``: ≤ 12 tables → no disclosure (byte-equal to
  the old render); > 12 → the rest under ONE disclosure, count = hidden
  TABLES (not lines), every table still on the page.
- Spec ``_render_schemas_folded``: top-10 by field count visible; small
  specs unchanged.
- Frontend smoke: the Markdown component's sanitize schema still extends
  rehype-sanitize's ``defaultSchema`` (details/summary survive it).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import api.docgen.fact_fold as ff


# ===========================================================================
# fold
# ===========================================================================
class TestFold:
    def test_empty_body_yields_no_block(self):
        assert ff.fold("X", []) == []

    def test_structure_and_blank_line_after_summary(self):
        lines = ff.fold("Остальные таблицы", ["### `a`", "", "row"])
        assert lines[0] == "<details>"
        assert lines[1] == "<summary>Остальные таблицы (3)</summary>"
        assert lines[2] == ""  # load-bearing: GFM tables inside need it
        assert lines[-1] == "</details>"

    def test_count_overrides_line_count(self):
        lines = ff.fold("Tables", ["l1", "l2", "l3", "l4"], count=40)
        assert "(40)" in lines[1]

    def test_title_escaped(self):
        lines = ff.fold("A & <b>", ["x"])
        assert "<summary>A &amp; &lt;b&gt; (1)</summary>" in lines

    def test_closing_tag_neutralised(self):
        lines = ff.fold("T", ["ddl line", "</details>", "</DETAILS >", "</ details>"])
        body = "\n".join(lines)
        # Only the block's OWN closing tag remains as a raw tag.
        assert body.count("</details>") == 1
        assert "&lt;/details&gt;" in body

    def test_count_none_uses_len(self):
        lines = ff.fold("T", ["a", "b"])
        assert "(2)" in lines[1]


# ===========================================================================
# rank_split / is_conserved
# ===========================================================================
class TestRankSplit:
    def test_conservation_with_multiplicity(self):
        items = ["a", "b", "a", "c", "b", "a"]
        visible, hidden = ff.rank_split(items, key=lambda x: 0, keep=2)
        assert ff.is_conserved(items, visible, hidden)
        assert list(visible) == ["a", "b"]  # stable input order

    def test_hidden_continues_ranking(self):
        items = [3, 1, 2]
        visible, hidden = ff.rank_split(items, key=lambda x: x, keep=2)
        assert visible == (1, 2)
        assert hidden == (3,)

    def test_keep_below_one_rejected(self):
        try:
            ff.rank_split([1], key=lambda x: x, keep=0)
        except ValueError:
            pass
        else:
            raise AssertionError("keep=0 must raise")

    def test_empty(self):
        assert ff.rank_split([], key=lambda x: x, keep=3) == ((), ())

    def test_exploding_key_falls_back_to_input_order(self):
        def boom(x):
            raise RuntimeError("bad key")

        visible, hidden = ff.rank_split([5, 3, 1], key=boom, keep=2)
        assert visible == (5, 3)
        assert hidden == (1,)

    def test_is_conserved_catches_lost_duplicate(self):
        assert not ff.is_conserved(["a", "a"], ["a"], [])
        assert ff.is_conserved(["a", "a"], ["a"], ["a"])


# ===========================================================================
# database skeleton: tables fold
# ===========================================================================
class _Entity:
    name = "TestDB"
    dsn_masked = "postgres://u:***@h/db"


def _db_info(n_tables: int, definition: str = "CREATE TABLE x (id int)") -> dict:
    return {
        "schemas": [],
        "tables": {
            f"table_{i:03d}": {"schema": None, "table": f"table_{i:03d}",
                               "definition": definition}
            for i in range(n_tables)
        },
    }


class TestDatabaseTablesFold:
    """Post-restructure contract: the tables ROOT lists ranked names only
    (descriptions/relations/ER live there, DDL on per-table subpages) — the
    fold hides surplus NAME rows, never structure blocks."""

    def test_small_schema_no_disclosure(self):
        from api.docgen.database import _render_skeleton

        _, tables_md = _render_skeleton(_Entity(), _db_info(5))
        assert "<details>" not in tables_md
        for i in range(5):
            assert f"- `table_{i:03d}`" in tables_md
        # The root lists names only — DDL lives on subpages.
        assert "CREATE TABLE" not in tables_md

    def test_large_schema_folds_with_count(self):
        from api.docgen.database import _TABLES_ROOT_VISIBLE, _render_skeleton

        n = _TABLES_ROOT_VISIBLE + 20
        _, tables_md = _render_skeleton(_Entity(), _db_info(n))
        assert "<details>" in tables_md
        assert (
            f"<summary>Remaining tables ({n - _TABLES_ROOT_VISIBLE})</summary>"
            in tables_md
        )
        # Conservation: EVERY table is still on the page (60 visible + 20 folded).
        for i in range(n):
            assert f"- `table_{i:03d}`" in tables_md
        # The disclosure comes after the visible ones.
        first_hidden = tables_md.index(f"- `table_{_TABLES_ROOT_VISIBLE:03d}`")
        details_at = tables_md.index("<details>")
        assert details_at < first_hidden

    def test_definitions_live_on_subpages_not_root(self):
        # The restructure moved DDL/structure into per-table subpages: the
        # tables ROOT is names + relations + ER only, so raw DDL (and any
        # stray ``</details>`` inside it) can never break a fold.
        from api.docgen.database import _render_skeleton

        info = _db_info(3, definition="CREATE TABLE x (id int); -- </details>")
        _, tables_md = _render_skeleton(_Entity(), info)
        assert "CREATE TABLE" not in tables_md
        assert "</details>" not in tables_md
        for i in range(3):
            assert f"- `table_{i:03d}`" in tables_md

    def test_rows_conserved(self):
        from api.docgen.database import _TABLES_ROOT_VISIBLE, _render_skeleton

        n = _TABLES_ROOT_VISIBLE + 8
        _, tables_md = _render_skeleton(_Entity(), _db_info(n))
        names = sorted(f"table_{i:03d}" for i in range(n))
        head, _, tail = tables_md.partition("<details>")
        visible = [ln for ln in head.splitlines() if ln.startswith("- `")]
        hidden = [ln for ln in tail.splitlines() if ln.startswith("- `")]
        # What a reader counts as facts: the rendered name rows.
        assert ff.is_conserved([f"- `{f}`" for f in names], visible, hidden)
        assert len(hidden) == 8


# ===========================================================================
# spec renderers: schemas fold
# ===========================================================================
def _spec_with_schemas(n: int) -> dict:
    schemas = {}
    for i in range(n):
        schemas[f"Model{i:02d}"] = {
            "type": "object",
            "properties": {
                f"field_{j}": {"type": "string"} for j in range((i % 5) + 1)
            },
        }
    return {
        "info": {"title": "API", "version": "1.0"},
        "openapi": "3.0.0",
        "paths": {},
        "components": {"schemas": schemas},
    }


class TestSpecSchemasFold:
    def test_small_spec_unchanged(self):
        from api.docgen.spec import _render_openapi_skeleton

        md = _render_openapi_skeleton(_spec_with_schemas(3))
        assert "<details>" not in md
        assert "### Model00" in md

    def test_large_spec_folds_rest(self):
        from api.docgen.spec import _SCHEMAS_VISIBLE, _render_openapi_skeleton

        md = _render_openapi_skeleton(_spec_with_schemas(25))
        assert f"<summary>Остальные схемы ({25 - _SCHEMAS_VISIBLE})</summary>" in md
        for i in range(25):  # conservation: every schema still present
            assert f"### Model{i:02d}" in md

    def test_asyncapi_folds_too(self):
        from api.docgen.spec import _render_asyncapi_skeleton

        spec = _spec_with_schemas(15)
        spec["asyncapi"] = "2.6.0"
        spec.pop("openapi")
        spec["channels"] = {}
        md = _render_asyncapi_skeleton(spec)
        assert "<details>" in md

    def test_most_fields_first(self):
        from api.docgen.spec import _SCHEMAS_VISIBLE, _render_schemas_folded

        schemas = {
            "Small": {"properties": {"a": {}}},
            "Huge": {"properties": {f"f{i}": {} for i in range(20)}},
        }
        md = "\n".join(_render_schemas_folded(schemas))
        assert md.index("### Huge") < md.index("### Small")
        # Both visible — under the keep limit there is no fold at all.
        assert "<details>" not in md
        assert len(schemas) <= _SCHEMAS_VISIBLE


# ===========================================================================
# frontend smoke: details/summary survive the sanitize schema
# ===========================================================================
class TestFrontendSanitizeSmoke:
    def test_markdown_component_extends_default_schema(self):
        """`<details>`/`<summary>` are in rehype-sanitize's defaultSchema; the
        frontend only ADDS attributes on top. If the component ever swaps the
        base schema, disclosures would start rendering as literal text."""
        src = (project_root / "src" / "components" / "Markdown.tsx").read_text(
            encoding="utf-8"
        )
        assert "defaultSchema" in src
        assert "...defaultSchema" in src
        assert "rehypeSanitize" in src
