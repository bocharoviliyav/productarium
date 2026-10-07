"""Direct tests for ``collect_entity_parts`` (shared by summary + HLD).

Covers the edges the pipeline tests don't: name/id fallbacks, spec kind
default, links JSON handling and the joined-budget invariant.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.docgen._common import collect_entity_parts


class _Cb:
    def __init__(self, name=None, docs="d"):
        self.id = "cb_1"
        self.name = name
        self.generated_docs = docs


class _Spec:
    def __init__(self, name=None, kind=None, content="spec body"):
        self.id = "spec_1"
        self.name = name
        self.kind = kind
        self.content = content


class _Links:
    def __init__(self, content):
        self.id = "links_1"
        self.name = None
        self.content = content


class TestCollectEntityParts:
    def test_name_falls_back_to_id(self):
        parts = collect_entity_parts([_Cb(name=None, docs="docs")])
        assert parts == ["## Codebase: cb_1\n\ndocs"]

    def test_spec_kind_defaults(self):
        parts = collect_entity_parts(specs=[_Spec(kind=None)])
        assert "## Спецификация (spec): spec_1" in parts[0]

    def test_links_valid_json_items(self):
        parts = collect_entity_parts(
            links=[_Links('[{"url": "https://a", "description": "desc"}]')]
        )
        assert "- https://a: desc" in parts[0]

    def test_links_malformed_json_skipped(self):
        assert collect_entity_parts(links=[_Links("[not json")]) == []

    def test_links_non_array_passthrough(self):
        parts = collect_entity_parts(links=[_Links("plain text")])
        assert "plain text" in parts[0]

    def test_budget_bounds_joined_total(self):
        # 30 entities with a 20k budget: the 1.5k floor would inflate the
        # joined context to ~45k — the sum must stay within budget_chars.
        cbs = [_Cb(name=f"c{i}", docs="x" * 50_000) for i in range(30)]
        parts = collect_entity_parts(cbs, budget_chars=20_000)
        assert len(parts) == 30
        # Slack covers the per-part truncation marker appended by _cap.
        assert sum(len(p) for p in parts) <= 20_000 + 30 * 100

    def test_floor_keeps_small_sets_rich(self):
        cbs = [_Cb(name=f"c{i}", docs="x" * 50_000) for i in range(4)]
        parts = collect_entity_parts(cbs, budget_chars=20_000)
        assert all(len(p) >= 5_000 - 100 for p in parts)

    def test_empty_inputs(self):
        assert collect_entity_parts() == []
