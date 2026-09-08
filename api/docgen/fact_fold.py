"""Disclosure blocks for deterministic renders: hide surplus facts, never drop.

Port of the fork's ``fact_fold.py`` + ``fact_rank.py``. The reader
contradiction this resolves: an analyst cannot read 300 table definitions in
a row, and a working engineer needs exactly those 300. The rule is «не
резать, а прятать» — the selection goes on top, the remainder goes under a
``<details>`` disclosure, and the page keeps every fact it had. This matters
most for database docs: a schema can be hundreds of megabytes of DDL with
thousands of tables.

**The blank lines are load-bearing.** Measured through the real frontend
pipeline (remark-parse -> remark-gfm -> remark-rehype -> rehype-raw ->
rehype-sanitize, the plugin chain of ``src/components/Markdown.tsx``): with
a blank line after ``</summary>`` a GFM table inside the block becomes a
real ``<table>``; without it the disclosure still opens and the table stays
literal pipes. ``<details>``/``<summary>`` survive sanitisation because they
are in rehype-sanitize's ``defaultSchema``, which the frontend schema
extends.

**Every function here is total.** A renderer that raises would trade
completeness for decoration; unexpected input degrades to input order.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

__all__ = ["fold", "is_conserved", "rank_split"]

T = TypeVar("T")


def _escape_title(text: object) -> str:
    """Ampersand first, or the replacements escape each other's output."""
    return (str(text if text is not None else "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


_CLOSING_TAG = re.compile(r"</\s*details\s*>", re.IGNORECASE)


def _neutralise(line: object) -> str:
    """The body is Markdown, not plain text: escaping it wholesale would
    destroy tables, code spans and links. Only the closing tag is
    neutralised — arbitrary source text (a DDL body, a table comment)
    carrying it would end the disclosure early and spill the rest of the
    page into the open. Case-insensitive: a literal ``</DETAILS>`` closes
    the block exactly as ``</details>`` does.
    """
    return _CLOSING_TAG.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), str(line))


def fold(title: str, hidden_body: List[str], *, count: Optional[int] = None) -> List[str]:
    """Wrap ``hidden_body`` in a ``<details>`` disclosure with a count.

    ``count`` overrides the line count when the unit of hiding is coarser
    than a line — 40 folded tables are ~200 lines, and the reader is told
    about TABLES. Derived here rather than interpolated into ``title`` by
    the caller, so summary and body cannot drift apart silently.

    An empty body yields no block at all: an empty disclosure is noise.
    """
    if not hidden_body:
        return []
    shown = len(hidden_body) if count is None else count
    return [
        "<details>",
        f"<summary>{_escape_title(title)} ({shown})</summary>",
        "",
        *(_neutralise(line) for line in hidden_body),
        "",
        "</details>",
    ]


def rank_split(
    items: Sequence[T],
    *,
    key: Callable[[T], object],
    keep: int,
) -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """Top-``keep`` by ``key`` (ascending — invert the key for "most first");
    the rest, still ranked, is returned as ``hidden``.

    Both contracts return ``(visible, hidden)`` rather than a truncated
    list, and that pair IS the positive control: a selector that silently
    threw everything away shows up as an empty ``visible`` beside a
    non-empty ``hidden``. ``hidden`` continues the ranking, so a reader who
    opens the disclosure meets the next item first. The sort is stable.

    ``keep < 1`` is rejected rather than treated as "select nothing".
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    if not items:
        return (), ()
    try:
        ordered = sorted(items, key=key)
    except Exception:
        # A key that explodes must not take the section down with it: fall
        # back to input order — a worse ranking, but a whole page.
        ordered = list(items)
    return tuple(ordered[:keep]), tuple(ordered[keep:])


def is_conserved(
    items: Sequence[T],
    visible: Sequence[T],
    hidden: Sequence[T],
) -> bool:
    """Conservation by identity AND multiplicity (a set would lose a duplicate
    fact — two tables that render alike are still two facts)."""
    return Counter(items) == Counter(visible) + Counter(hidden)
