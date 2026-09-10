"""Corroborate: anti-fabrication identifier filter for generated prose.

Port of the fork's ``api/generation/corroborate.py``
(``filter_ungrounded_sentences`` + the grounding index), adapted to
productarium's flows:

- **Fence-safe** — the filter runs through
  :func:`api.docgen.citation_guard.apply_outside_fences`, so code blocks
  (``` ``` ```, ``` ```sql ```) are never rewritten.
- **Reported, not silent** — every dropped identifier lands in
  :class:`CorroborateReport` and from there in the section provenance
  (``provenance["corroborate"]["removed"]``).
- **Fail-open** — when the filter would empty the ENTIRE text, the original
  is kept (a broken grounding set must never destroy a section).
- **ALLCAPS narrowed to SCREAMING_SNAKE** — the fork also treated any 3+
  letter ALLCAPS word as a code token; our sections are Russian prose where
  Latin ALLCAPS is common emphasis ("RUN ONE section body…"), so only
  underscore-joined enums/constants (``CASH_CARD``, ``QRTZ_JOB``) count as
  code-like. Single-word acronyms that matter are covered by
  ``_ALWAYS_GROUNDED`` below.

Grounding evidence is entity-typed:
- codebase — identifiers from the CONTENT of the section's source files +
  the repo's file paths (so `` `main.py` `` cites and dotted module mentions
  ground), built by :func:`build_identifier_grounding`;
- database — schema/table names + identifiers from the introspected
  definitions, built by :func:`grounding_from_introspection`.

Pure & stdlib-only (re + dataclass + os for the bounded file scan); the
module never touches the network and never raises past its public API.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from api.utils.fs import open_read_nofollow

__all__ = [
    "CorroborateReport",
    "build_identifier_grounding",
    "filter_ungrounded_prose",
    "grounding_from_introspection",
    "grounding_index",
]


# Code-like tokens whose grounding is checked in prose: CamelCase
# (OrderService), SCREAMING_SNAKE enums/constants (CASH_CARD, QRTZ_JOB) and
# dotted identifiers (com.example.app, public.users, main.py).
_CODE_TOKEN_RE = re.compile(
    r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"      # CamelCase
    r"|[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+"         # SCREAMING_SNAKE only
    r"|[a-z][a-z0-9]*\.[a-z][\w.]*)\b"           # dotted identifiers
)

# Identifiers (>= 3 chars) harvested from grounding evidence text.
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Common technical vocabulary that is never a *specific* identifier: it
# grounds everywhere so CamelCase segments like ``Json…``/``Api…`` and
# acronyms used in prose do not trip the filter. Lowercase.
_ALWAYS_GROUNDED = frozenset({
    "redacted",  # the secret-mask placeholder must never look fabricated
    "api", "sql", "ddl", "dml", "json", "yaml", "xml", "html", "css",
    "http", "https", "url", "uri", "uuid", "orm", "crud", "acid", "sdk",
    "cli", "gui", "grpc", "rest", "jwt", "oauth", "cors", "sso", "rbac",
})


def grounding_index(tokens: Iterable[str]) -> Set[str]:
    """Lowercase grounding index: each token plus its dot/underscore/camel
    segments, so a class named ``LoginController`` grounds when the evidence
    carries ``login`` or ``LoginController`` (port of the fork's
    ``_grounding_index``).
    """
    idx: Set[str] = set()
    for t in tokens:
        if not t:
            continue
        low = t.lower()
        idx.add(low)
        for seg in re.split(r"[._]", low):
            if len(seg) > 2:
                idx.add(seg)
        for seg in re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", t):
            if len(seg) > 2:
                idx.add(seg.lower())
    return idx


@dataclass
class CorroborateReport:
    """What the corroborate filter dropped (for provenance/logging)."""

    sentences_removed: int = 0
    emptied: bool = False
    _removed: Set[str] = field(default_factory=set, repr=False)

    @property
    def removed_identifiers(self) -> List[str]:
        """Sorted unique identifiers whose sentences were dropped."""
        return sorted(self._removed)

    @property
    def touched(self) -> bool:
        return bool(self._removed)


def _ungrounded_tokens(sentence: str, idx: Set[str]) -> List[str]:
    """Code-like tokens in ``sentence`` that are NOT grounded by ``idx``.

    Segment grounding is SYMMETRIC with :func:`grounding_index`: the checked
    token is split on dot/underscore/camel boundaries the same way, so
    ``CASH_CARD`` in prose grounds when the evidence carries ``cash`` and
    ``card`` — exactly as ``LoginController`` grounds via its camel parts.
    """
    out: List[str] = []
    for tok in _CODE_TOKEN_RE.findall(sentence):
        low = tok.lower()
        if low in idx:
            continue
        segs: Set[str] = set()
        for seg in re.split(r"[._]", tok):
            if len(seg) > 2:
                segs.add(seg.lower())
        for seg in re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", tok):
            if len(seg) > 2:
                segs.add(seg.lower())
        if segs and all(s in idx for s in segs):
            continue
        out.append(tok)
    return out


def _filter_region(region: str, idx: Set[str], report: CorroborateReport) -> str:
    """Filter one outside-fence region (port of the fork's paragraph walk)."""
    if not region.strip():
        return region
    out_paras: List[str] = []
    for para in re.split(r"\n\s*\n", region):
        # keep non-prose blocks (bullets, headings, quotes, tables) untouched
        stripped = para.lstrip()
        if stripped[:1] in ("-", "*", "#", ">", "|") or not stripped:
            out_paras.append(para)
            continue
        kept: List[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", para):
            ungrounded = _ungrounded_tokens(sent, idx)
            if ungrounded:
                report.sentences_removed += 1
                report._removed.update(ungrounded)
            else:
                kept.append(sent)
        if kept:
            out_paras.append(" ".join(kept))
        # a paragraph whose every sentence was dropped disappears entirely
    return "\n\n".join(out_paras)


def filter_ungrounded_prose(
    markdown: str, grounding: Iterable[str]
) -> Tuple[str, CorroborateReport]:
    """Drop sentences that name code-like identifiers absent from ``grounding``.

    Plain-language sentences with no code tokens pass unchanged, so the
    narrative stays rich; only invented identifiers are removed. Fenced code
    blocks, bullets, headings, blockquotes and tables are never touched.
    Empty/falsy ``grounding`` is a passthrough (no evidence → no claim of
    fabrication). Fail-open: a filter result that would empty the whole text
    returns the original with ``report.emptied = True``.
    """
    report = CorroborateReport()
    if not (markdown or "").strip():
        return markdown, report
    tokens = {t for t in (grounding or ()) if t}
    if not tokens:
        return markdown, report
    idx = grounding_index(tokens) | _ALWAYS_GROUNDED

    from api.docgen.citation_guard import apply_outside_fences

    cleaned = apply_outside_fences(
        markdown, lambda region: _filter_region(region, idx, report)
    )
    if not cleaned.strip():
        # Fail-open: the grounding set is clearly unusable — keep the text.
        return markdown, CorroborateReport(emptied=True)
    return cleaned, report


# --------------------------------------------------------------------------- #
# Grounding evidence builders (entity-typed)
# --------------------------------------------------------------------------- #
def grounding_from_text(*texts: Optional[str]) -> Set[str]:
    """Identifiers (>= 3 chars) harvested from free-form evidence text."""
    tokens: Set[str] = set()
    for text in texts:
        if text:
            tokens.update(_WORD_RE.findall(text))
    return tokens


def build_identifier_grounding(
    repo_dir: str,
    rel_files: Sequence[str],
    *,
    repo_files: Sequence[str] = (),
    max_files: int = 20,
    per_file_chars: int = 8_000,
    total_chars: int = 100_000,
) -> Set[str]:
    """Codebase grounding: identifiers from the section's SOURCE FILES plus
    the repo's file paths (a cited `` `main.py` `` and dotted module mentions
    must ground even when the file's content vocabulary differs).

    Bounded: at most ``max_files`` files, ``per_file_chars`` per file and
    ``total_chars`` overall. Realpath-RESOLVED confinement inside
    ``repo_dir`` + O_NOFOLLOW on the final component — the same hygiene as
    ``compute_file_hashes`` / ``count_file_lines``. Unreadable/escaping
    files are skipped, never fatal.
    """
    tokens: Set[str] = set()
    repo_abs = os.path.realpath(repo_dir or "")
    total = 0
    for rel in list(rel_files or [])[:max_files]:
        if not isinstance(rel, str) or not rel:
            continue
        full = os.path.realpath(os.path.normpath(os.path.join(repo_dir, rel)))
        try:
            inside = os.path.commonpath([repo_abs, full]) == repo_abs
        except ValueError:
            inside = False
        if not inside or not os.path.isfile(full):
            continue
        try:
            with open_read_nofollow(full, binary=False, errors="replace") as f:
                text = f.read(per_file_chars)
        except (OSError, ValueError, UnicodeError):
            continue
        tokens.update(_WORD_RE.findall(text))
        total += len(text)
        if total >= total_chars:
            break
    for p in repo_files or []:
        norm = str(p).strip("/")
        if norm:
            tokens.add(norm)
            tokens.add(norm.replace("/", "."))
    return tokens


def grounding_from_introspection(info: Optional[Dict[str, Any]]) -> Set[str]:
    """Database grounding: schema/table names + identifiers from the
    introspected payload (``_introspect``): schemas, per-table definitions
    (columns live inside them) and — since the DB RE restructure — the FK
    edge names and every category collection (views, triggers, routines,
    sequences, types with their sources/meta).

    Function/trigger/column names live inside the definitions, so the word
    harvest over them grounds exactly the names the MCP surface reported.
    Every collection is read defensively: an older cached payload (or a
    hand-built fixture) without a collection simply contributes nothing.
    """
    tokens: Set[str] = set()
    if not isinstance(info, dict):
        return tokens
    for schema in info.get("schemas") or []:
        if isinstance(schema, str) and schema:
            tokens.add(schema)
    tables = info.get("tables") or {}
    if isinstance(tables, dict):
        for full, meta in tables.items():
            tokens.add(str(full))
            if isinstance(meta, dict):
                bare = meta.get("table")
                if isinstance(bare, str) and bare:
                    tokens.add(bare)
                definition = meta.get("definition") or ""
                if isinstance(definition, str):
                    tokens.update(_WORD_RE.findall(definition))
    for edge in info.get("fk_edges") or []:
        if isinstance(edge, dict):
            for key in ("from", "to", "constraint"):
                value = edge.get(key)
                if isinstance(value, str) and value:
                    tokens.update(_WORD_RE.findall(value))
    for collection in (
        "views", "triggers", "routines", "sequences", "types",
    ):
        entries = info.get(collection) or {}
        if not isinstance(entries, dict):
            continue
        for full, meta in entries.items():
            tokens.add(str(full))
            if isinstance(meta, dict):
                bare = meta.get("name")
                if isinstance(bare, str) and bare:
                    tokens.add(bare)
                source = meta.get("source") or meta.get("definition") or ""
                if isinstance(source, str):
                    tokens.update(_WORD_RE.findall(source))
                meta_blob = meta.get("meta")
                if isinstance(meta_blob, dict):
                    for value in meta_blob.values():
                        if isinstance(value, str):
                            tokens.update(_WORD_RE.findall(value))
    return tokens
