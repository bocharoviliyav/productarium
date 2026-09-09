"""Citation guard: REWRITE unresolvable file citations, not just warn.

Port of the fork's deterministic citation validator (``validators.py`` +
``fence_safe.py``), adapted to productarium's citation format (inline-code
`` `path` `` / `` `path:42-58` `` and bare ``path:42-58`` spans — the same
regexes :func:`api.docgen.verification.extract_citations` uses, so the guard
and the extractor can never disagree about what a citation is).

Policy change vs the Wave D "warn only" contract: a citation whose path is
NOT among the files the flow actually has (the repo clone for codebases) is
REMOVED from the persisted text, and a line span that cannot exist in the
cited file is stripped — citations turn from decorative into verified. Every
removal/fix is recorded in the guard report and lands in the section
provenance, so reviewers see exactly what was dropped.

Flows without a real file set (database reverse-engineering) keep the
warning-only behaviour: they simply never call the guard.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Set, Tuple

from api.docgen.verification import (
    _BARE_SPAN_RE,
    _CITATION_RE,
    _PATH_EXT_RE,
    normalize_repo_path,
)
from api.utils.fs import open_read_nofollow

__all__ = [
    "CitationGuardReport",
    "apply_outside_fences",
    "count_file_lines",
    "guard_citations",
]


def apply_outside_fences(markdown: str, transform: Callable[[str], str]) -> str:
    """Apply a text transform while leaving fenced blocks untouched.

    Port of the fork's ``fence_safe.apply_outside_fences``. The transform
    receives each outside-fence region as ONE string rather than line by
    line: the citation pass parses multi-line constructs and would break on
    fragments.
    """
    parts: List[str] = []
    buffer: List[str] = []
    in_fence = False

    def flush() -> None:
        if buffer:
            parts.append(transform("\n".join(buffer)))
            buffer.clear()

    for line in markdown.split("\n"):
        if line.lstrip().startswith("```"):
            if not in_fence:
                flush()
            parts.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            parts.append(line)
        else:
            buffer.append(line)
    flush()
    return "\n".join(parts)


@dataclass
class CitationGuardReport:
    """What the guard removed / fixed (raw tokens, for provenance)."""

    removed: List[str] = field(default_factory=list)
    fixed: List[str] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        return bool(self.removed or self.fixed)


def count_file_lines(repo_dir: str, rel_path: str) -> int:
    """Line count of ``repo_dir/rel_path`` (0 when unreadable/outside).

    Realpath-RESOLVED confinement (a symlink planted in the clone cannot
    make the guard read outside ``repo_dir``) and O_NOFOLLOW on the final
    component — the same hygiene as ``compute_file_hashes``.
    """
    repo_abs = os.path.realpath(repo_dir)
    full = os.path.realpath(os.path.normpath(os.path.join(repo_dir, rel_path)))
    try:
        inside = os.path.commonpath([repo_abs, full]) == repo_abs
    except ValueError:
        return 0
    if not inside:
        return 0
    try:
        with open_read_nofollow(full, binary=False) as f:
            return len(f.read().splitlines())
    except (OSError, ValueError, UnicodeError):
        return 0


def _looks_like_path(path: str) -> bool:
    """Same path-shape filter as :func:`extract_citations` (no bare words)."""
    if path.startswith(("http://", "https://", "ftp://")):
        return False
    return "/" in path or bool(_PATH_EXT_RE.search(path))


def _span_is_valid(span: str, max_line: int) -> bool:
    lo, _, hi = span.partition("-")
    try:
        low = int(lo)
        high = int(hi) if hi else low
    except ValueError:
        return False
    # Unknown line count (max_line == 0) fails CLOSED: the span cannot be
    # verified, so it is stripped (the fork's behaviour).
    if max_line <= 0:
        return False
    return 1 <= low <= high <= max_line


def _rewrite_region(
    region: str,
    allowed: Set[str],
    line_counts: Dict[str, int],
    report: CitationGuardReport,
) -> str:
    """Rewrite one outside-fence region: drop/fold citations in order."""
    matches = sorted(
        [(m.start(), m.end(), m) for m in _CITATION_RE.finditer(region)]
        + [(m.start(), m.end(), m) for m in _BARE_SPAN_RE.finditer(region)],
        key=lambda t: t[0],
    )
    out: List[str] = []
    pos = 0
    for start, end, m in matches:
        if start < pos:
            continue  # overlapping (e.g. a bare span inside a code span)
        path = m.group("path").strip("/")
        if not path or not _looks_like_path(path):
            continue  # plain backticked word / URL — not a citation
        norm = normalize_repo_path(path)
        if norm not in allowed:
            report.removed.append(m.group(0))
            out.append(region[pos:start])
            # Swallow ONE trailing separator space so prose does not end up
            # with doubled gaps ("entry `gone.py` point" → "entry point").
            if end < len(region) and region[end] == " ":
                end += 1
            pos = end
            continue
        span = m.group("span")
        if span and not _span_is_valid(span, line_counts.get(norm, 0)):
            report.fixed.append(f"{norm} (lines {span} stripped)")
            out.append(region[pos:start])
            # Keep the (verified) path, drop only the span. Backticked
            # citations keep their backticks; bare ones stay bare.
            if m.re is _CITATION_RE:
                out.append(f"`{path}`")
            else:
                out.append(path)
            pos = end
    out.append(region[pos:])
    return "".join(out)


def guard_citations(
    markdown: str,
    allowed_files: Set[str],
    file_line_counts: Dict[str, int],
) -> Tuple[str, CitationGuardReport]:
    """Remove unresolvable citations and impossible line spans.

    ``allowed_files`` / ``file_line_counts`` are keyed by NORMALIZED repo
    paths (see :func:`normalize_repo_path`). Code fences are never touched.
    """
    report = CitationGuardReport()
    if not markdown or not allowed_files:
        return markdown, report
    cleaned = apply_outside_fences(
        markdown,
        lambda region: _rewrite_region(region, allowed_files, file_line_counts, report),
    )
    return cleaned, report


def line_counts_for_citations(
    repo_dir: str,
    citations: Sequence,
) -> Dict[str, int]:
    """Line counts for the ALLOWED cited paths only (cheap: few files)."""
    out: Dict[str, int] = {}
    for cite in citations or []:
        norm = normalize_repo_path(cite.path)
        if norm not in out:
            out[norm] = count_file_lines(repo_dir, norm)
    return out
