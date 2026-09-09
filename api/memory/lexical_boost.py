"""Lexical boost for expert recall (fork port: ladder-Ask exact-identifier resolve).

Semantic cosine recall loses exact identifiers: an embedding of "how is the
``audit_log`` table arranged" drifts toward generic database prose, while the
chunk that literally names ``audit_log`` may sit at rank 40. The fork solved
this with a lexical exact-resolve stage layered OVER the semantic ranking
(``graph_query.py``: identifiers, HTTP paths, aliases resolved textually
before the graph walk). Productarium has no graph, so the port is:

1. extract the code-like tokens of the question (backticked names are
   documentation's way of marking exact names — trusted as-is; bare words
   must look like compound identifiers: camel / snake / dotted / path);
2. run a bounded exact-text scan over ``knowledge_chunks.content``
   (``ILIKE ANY`` on the top tokens — Postgres-only, pgvector backend);
3. merge the lexical ranking with the cosine ranking via Reciprocal Rank
   Fusion so agreement (a chunk that is BOTH semantically close and contains
   the identifier) wins, and exact-name hits that cosine missed still
   surface near the top.

One layer, one flag (``EXPERT_LEXICAL_BOOST``, default on) for
measurability: set it off and the backend degrades to pure cosine. The
textual stage is bounded by the same ``memory_query`` timeout registry key
as the cosine stage and degrades to "no lexical hits" on ANY error, so the
boost can never take the expert recall down with it. Stdlib-only.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

__all__ = [
    "escape_ilike",
    "extract_query_tokens",
    "lexical_boost_enabled",
    "rank_lexical_rows",
    "reciprocal_rank_fusion",
]

_TRUTHY = ("1", "true", "yes", "on")

# Backticked span in a question marks an exact name (file path, table,
# function). Mirrors the identifiers-layer regex; re-declared locally to keep
# this module stdlib-only and independent of the embedder-boundary layer.
_BACKTICK = re.compile(r"`([^`\n]{2,80})`")


def lexical_boost_enabled() -> bool:
    """``EXPERT_LEXICAL_BOOST`` (default true, read on every call)."""
    raw = (os.environ.get("EXPERT_LEXICAL_BOOST", "true") or "").strip().lower()
    return raw in _TRUTHY


def extract_query_tokens(query: str, *, limit: int = 6) -> List[str]:
    """Code-like tokens of ``query`` for the exact-text scan.

    Backticked spans are trusted as exact names even when single-word (a bare
    table name like `` `orders` `` is precisely the case the boost exists
    for); bare words go through the identifiers layer, which keeps only
    compound tokens (camel/snake/dotted) so human prose never reaches the
    ILIKE stage. Deduplicated case-insensitively, capped at ``limit``.
    """
    try:
        from api.utils.identifiers import extract_identifiers
    except Exception:  # pragma: no cover - stdlib-only module
        return []
    text = query or ""
    out: List[str] = []
    seen: set = set()

    def _add(tok: str) -> None:
        tok = (tok or "").strip()
        if len(tok) < 3 or tok.lower() in seen:
            return
        seen.add(tok.lower())
        out.append(tok)

    for m in _BACKTICK.finditer(text):
        _add(m.group(1))
        if len(out) >= limit:
            return out
    for tok in extract_identifiers(text, limit=limit * 3):
        _add(tok)
        if len(out) >= limit:
            break
    return out[:limit]


def escape_ilike(token: str) -> str:
    """Escape a token for a Postgres ``ILIKE`` pattern (backslash escape).

    ``ILIKE ANY (:toks)`` uses the default backslash escape character, so
    ``%``/``_``/``\\`` inside an identifier must not act as wildcards —
    ``user_session_log`` would otherwise match any single-char gaps.
    """
    return (
        (token or "")
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def rank_lexical_rows(
    rows: Sequence[str], tokens: Sequence[str], *, top_k: int
) -> List[str]:
    """Order lexical candidates by total occurrences of the tokens.

    The SQL stage only filters (``ILIKE ANY``); this re-ranks the candidates
    by how often the question's tokens actually occur, so a chunk naming the
    token five times beats one with a single passing mention. Stable sort:
    candidates tied on score keep their SQL order (``ORDER BY id``), which
    keeps the result deterministic.
    """
    lows = [t.lower() for t in tokens if t]
    if not rows or not lows:
        return []
    scores: Dict[str, int] = {}
    for row in rows:
        low = (row or "").lower()
        scores[row] = sum(low.count(t) for t in lows)
    return sorted(rows, key=lambda r: -scores.get(r, 0))[: max(0, top_k)]


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[str], k: int = 60, limit: Optional[int] = None
) -> List[str]:
    """Merge several ranked lists into one (standard RRF, k=60).

    Each list contributes ``1 / (k + rank)`` per item; an item present in
    both the cosine and the lexical ranking out-scores anything present in
    only one. Content strings are the identities (chunks are joined into
    the final context verbatim). Ties break on the content itself so the
    merge is deterministic.
    """
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst or [], start=1):
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out = [item for item, _ in ordered]
    return out[:limit] if limit is not None else out
