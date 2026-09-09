"""Identifier layer (port of the fork's "layer A", design §5.6).

The point of the layer is lexical overlap between a human-language question
and a code identifier: ``sendSmsCode`` -> ``send sms code``. Identifiers are
kept VERBATIM as well (layer C), otherwise a literal search for
``SmsGatewayFacade`` would stop working.

Productarium has no knowledge graph; the layer is applied at the EMBEDDING
boundary instead: the text sent to the embedder is NOT the stored chunk —
it is the chunk plus a bounded suffix of its identifiers in both forms
(verbatim + human). Chunk-local extraction keeps it entity-agnostic: a
codebase doc chunk carries file/class/function names, a database doc chunk
carries table/function names, with no per-flow plumbing.
"""
from __future__ import annotations

import re
from typing import List

__all__ = [
    "build_embed_payload",
    "build_query_payload",
    "extract_identifiers",
    "split_identifier",
]

# Fork port: camel/Pascal boundaries. Two lookbehind/lookahead pairs cover
# "sendSmsCode" (lower|digit -> Upper) and "APIGateway" (Upper -> Upper+lower).
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Fork port: identifier separators — dots, underscores, slashes, hashes, dashes.
_SEPARATORS = re.compile(r"[._/#\-]+")

# Bare identifier-shaped word inside prose/code (letters, digits, _).
_BARE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,63}")
# Backticked span in markdown (`` `src/main.py` ``, `` `user_accounts` ``).
_BACKTICK = re.compile(r"`([^`\n]{2,80})`")
# Markdown heading text (a chunk often opens with the section title).
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def split_identifier(name: str) -> str:
    """Human form of an identifier: ``sendSmsCode`` → ``send sms code``."""
    words = _SEPARATORS.sub(" ", name or "")
    words = _CAMEL.sub(" ", words)
    return " ".join(words.lower().split())


def _keep_token(token: str) -> bool:
    """True for tokens with >= 2 word parts (camel/snake/dotted/path)."""
    token = token.strip()
    if not (2 <= len(token) <= 80):
        return False
    return len(split_identifier(token).split()) >= 2


def extract_identifiers(text: str, *, limit: int = 24) -> List[str]:
    """Identifier-like tokens of ``text`` (backticked first, then bare).

    Backticked spans are documentation's way of marking exact names — file
    paths, table names, functions — so they are trusted as-is; bare words
    must still look like compound identifiers to filter prose noise.
    Deduplicated, in order of appearance.
    """
    out: List[str] = []
    seen = set()

    def add(tok: str) -> None:
        tok = tok.strip().strip("`")
        if tok and tok not in seen and _keep_token(tok):
            seen.add(tok)
            out.append(tok)

    for m in _BACKTICK.finditer(text or ""):
        add(m.group(1))
    for tok in _BARE_TOKEN.findall(text or ""):
        add(tok)
        if len(out) >= limit:
            break
    return out[:limit]


def _suffix(tokens: List[str], budget: int) -> str:
    """``tok verbatim | tok human | ...`` bounded by ``budget`` chars."""
    parts: List[str] = []
    used = 0
    for tok in tokens:
        human = split_identifier(tok)
        piece = tok if human == tok.lower() else f"{tok} = {human}"
        if used + len(piece) > budget:
            break
        parts.append(piece)
        used += len(piece) + 3
    return " | ".join(parts)


def build_embed_payload(chunk: str, *, budget: int = 300) -> str:
    """The text to EMBED for ``chunk`` (the stored chunk stays verbatim).

    Chunk text + a bounded suffix: markdown headings and identifiers in both
    forms (verbatim for literal recall, human form for language-model
    overlap). Returns the chunk unchanged when nothing was found.
    """
    if not chunk:
        return chunk
    tokens = [h.strip() for h in _HEADING.findall(chunk) if h.strip()]
    tokens += extract_identifiers(chunk)
    suffix = _suffix(tokens, budget)
    if not suffix:
        return chunk
    return f"{chunk}\n{suffix}"


def build_query_payload(query: str, *, budget: int = 200) -> str:
    """Query text + human forms of identifier-like tokens in the query.

    Symmetric with :func:`build_embed_payload`: "как работает sendSmsCode"
    embeds together with "send sms code", so the human-language question
    lands closer to the identifier's human form in vector space.
    """
    if not query:
        return query
    tokens = extract_identifiers(query, limit=8)
    suffix = _suffix(tokens, budget)
    if not suffix:
        return query
    return f"{query}\n{suffix}"
