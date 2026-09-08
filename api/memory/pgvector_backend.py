"""Pgvector-direct memory backend.

Stores product-scoped text chunks with their embeddings in the
``knowledge_chunks`` table (``KnowledgeChunkORM``) and serves semantic recall
via a pgvector cosine-distance ``ORDER BY`` query accelerated by the HNSW
index (pinned + created by ``api.db.ensure_embedding_dimension_and_hnsw``
on the first index run — the column is dimensionless until the embedder
dimension is known).

Indexing performs NO LLM work — only chunking + embedding — so indexing a
codebase is bounded by the embedder latency (batched /v1/embeddings calls).

The ``embedding`` column is a dimensionless pgvector ``Vector`` on Postgres
and degrades to ``Text`` on SQLite. On SQLite (tests) the cosine query is not
meaningful, so ``query`` returns "" rather than attempting an unsupported
operator — tests assert the SQL shape via a mocked session instead.

All public methods are async and non-fatal: on any error (DB down, embedder
unavailable, pgvector absent, dimension mismatch) they log and return a safe
empty/zero result so the expert/docgen paths fall back to artifact docs.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from api.memory.base import MemoryBackend
from api.memory.lexical_boost import (
    escape_ilike,
    extract_query_tokens,
    lexical_boost_enabled,
    rank_lexical_rows,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)

# Reuse the shared TextSplitter config (embedder.json: text_splitter). The
# splitter is synchronous and CPU-only, so it runs in a worker thread via
# asyncio.to_thread to avoid blocking the event loop on large documents.
_DEFAULT_TOP_K = 20
# Soft cap on chunks per source to bound a runaway embedder bill on a giant
# repo blob; the expert recall only needs the most relevant top_k anyway.
_MAX_CHUNKS_PER_SOURCE = 2000
# Candidate cap for the lexical-boost ILIKE scan (3.2): SQL filters by
# ILIKE ANY, then the Python stage re-ranks and keeps top_k. The cap bounds
# the scan on products with thousands of chunks.
_LEXICAL_CANDIDATE_CAP = 200

# Local embedding servers (llama.cpp / LM Studio / vLLM over nomic-style BPE
# tokenizers) reject the WHOLE /v1/embeddings request with HTTP 400
# ``Prompt contains invalid tokens`` when any input text contains codepoints
# the tokenizer cannot process: unpaired UTF-16 surrogates (not encodable to
# UTF-8 at all), C0 control characters (NUL in particular), DEL/C1 controls
# and invisible format characters (zero-width, BOM, bidi controls). Tab,
# newline and carriage return are legitimate text and survive.
_EMBED_STRIP_CONTROL = {c: None for c in range(0x20) if c not in (0x09, 0x0A, 0x0D)}
# DEL + C1 controls: another class local tokenizers reject outright.
_EMBED_STRIP_CONTROL.update({c: None for c in range(0x7F, 0xA0)})
# Invisible format characters: zero-width space/joiners, directional marks,
# soft hyphen, word joiner and the BOM. They carry no recall value and are
# known tokenizer poison. Line/paragraph separators become plain newlines.
_EMBED_STRIP_FORMAT = {
    c: None for c in (
        list(range(0x200B, 0x2010))   # zero-width space..RLM + general punct
        + list(range(0x202A, 0x202F))  # bidi embedding/override controls
        + list(range(0x2060, 0x2070))  # word joiner / invisible operators
        + [0x00AD, 0xFEFF]             # soft hyphen, BOM / ZWNBSP
    )
}
_EMBED_STRIP_FORMAT[0x2028] = "\n"  # line separator
_EMBED_STRIP_FORMAT[0x2029] = "\n"  # paragraph separator


def _sanitize_for_embedder(text: str) -> str:
    """Make ``text`` safe for the local /v1/embeddings tokenizer.

    Drops unpaired surrogates via a UTF-8 round-trip with ``errors="ignore"``,
    strips C0/DEL/C1 control characters (except ``\t``/``\n``/``\r``) and
    removes invisible format characters (zero-width, BOM, bidi controls).
    Idempotent and best-effort: the result may be shorter, possibly empty.
    Deliberately does NOT unicode-normalize (NFC/NFKD): stored chunks must
    stay byte-identical to the source document so ``_compute_char_spans``
    can locate them for citation metadata.
    """
    if not text:
        return ""
    cleaned = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return cleaned.translate({**_EMBED_STRIP_CONTROL, **_EMBED_STRIP_FORMAT})


def _fold_for_embedder(text: str) -> str:
    """Last-resort ASCII fold of a single text the embedder rejected.

    Applied ONLY to an item the server already refused with 400 ``invalid
    tokens`` after the regular sanitizer ran, as one bounded retry before
    dropping it: NFKD-fold to printable ASCII, collapse whitespace. A text
    that is already ASCII folds to itself (the retry is skipped by the
    caller), and a non-Latin text folds to near-empty (dropped anyway), so
    this can only ever rescue an otherwise-lost chunk, never silently
    degrade a good one.
    """
    import unicodedata

    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    folded = folded.translate(_EMBED_STRIP_CONTROL)
    return " ".join(folded.split())


def _suspect_codepoints(text: str, limit: int = 12) -> str:
    """Human-readable list of non-ASCII codepoints in ``text`` (diagnostics)."""
    seen: list = []
    for ch in text or "":
        if ord(ch) > 0x7E and ch not in seen:
            seen.append(ch)
        if len(seen) >= limit:
            break
    return " ".join(f"U+{ord(ch):04X}({ch!r})" for ch in seen) or "(none)"


def _new_chunk_id() -> str:
    """Frontend-compatible chunk id: ``chunk_<base36 ts><6 hex>``."""
    ts = format(int(time.time()), "x")
    return f"chunk_{ts}{secrets.token_hex(3)}"


# Whether the live ``knowledge_chunks`` table carries the Wave D citation
# columns (chunk_id / source_path / char_span). ``create_all`` adds them only
# to FRESH tables; pre-existing installs keep the old schema, in which case the
# citation metadata is silently skipped (rows/old schema stay valid).
_citation_columns_cache: Dict[str, Optional[bool]] = {"available": None}


def _citation_columns_available() -> bool:
    """True when knowledge_chunks has the citation columns (cached)."""
    cached = _citation_columns_cache.get("available")
    if cached is not None:
        return cached
    available = False
    try:
        import sqlalchemy as sa

        from api.db import engine
        from api.models import KnowledgeChunkORM

        insp = sa.inspect(engine)
        if insp.has_table(KnowledgeChunkORM.__tablename__):
            cols = {
                c["name"] for c in insp.get_columns(KnowledgeChunkORM.__tablename__)
            }
            available = {"chunk_id", "source_path", "char_span"} <= cols
    except Exception as e:  # pragma: no cover - DB-dependent
        logger.debug("citation column introspection failed: %s", e)
        available = False
    _citation_columns_cache["available"] = available
    return available


def reset_citation_columns_cache() -> None:
    """Drop the cached citation-columns probe result (tests)."""
    _citation_columns_cache["available"] = None


def _compute_char_spans(content: str, chunks: List[str]) -> List[Optional[List[int]]]:
    """Best-effort [start, end] offsets of each chunk within ``content``.

    Walks the source once, searching each chunk from the previous match
    position. Overlap splitters emit chunk N+1 that STARTS INSIDE chunk N,
    so a forward miss is retried from the previous chunk's start (the
    overlap window) before giving up. A chunk that still cannot be located
    gets None — the span is provenance metadata, never a hard requirement.
    """
    spans: List[Optional[List[int]]] = []
    search_from = 0
    last_start = 0
    for chunk in chunks:
        idx = content.find(chunk, search_from)
        if idx < 0:
            # Overlap splitter: the chunk may begin inside the previous one.
            idx = content.find(chunk, last_start)
        if idx < 0:
            spans.append(None)
            continue
        spans.append([idx, idx + len(chunk)])
        last_start = idx
        search_from = idx + max(1, len(chunk))
    return spans


def _split_text(content: str) -> List[str]:
    """Chunk ``content`` using the shared text-splitter config.

    Runs in a worker thread (the splitter is CPU-bound). Uses the langchain
    ``RecursiveCharacterTextSplitter`` configured from ``embedder.json``
    (``text_splitter``: chunk_size 350 / chunk_overlap 100, word-based).
    Returns non-empty chunk strings. On any error (config missing, langchain
    absent) falls back to a simple paragraph/sentence split so indexing
    still proceeds.
    """
    if not content or not content.strip():
        return []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from api.config import configs

        splitter_cfg = dict(configs.get("text_splitter") or {})
        if not splitter_cfg:
            splitter_cfg = {"chunk_size": 350, "chunk_overlap": 100, "split_by": "word"}
        chunk_size = int(splitter_cfg.get("chunk_size") or 350)
        chunk_overlap = int(splitter_cfg.get("chunk_overlap") or 100)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_text(content)
        return [c for c in (chunks or []) if c and str(c).strip()]
    except Exception as e:
        logger.warning(
            "pgvector memory: text splitter unavailable (%s); falling back to "
            "naive split.",
            e,
        )
        out: List[str] = []
        for para in content.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) <= 600:
                out.append(para)
            else:
                # Greedy ~350-word windows.
                words = para.split()
                for i in range(0, len(words), 350):
                    out.append(" ".join(words[i:i + 350]))
        return out


async def _embed_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Embed a batch of texts via the configured embedder.

    Returns one entry per input text, position-aligned: the embedding
    vector, or ``None`` when that text could not be embedded. Never raises.
    Uses the shared ``get_embedder`` (langchain ``OpenAIEmbeddings`` over the
    OpenAI-compatible /v1/embeddings endpoint, wired to admin
    ``models.embedder.*``). The embedder call is sync, so it runs in a
    worker thread.

    Every input first goes through :func:`_sanitize_for_embedder` — local
    tokenizers reject the whole batch with HTTP 400 ``Prompt contains
    invalid tokens`` when any input carries unpaired surrogates, control
    characters or invisible format characters. If a batch request still
    fails with a 400 (an ``status_code`` attribute of 400 on the exception),
    the batch is split in halves until the offending item(s) are isolated,
    so one poisoned chunk no longer zeroes the whole source. An isolated
    item gets one bounded ASCII-fold retry (:func:`_fold_for_embedder`)
    before being dropped, and the drop logs the payload snippet + suspect
    codepoints so a tokenizer that rejects a whole script (seen with
    GGUF-converted nomic embedders) is diagnosable from the logs. Systemic
    failures (connection, timeout, 5xx, exhausted 429 retries) abort
    without isolation: every entry stays ``None``.
    """
    out: List[Optional[List[float]]] = [None] * len(texts)
    if not texts:
        return []
    cleaned = [_sanitize_for_embedder(t) for t in texts]
    positions = [i for i, t in enumerate(cleaned) if t and t.strip()]
    if not positions:
        return out
    try:
        from api.tools.embedder import get_embedder

        embedder = get_embedder()
    except Exception as e:
        logger.warning("pgvector memory: embedding batch failed: %s", e)
        return out

    from api.tools.rate_limiter import _embedder_rate_limiter

    async def _call(items: List[str]) -> List[List[float]]:
        def _do_embed() -> List[List[float]]:
            # langchain ``OpenAIEmbeddings.embed_documents(list)`` returns a
            # plain ``List[List[float]]`` (one vector per input text).
            vectors = embedder.embed_documents(items)
            out_vecs: List[List[float]] = []
            for vec in vectors or []:
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                out_vecs.append([float(x) for x in vec])
            if len(out_vecs) != len(items):
                raise ValueError(
                    f"embedder returned {len(out_vecs)} vectors for {len(items)} texts"
                )
            return out_vecs

        # Throttle concurrent/bursty embedding calls via the shared embedder
        # rate limiter (admin ``embedder.*`` settings). The limiter runs the
        # to_thread coroutine under an async semaphore + request spacing.
        return await _embedder_rate_limiter.execute(asyncio.to_thread, _do_embed)

    async def _fill(indices: List[int], items: List[str]) -> None:
        try:
            vectors = await _call(items)
        except Exception as e:
            if getattr(e, "status_code", None) != 400:
                logger.warning("pgvector memory: embedding batch failed: %s", e)
                return
            if len(items) == 1:
                # One poisoned item isolated. Bounded retry with an ASCII
                # fold before dropping: some GGUF tokenizers reject whole
                # scripts/classes of codepoints (not just controls), and the
                # fold salvages the chunk when only a few codepoints are the
                # problem. The stored chunk keeps its original (sanitized)
                # text — only the vector is computed on the folded input.
                folded = _fold_for_embedder(items[0])
                if folded and folded != items[0]:
                    try:
                        vectors = await _call([folded])
                        out[indices[0]] = vectors[0]
                        return
                    except Exception:
                        pass  # fall through to the drop with diagnostics
                logger.warning(
                    "pgvector memory: one text rejected by the embedder "
                    "(invalid tokens); dropping it. Payload (first 120 "
                    "chars, repr): %r; non-ascii codepoints: %s; error: %s",
                    items[0][:120], _suspect_codepoints(items[0]), e,
                )
                return
            mid = len(items) // 2
            await _fill(indices[:mid], items[:mid])
            await _fill(indices[mid:], items[mid:])
            return
        for pos, vec in zip(indices, vectors):
            out[pos] = vec

    await _fill(positions, [cleaned[i] for i in positions])
    return out


async def _embed_query(query: str) -> Optional[List[float]]:
    """Embed a single query string. Returns None on failure."""
    vecs = await _embed_batch([query])
    if not vecs or vecs[0] is None:
        return None
    return vecs[0]


def _is_pgvector_capable() -> bool:
    """True only when the active DB is Postgres with pgvector available."""
    try:
        from api.db import DB_PROVIDER
        from api.models import _PGVECTOR_AVAILABLE
        return (DB_PROVIDER or "").lower() in ("postgres", "postgresql") and bool(_PGVECTOR_AVAILABLE)
    except Exception:
        return False


class PgVectorMemoryBackend(MemoryBackend):
    """Direct Postgres+pgvector chunk store with cosine recall (no graph)."""

    name = "pgvector"

    async def index(
        self,
        content: str,
        product_id: str,
        source_type: str = "codebase",
        source_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> int:
        """Chunk, embed, and upsert ``content`` into ``knowledge_chunks``.

        Idempotent per (product_id, source_id): existing chunks for that pair
        are deleted before insert. Returns the number of chunks stored (0 on
        empty content / embedder failure / DB down). Non-fatal.

        ``source_path`` (optional, repo-relative) is stored on every chunk
        together with a stable ``chunk_id`` and a best-effort ``char_span``
        (character offsets within the source document) so recall results can
        be cited back to their origin. Skipped silently when the live table
        lacks the citation columns (pre-Wave-D installs).
        """
        if not content or not content.strip() or not product_id:
            return 0
        try:
            chunks = await asyncio.to_thread(_split_text, content)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("pgvector memory: chunking failed: %s", e)
            return 0
        if not chunks:
            return 0
        # Cap to bound embedder cost on giant blobs.
        if len(chunks) > _MAX_CHUNKS_PER_SOURCE:
            logger.info(
                "pgvector memory: capping %d chunks to %d for product %s source %s.",
                len(chunks), _MAX_CHUNKS_PER_SOURCE, product_id, source_id,
            )
            chunks = chunks[:_MAX_CHUNKS_PER_SOURCE]
        # Sanitize before embedding/storing: NUL bytes, other C0 controls and
        # unpaired surrogates make local tokenizers reject the whole request
        # (400 "Prompt contains invalid tokens") and would pollute recall
        # output even if the embedder accepted them.
        chunks = [c for c in (_sanitize_for_embedder(c) for c in chunks) if c.strip()]
        if not chunks:
            return 0

        # Layer A (fork port): the text sent to the EMBEDDER is not the stored
        # chunk — each chunk is embedded together with a bounded suffix of its
        # identifiers (verbatim + human form) so a human-language question
        # lands closer to "sendSmsCode" via "send sms code". The STORED
        # content stays the verbatim chunk (recall output unchanged).
        try:
            from api.utils.identifiers import build_embed_payload

            embed_payloads = [build_embed_payload(c) for c in chunks]
        except Exception as e:  # pragma: no cover - stdlib-only module
            logger.debug("identifier payload layer skipped: %s", e)
            embed_payloads = chunks

        embeddings = await _embed_batch(embed_payloads)
        if not embeddings or len(embeddings) != len(chunks):
            logger.warning(
                "pgvector memory: embedder returned %d vectors for %d chunks "
                "(product %s source %s); skipping index.",
                len(embeddings) if embeddings else 0, len(chunks), product_id, source_id,
            )
            return 0
        pairs = [(c, v) for c, v in zip(chunks, embeddings) if v is not None]
        dropped = len(chunks) - len(pairs)
        if dropped:
            logger.warning(
                "pgvector memory: dropped %d of %d chunk(s) rejected by the "
                "embedder (product %s source %s).",
                dropped, len(chunks), product_id, source_id,
            )
        if not pairs:
            return 0
        chunks = [c for c, _ in pairs]
        embeddings = [v for _, v in pairs]

        # The embedding column is dimensionless until the first real batch
        # reveals the embedder dimension: pin it + build the HNSW index now
        # (pgvector cannot index a dimensionless column; no-op once ready).
        try:
            from api.db import ensure_embedding_dimension_and_hnsw

            await asyncio.to_thread(
                ensure_embedding_dimension_and_hnsw, len(embeddings[0])
            )
        except Exception as e:  # pragma: no cover - helper is non-fatal
            logger.debug("pgvector memory: HNSW ensure skipped: %s", e)

        try:
            return await self._upsert_chunks(
                chunks, embeddings, product_id, source_type, source_id,
                source_path=source_path, source_content=content,
            )
        except Exception as e:
            logger.warning(
                "pgvector memory: upsert failed for product %s source %s: %s",
                product_id, source_id, e,
            )
            return 0

    async def _upsert_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        product_id: str,
        source_type: str,
        source_id: Optional[str],
        *,
        source_path: Optional[str] = None,
        source_content: Optional[str] = None,
    ) -> int:
        """Delete existing chunks for (product_id, source_id) then insert."""
        from api.db import SessionLocal
        from api.models import KnowledgeChunkORM

        store_citations = _citation_columns_available()
        spans: List[Optional[List[int]]] = (
            _compute_char_spans(source_content or "", chunks)
            if store_citations and source_content
            else [None] * len(chunks)
        )
        now = datetime.utcnow()
        rows: List[Dict[str, Any]] = []
        for i, (text, vec) in enumerate(zip(chunks, embeddings)):
            row: Dict[str, Any] = {
                "id": _new_chunk_id(),
                "product_id": product_id,
                "source_type": source_type,
                "source_id": source_id,
                "chunk_index": i,
                "content": text,
                "embedding": vec,
                "created_at": now,
            }
            if store_citations:
                row["chunk_id"] = f"c:{source_id or 'unknown'}:{i}"
                row["source_path"] = source_path
                row["char_span"] = spans[i]
            rows.append(row)

        # Insert via Core with an EXPLICIT column list. The ORM unit-of-work
        # emits every mapped column (NULL for unset ones), which fails on
        # pre-citation-column schemas with ``column "chunk_id" ... does not
        # exist`` even when the citation values are skipped — the explicit
        # keys keep the statement valid on both schemas.
        from sqlalchemy import insert as sa_insert
        insert_stmt = sa_insert(KnowledgeChunkORM.__table__)

        # Insert in a worker thread (SessionLocal is sync).
        def _do() -> int:
            with SessionLocal() as db:
                if source_id is not None:
                    db.query(KnowledgeChunkORM).filter(
                        KnowledgeChunkORM.product_id == product_id,
                        KnowledgeChunkORM.source_id == source_id,
                    ).delete(synchronize_session=False)
                if rows:
                    db.execute(insert_stmt, rows)
                db.commit()
                return len(rows)

        return await asyncio.to_thread(_do)

    async def query(self, query: str, product_id: str, top_k: int = 20) -> str:
        """Cosine recall + optional lexical boost, top-k chunks as joined text.

        Returns "" when pgvector is unavailable, the product has no chunks, or
        any error occurs. Capped by the ``memory_query`` timeout so a slow
        embedder cannot stall the expert SSE stream.

        Lexical boost (fork port 3.2): when ``EXPERT_LEXICAL_BOOST`` is on and
        the question carries code-like tokens (identifier / table name), the
        chunks are ALSO matched exactly by text (``ILIKE ANY``) and the two
        rankings are merged via reciprocal rank fusion — agreement wins,
        exact-name hits cosine missed still surface. The boost also keeps
        recall alive when the embedder is down: with no query vector but
        non-empty lexical hits, the lexical top-k is served directly.
        """
        if not query or not query.strip() or not product_id:
            return ""
        if not _is_pgvector_capable():
            # On SQLite / pgvector-absent the cosine operator is unsupported;
            # return "" so the expert path falls back to artifact docs.
            return ""
        raw_query = query
        # Layer A on the query side too: identifier-like tokens in the
        # question are embedded together with their human forms, symmetric
        # with the chunk payloads built at index time.
        try:
            from api.utils.identifiers import build_query_payload

            query = build_query_payload(query) or query
        except Exception as e:  # pragma: no cover - stdlib-only module
            logger.debug("identifier query layer skipped: %s", e)
        try:
            qvec = await _embed_query(query)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("pgvector memory: query embedding failed: %s", e)
            qvec = None
        k = max(1, min(top_k or _DEFAULT_TOP_K, 100))
        # Lexical leg (runs regardless of the embedder state). Code-like
        # tokens are extracted from the RAW question — not the layer-A
        # payload — so the ILIKE patterns stay verbatim identifier names.
        lexical_rows: List[str] = []
        if lexical_boost_enabled():
            tokens = extract_query_tokens(raw_query)
            if tokens:
                lexical_rows = await self._lexical_search(product_id, tokens, k)
        if not qvec:
            # Embedder unavailable: the boost degrades gracefully to exact-
            # name recall instead of an empty context.
            if lexical_rows:
                logger.info(
                    "pgvector memory: embedder unavailable; serving %d lexical "
                    "hit(s) for product %s.",
                    len(lexical_rows), product_id,
                )
                return "\n\n".join(lexical_rows)
            return ""
        if lexical_rows:
            try:
                cosine_rows = await self._cosine_rows(qvec, product_id, k)
            except Exception as e:
                logger.warning(
                    "pgvector memory: cosine leg failed for product %s: %s",
                    product_id, e,
                )
                cosine_rows = []
            merged = reciprocal_rank_fusion(cosine_rows, lexical_rows, limit=k)
            return "\n\n".join(merged) if merged else ""
        try:
            return await self._cosine_search(qvec, product_id, k)
        except Exception as e:
            logger.warning(
                "pgvector memory: query failed for product %s: %s", product_id, e
            )
            return ""

    async def _cosine_search(self, qvec: List[float], product_id: str, top_k: int) -> str:
        """Cosine recall as joined text ("" when nothing matched)."""
        rows = await self._cosine_rows(qvec, product_id, top_k)
        if not rows:
            return ""
        return "\n\n".join(rows)

    async def _cosine_rows(
        self, qvec: List[float], product_id: str, top_k: int
    ) -> List[str]:
        """Run the cosine-distance ORDER BY query with a timeout ([] on timeout)."""
        from api.config.timeout import resolve_timeout
        from api.db import engine
        from sqlalchemy import text

        query_timeout = resolve_timeout("memory_query")
        # pgvector cosine distance: embedding <=> :q (smaller = more similar).
        # Cast the parameter to vector so the <=> operator resolves. The HNSW
        # index on embedding (created in init_db) accelerates the ORDER BY.
        sql = text(
            "SELECT content FROM knowledge_chunks "
            "WHERE product_id = :pid "
            "ORDER BY embedding <=> CAST(:q AS vector) "
            "LIMIT :k"
        )
        # pgvector accepts a "[1,2,3]"-style string literal for the cast.
        vec_literal = "[" + ",".join(str(float(x)) for x in qvec) + "]"

        def _do() -> List[str]:
            with engine.connect() as conn:
                rows = conn.execute(
                    sql, {"pid": product_id, "q": vec_literal, "k": top_k}
                ).fetchall()
                return [r[0] for r in rows if r and r[0]]

        try:
            return await asyncio.wait_for(asyncio.to_thread(_do), timeout=query_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "pgvector memory: query timed out after %.0fs for product %s.",
                query_timeout, product_id,
            )
            return []

    async def _lexical_search(
        self, product_id: str, tokens: List[str], top_k: int
    ) -> List[str]:
        """Exact-text recall: ``content ILIKE ANY (:toks)`` over the product.

        Postgres-only stage of the lexical boost. Candidates are capped by
        ``_LEXICAL_CANDIDATE_CAP`` (ordered by id for determinism) and then
        re-ranked in Python by token occurrences. Bounded by the same
        ``memory_query`` timeout as the cosine leg; ANY failure logs and
        degrades to [] so the boost never breaks the cosine recall.
        """
        from api.config.timeout import resolve_timeout
        from api.db import engine
        from sqlalchemy import text

        timeout = resolve_timeout("memory_query")
        patterns = [escape_ilike(t) for t in tokens if t]
        if not patterns:
            return []
        sql = text(
            "SELECT content FROM knowledge_chunks "
            "WHERE product_id = :pid AND content ILIKE ANY (:toks) "
            "ORDER BY id LIMIT :cap"
        )

        def _do() -> List[str]:
            with engine.connect() as conn:
                rows = conn.execute(
                    sql, {"pid": product_id, "toks": patterns, "cap": _LEXICAL_CANDIDATE_CAP}
                ).fetchall()
                return [r[0] for r in rows if r and r[0]]

        try:
            rows = await asyncio.wait_for(asyncio.to_thread(_do), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "pgvector memory: lexical boost timed out after %.0fs for "
                "product %s.",
                timeout, product_id,
            )
            return []
        except Exception as e:
            logger.warning(
                "pgvector memory: lexical boost failed for product %s: %s",
                product_id, e,
            )
            return []
        return rank_lexical_rows(rows, tokens, top_k=top_k)

    async def clear_product(self, product_id: str) -> bool:
        """Delete all chunks for a product. Returns True on success."""
        if not product_id:
            return False
        try:
            from api.db import SessionLocal
            from api.models import KnowledgeChunkORM

            def _do() -> bool:
                with SessionLocal() as db:
                    db.query(KnowledgeChunkORM).filter(
                        KnowledgeChunkORM.product_id == product_id
                    ).delete(synchronize_session=False)
                    db.commit()
                    return True

            return await asyncio.to_thread(_do)
        except Exception as e:
            logger.warning("pgvector memory: clear_product(%s) failed: %s", product_id, e)
            return False

    async def reindex_product(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """Rebuild the index from source artifacts for one or all products."""
        try:
            from api.db import SessionLocal
            from api.models import ProductORM
            from sqlalchemy.orm import selectinload

            def _load() -> List[Tuple[str, List[Tuple[str, str, Optional[str]]]]]:
                """Return [(product_id, [(content, source_type, source_id), ...])]."""
                with SessionLocal() as db:
                    q = db.query(ProductORM).options(
                        selectinload(ProductORM.codebases),
                        selectinload(ProductORM.specs),
                        selectinload(ProductORM.links),
                        selectinload(ProductORM.knowledge_nodes),
                    )
                    if product_id:
                        q = q.filter(ProductORM.id == product_id)
                    products = q.all()
                    out = []
                    for p in products:
                        items: List[Tuple[str, str, Optional[str]]] = []
                        for c in p.codebases:
                            docs = getattr(c, "generated_docs", None) or ""
                            if docs and docs.strip():
                                # Pages are section slices of generated_docs —
                                # indexing both would duplicate the content.
                                items.append((docs.strip(), "codebase", c.id))
                                continue
                            # Legacy artifact WITHOUT generated_docs: fall
                            # back to pages, each under a DISTINCT source id —
                            # reusing the codebase id would make every page's
                            # upsert delete the previous page's chunks.
                            pages = getattr(c, "pages", None) or {}
                            if isinstance(pages, dict):
                                for page_id, page in pages.items():
                                    pc = ""
                                    if isinstance(page, dict):
                                        pc = page.get("content") or ""
                                    elif isinstance(page, str):
                                        pc = page
                                    if pc and pc.strip():
                                        items.append((
                                            pc.strip(), "codebase",
                                            f"{c.id}::page::{page_id}",
                                        ))
                        for s in p.specs:
                            c = getattr(s, "content", None) or ""
                            if c and c.strip():
                                items.append((c.strip(), "spec", s.id))
                        for l in p.links:
                            c = getattr(l, "content", None) or ""
                            if c and c.strip():
                                items.append((c.strip(), "links", l.id))
                        for n in p.knowledge_nodes:
                            md = getattr(n, "content_md", None) or ""
                            if md and md.strip():
                                items.append((md.strip(), "knowledge_node", n.id))
                        out.append((p.id, items))
                    return out

            batches = await asyncio.to_thread(_load)
            if not batches:
                return {"success": True, "message": "No products found to reindex.", "reindexed_count": 0}

            reindexed = 0
            for pid, items in batches:
                if not items:
                    continue
                await self.clear_product(pid)
                for content, source_type, source_id in items:
                    await self.index(content, pid, source_type=source_type, source_id=source_id)
                reindexed += 1
            return {
                "success": True,
                "message": f"Reindexed {reindexed} product(s) into pgvector memory.",
                "reindexed_count": reindexed,
            }
        except Exception as e:
            logger.error("pgvector memory: reindex failed: %s", e, exc_info=True)
            return {"success": False, "message": f"Reindex error: {e}", "reindexed_count": 0}

    def status(self) -> Dict[str, Any]:
        """Chunk + product counts for the admin UI (non-fatal on DB down)."""
        out: Dict[str, Any] = {"backend": self.name, "available": _is_pgvector_capable()}
        try:
            # status() is synchronous (called from the admin GET handler), so
            # run the count query directly on the calling thread.
            total, prods = _counts_safe()
            out["chunk_count"] = total
            out["product_count"] = prods
        except Exception as e:
            logger.debug("pgvector memory: status counts failed: %s", e)
            out["chunk_count"] = 0
            out["product_count"] = 0
        return out


def _counts_safe() -> Tuple[int, int]:
    """Synchronous chunk/product counts (best-effort, used by status)."""
    from api.db import SessionLocal
    from api.models import KnowledgeChunkORM
    from sqlalchemy import func

    with SessionLocal() as db:
        total = db.query(func.count(KnowledgeChunkORM.id)).scalar() or 0
        prods = db.query(func.count(func.distinct(KnowledgeChunkORM.product_id))).scalar() or 0
        return int(total), int(prods)
