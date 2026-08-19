"""Pgvector-direct memory backend.

Stores product-scoped text chunks with their embeddings in the
``knowledge_chunks`` table (``KnowledgeChunkORM``) and serves semantic recall
via a pgvector cosine-distance ``ORDER BY`` query accelerated by the HNSW
index created in ``init_db``.

Unlike the cognee backend this path performs NO LLM graph extraction at index
time — only chunking + embedding. Indexing a codebase is therefore bounded by
the embedder latency (batched /v1/embeddings calls), not by a multi-hour
cognify over a local model, which is the primary source of timeouts in the
cognee path.

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
from typing import Any, Dict, List, Optional, Tuple

from api.memory.base import MemoryBackend

logger = logging.getLogger(__name__)

# Reuse the shared TextSplitter config (embedder.json: text_splitter). The
# splitter is synchronous and CPU-only, so it runs in a worker thread via
# asyncio.to_thread to avoid blocking the event loop on large documents.
_DEFAULT_TOP_K = 20
# Soft cap on chunks per source to bound a runaway embedder bill on a giant
# repo blob; the expert recall only needs the most relevant top_k anyway.
_MAX_CHUNKS_PER_SOURCE = 2000


def _new_chunk_id() -> str:
    """Frontend-compatible chunk id: ``chunk_<base36 ts><6 hex>``."""
    ts = format(int(time.time()), "x")
    return f"chunk_{ts}{secrets.token_hex(3)}"


def _split_text(content: str) -> List[str]:
    """Chunk ``content`` using the shared TextSplitter config.

    Runs in a worker thread (the splitter is CPU-bound). Returns non-empty
    chunk strings. On any error (config missing, adalflow absent) falls back
    to a simple paragraph/sentence split so indexing still proceeds.
    """
    if not content or not content.strip():
        return []
    try:
        from adalflow.components.data_process import TextSplitter
        from api.config import configs

        splitter_cfg = dict(configs.get("text_splitter") or {})
        if not splitter_cfg:
            splitter_cfg = {"chunk_size": 350, "chunk_overlap": 100, "split_by": "word"}
        splitter = TextSplitter(**splitter_cfg)
        chunks = splitter.split_text(content)
        return [c for c in (chunks or []) if c and str(c).strip()]
    except Exception as e:
        logger.warning(
            "pgvector memory: TextSplitter unavailable (%s); falling back to "
            "naive split.", e,
        )
        # Naive fallback: split on double newlines, then by sentence length.
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


async def _embed_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed a batch of texts via the configured embedder.

    Returns a list of float vectors (one per text), or None on failure. Uses
    the shared ``get_embedder`` (OpenAI-compatible /v1/embeddings, wired to
    admin ``models.embedder.*``). The adalflow Embedder is sync, so the call
    runs in a worker thread.
    """
    if not texts:
        return []
    try:
        from api.tools.embedder import get_embedder

        embedder = get_embedder()

        def _do_embed() -> List[List[float]]:
            # adalflow ``Embedder.__call__(input=list)`` returns an
            # ``EmbedderOutput`` dataclass (NOT a list). The per-text vectors
            # live on its ``.data`` field as a ``List[Embedding]``, each with
            # ``.embedding: List[float]``. Iterating the dataclass directly or
            # calling a nonexistent ``.encode()`` raises
            # "'EmbedderOutput' object is not iterable".
            results = embedder(input=texts)
            err = getattr(results, "error", None)
            if err:
                logger.warning("pgvector memory: embedder returned error: %s", err)
                return []
            data = getattr(results, "data", None) or []
            out: List[List[float]] = []
            for r in data:
                vec = getattr(r, "embedding", None) or getattr(r, "vector", None) or r
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                out.append([float(x) for x in vec])
            return out

        # Throttle concurrent/bursty embedding calls via the shared embedder
        # rate limiter (admin ``embedder.*`` settings). The limiter runs the
        # to_thread coroutine under an async semaphore + request spacing.
        from api.tools.rate_limiter import _embedder_rate_limiter

        return await _embedder_rate_limiter.execute(asyncio.to_thread, _do_embed)
    except Exception as e:
        logger.warning("pgvector memory: embedding batch failed: %s", e)
        return None


async def _embed_query(query: str) -> Optional[List[float]]:
    """Embed a single query string. Returns None on failure."""
    vecs = await _embed_batch([query])
    if not vecs:
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
    ) -> int:
        """Chunk, embed, and upsert ``content`` into ``knowledge_chunks``.

        Idempotent per (product_id, source_id): existing chunks for that pair
        are deleted before insert. Returns the number of chunks stored (0 on
        empty content / embedder failure / DB down). Non-fatal.
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

        embeddings = await _embed_batch(chunks)
        if not embeddings or len(embeddings) != len(chunks):
            logger.warning(
                "pgvector memory: embedder returned %d vectors for %d chunks "
                "(product %s source %s); skipping index.",
                len(embeddings) if embeddings else 0, len(chunks), product_id, source_id,
            )
            return 0

        try:
            return await self._upsert_chunks(chunks, embeddings, product_id, source_type, source_id)
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
    ) -> int:
        """Delete existing chunks for (product_id, source_id) then insert."""
        from api.db import SessionLocal
        from api.models import KnowledgeChunkORM

        rows = []
        for i, (text, vec) in enumerate(zip(chunks, embeddings)):
            rows.append(KnowledgeChunkORM(
                id=_new_chunk_id(),
                product_id=product_id,
                source_type=source_type,
                source_id=source_id,
                chunk_index=i,
                content=text,
                embedding=vec,
            ))
        # Insert in a worker thread (SessionLocal is sync).
        def _do() -> int:
            with SessionLocal() as db:
                if source_id is not None:
                    db.query(KnowledgeChunkORM).filter(
                        KnowledgeChunkORM.product_id == product_id,
                        KnowledgeChunkORM.source_id == source_id,
                    ).delete(synchronize_session=False)
                db.add_all(rows)
                db.commit()
                return len(rows)

        return await asyncio.to_thread(_do)

    async def query(self, query: str, product_id: str, top_k: int = 20) -> str:
        """Cosine recall: embed query, return top-k chunks as joined text.

        Returns "" when pgvector is unavailable, the product has no chunks, or
        any error occurs. Capped by the ``memory_query`` timeout so a slow
        embedder cannot stall the expert SSE stream.
        """
        if not query or not query.strip() or not product_id:
            return ""
        if not _is_pgvector_capable():
            # On SQLite / pgvector-absent the cosine operator is unsupported;
            # return "" so the expert path falls back to artifact docs.
            return ""
        try:
            qvec = await _embed_query(query)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("pgvector memory: query embedding failed: %s", e)
            return ""
        if not qvec:
            return ""
        k = max(1, min(top_k or _DEFAULT_TOP_K, 100))
        try:
            return await self._cosine_search(qvec, product_id, k)
        except Exception as e:
            logger.warning(
                "pgvector memory: query failed for product %s: %s", product_id, e
            )
            return ""

    async def _cosine_search(self, qvec: List[float], product_id: str, top_k: int) -> str:
        """Run the cosine-distance ORDER BY query with a timeout."""
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
            results = await asyncio.wait_for(asyncio.to_thread(_do), timeout=query_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "pgvector memory: query timed out after %.0fs for product %s.",
                query_timeout, product_id,
            )
            return ""
        if not results:
            return ""
        return "\n\n".join(results)

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
                                items.append((docs.strip(), "codebase", c.id))
                            pages = getattr(c, "pages", None) or {}
                            if isinstance(pages, dict):
                                for page_id, page in pages.items():
                                    pc = ""
                                    if isinstance(page, dict):
                                        pc = page.get("content") or ""
                                    elif isinstance(page, str):
                                        pc = page
                                    if pc and pc.strip():
                                        items.append((pc.strip(), "codebase", c.id))
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
