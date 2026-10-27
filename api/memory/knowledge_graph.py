"""Per-product knowledge graph (SPEC-008): JSONB store + builder + retrieval.

One graph per product (``knowledge_graphs`` row, ``api.models.KnowledgeGraphORM``).
Nodes carry STABLE keys so merges are plain dict upserts:

- ``term:<normalized name>``   — domain term (description, aliases);
- ``page:<artifact_type>:<artifact_id>:<page_id>`` — documentation page.

Edges (``<src>--<rel>--><dst>`` keys):

- ``mentions``  term→page (weight + short usage context);
- ``synonym`` / ``part_of`` / ``related``  term—term (from the same LLM pass);
- ``depends_on``  page→page — shape reserved for SPEC-010 reuse, not
  generated here.

Disambiguation invariant: a term and its pages live inside ONE product's
graph only, so identical words ("RMS") can mean different things in
different products; recall stays scoped by ``product_id`` on both the graph
and the single pgvector namespace per product.

Builder rides the SPEC-010 post-indexation path: only changed pages are
re-extracted, dead pages/terms/edges are pruned on regeneration. All LLM
work is batched, gated (``knowledge_graph_enabled``) and bounded
(``knowledge_graph_batch``, ``knowledge_graph_build``, DOCGEN_LLM_CONCURRENCY
— ``api/config/timeout.py``). Everything is non-fatal: a failed extraction
leaves the previous graph untouched.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from api.docgen._common import utc_now_iso

logger = logging.getLogger(__name__)

#: Hard graph budget (LRU-trimmed by :func:`enforce_budget`).
MAX_NODES = 5000
MAX_EDGES = 20000
MAX_TERMS_PER_PAGE = 25
MAX_ALIASES = 8
#: term—term relation kinds produced by the extractor.
TERM_RELATIONS = ("synonym", "part_of", "related")
#: Shortest query/term token considered (keeps "RMS" as an acronym).
MIN_TOKEN_LEN = 2
#: Chars of a page body fed to one extraction batch entry.
_EXCERPT_CHARS = 2500
#: In-process graph cache TTL (the recall path reads the graph per query).
_GRAPH_CACHE_TTL = 30.0

_TOKEN_RE = re.compile(r"[\w\-.]+", re.UNICODE)

_graph_cache: Dict[str, Tuple[float, Optional[dict]]] = {}
_graph_lock: Optional[asyncio.Lock] = None


def _graph_write_lock() -> asyncio.Lock:
    """Process-wide lock for read-modify-write graph merges (lazily bound)."""
    global _graph_lock
    if _graph_lock is None:
        _graph_lock = asyncio.Lock()
    return _graph_lock


# ---------------------------------------------------------------------------
# Keys + normalization
# ---------------------------------------------------------------------------
def normalize_term(name: str) -> str:
    """Canonical term form: lowercase, single-spaced."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def term_key(name: str) -> str:
    return f"term:{normalize_term(name)}"


def page_key(artifact_type: str, artifact_id: str, page_id: str) -> str:
    return f"page:{artifact_type}:{artifact_id}:{page_id}"


def edge_key(src: str, rel: str, dst: str) -> str:
    return f"{src}--{rel}-->{dst}"


def new_graph() -> Dict[str, Dict[str, Any]]:
    return {"nodes": {}, "edges": {}}


def page_source_id(node: dict) -> str:
    """Chunk ``source_id`` a page node corresponds to (SPEC-010 convention)."""
    return f"{node['artifact_id']}::page::{node['page_id']}"


# ---------------------------------------------------------------------------
# Pure graph mutations (dict upserts by stable key)
# ---------------------------------------------------------------------------
def upsert_term(
    graph: dict, name: str, *, description: str = "", aliases: Sequence[str] = (),
) -> str:
    key = term_key(name)
    if not key[len("term:"):]:
        return ""
    node = graph["nodes"].get(key) or {
        "key": key, "type": "term", "name": normalize_term(name),
        "description": "", "aliases": [], "updated_at": utc_now_iso(),
    }
    if description:
        node["description"] = description[:500]
    merged = [normalize_term(a) for a in (aliases or []) if normalize_term(a)]
    for a in merged:
        if a.lower() not in [x.lower() for x in node["aliases"]] and a != node["name"]:
            node["aliases"].append(a)
    node["aliases"] = node["aliases"][-MAX_ALIASES:]
    node["updated_at"] = utc_now_iso()
    graph["nodes"][key] = node
    return key


def upsert_page(
    graph: dict,
    artifact_type: str,
    artifact_id: str,
    page_id: str,
    *,
    title: str = "",
) -> str:
    key = page_key(artifact_type, artifact_id, page_id)
    node = graph["nodes"].get(key) or {
        "key": key, "type": "page", "artifact_type": artifact_type,
        "artifact_id": artifact_id, "page_id": page_id, "title": "",
        "updated_at": utc_now_iso(),
    }
    if title:
        node["title"] = str(title)[:256]
    node["updated_at"] = utc_now_iso()
    graph["nodes"][key] = node
    return key


def set_page_terms(graph: dict, pkey: str, page_terms: Sequence[dict]) -> None:
    """Replace the ``mentions`` edges of ONE page (regeneration-safe upsert).

    Edges from terms not in the new set are dropped, so terms that died with
    the old page text disappear; surviving term nodes without any edge left
    are pruned by the caller (:func:`prune_orphan_terms`).
    """
    wanted: Dict[str, dict] = {}
    for raw in page_terms or []:
        if not isinstance(raw, dict):
            continue
        name = normalize_term(str(raw.get("term") or ""))
        if len(name) < MIN_TOKEN_LEN:
            continue
        tkey = upsert_term(
            graph, name,
            description=str(raw.get("description") or ""),
            aliases=[str(a) for a in (raw.get("aliases") or [])],
        )
        if not tkey or tkey in wanted:
            continue
        try:
            weight = max(0.0, min(1.0, float(raw.get("weight") or 0.5)))
        except (TypeError, ValueError):
            weight = 0.5
        wanted[tkey] = {"weight": weight, "context": str(raw.get("context") or "")[:200]}
    for ekey, edge in list(graph["edges"].items()):
        if edge.get("dst") == pkey and edge.get("rel") == "mentions" and edge.get("src") not in wanted:
            del graph["edges"][ekey]
    for tkey, meta in wanted.items():
        graph["edges"][edge_key(tkey, "mentions", pkey)] = {
            "src": tkey, "dst": pkey, "rel": "mentions",
            "weight": meta["weight"], "context": meta["context"],
            "updated_at": utc_now_iso(),
        }


def add_term_relations(graph: dict, relations: Sequence[dict]) -> int:
    """Upsert term—term edges; unknown endpoints become bare term nodes."""
    added = 0
    for raw in relations or []:
        if not isinstance(raw, dict):
            continue
        src = upsert_term(graph, str(raw.get("from") or ""))
        dst = upsert_term(graph, str(raw.get("to") or ""))
        rel = str(raw.get("kind") or "").strip().lower()
        if not src or not dst or src == dst or rel not in TERM_RELATIONS:
            continue
        graph["edges"][edge_key(src, rel, dst)] = {
            "src": src, "dst": dst, "rel": rel, "weight": 0.5,
            "updated_at": utc_now_iso(),
        }
        added += 1
    return added


def _drop_edges_touching(graph: dict, node_key: str) -> None:
    for ekey, edge in list(graph["edges"].items()):
        if node_key in (edge.get("src"), edge.get("dst")):
            del graph["edges"][ekey]


def prune_orphan_terms(graph: dict) -> int:
    """Drop term nodes with no edges left (dead terminology)."""
    referenced: Set[str] = set()
    for edge in graph["edges"].values():
        referenced.add(edge.get("src"))
        referenced.add(edge.get("dst"))
    dead = [
        k for k, n in graph["nodes"].items()
        if n.get("type") == "term" and k not in referenced
    ]
    for k in dead:
        del graph["nodes"][k]
    return len(dead)


def remove_page(graph: dict, pkey: str) -> None:
    graph["nodes"].pop(pkey, None)
    _drop_edges_touching(graph, pkey)


def prune_artifact_pages(
    graph: dict, artifact_type: str, artifact_id: str, keep_page_ids: Set[str]
) -> int:
    """Remove page nodes of one artifact whose page no longer exists."""
    dead = [
        k for k, n in graph["nodes"].items()
        if n.get("type") == "page"
        and n.get("artifact_type") == artifact_type
        and n.get("artifact_id") == artifact_id
        and n.get("page_id") not in keep_page_ids
    ]
    for k in dead:
        remove_page(graph, k)
    if dead:
        prune_orphan_terms(graph)
    return len(dead)


def enforce_budget(graph: dict) -> None:
    """Cap nodes/edges: weakest (weight, then age) edges are LRU-trimmed."""
    edges = graph["edges"]
    if len(edges) > MAX_EDGES:
        weakest = sorted(
            edges.items(),
            key=lambda kv: (float(kv[1].get("weight") or 0), kv[1].get("updated_at") or ""),
        )[: len(edges) - MAX_EDGES]
        for ekey, _ in weakest:
            del edges[ekey]
    nodes = graph["nodes"]
    if len(nodes) > MAX_NODES:
        degree: Dict[str, int] = {}
        for edge in edges.values():
            degree[edge["src"]] = degree.get(edge["src"], 0) + 1
            degree[edge["dst"]] = degree.get(edge["dst"], 0) + 1
        coldest = sorted(
            nodes.items(),
            key=lambda kv: (degree.get(kv[0], 0), kv[1].get("updated_at") or ""),
        )[: len(nodes) - MAX_NODES]
        for nkey, _ in coldest:
            nodes.pop(nkey, None)
            _drop_edges_touching(graph, nkey)
    prune_orphan_terms(graph)


# ---------------------------------------------------------------------------
# Retrieval helpers (graph-aware recall leg)
# ---------------------------------------------------------------------------
def _query_tokens(query: str) -> Set[str]:
    return {
        t for t in (normalize_term(tok) for tok in _TOKEN_RE.findall(query or ""))
        if len(t) >= MIN_TOKEN_LEN
    }


def resolve_query_terms(query: str, graph: Optional[dict], limit: int = 6) -> List[str]:
    """Lexically match query tokens against term names + aliases.

    Exact token hits win; token-in-name containment is the weak fallback.
    Strong matches are ordered by term degree (most connected first).
    """
    if not graph:
        return []
    tokens = _query_tokens(query)
    if not tokens:
        return []
    degree: Dict[str, int] = {}
    for edge in graph["edges"].values():
        if edge.get("rel") == "mentions":
            degree[edge["src"]] = degree.get(edge["src"], 0) + 1
    strong: List[Tuple[int, str]] = []
    weak: List[str] = []
    for key, node in graph["nodes"].items():
        if node.get("type") != "term":
            continue
        names = {str(node.get("name") or "").lower()}
        names.update(str(a).lower() for a in node.get("aliases") or [])
        if tokens & names:
            strong.append((degree.get(key, 0), key))
            continue
        if len(weak) < limit and any(
            len(t) >= 4 and any(t in n for n in names if len(n) > len(t))
            for t in tokens
        ):
            weak.append(key)
    strong.sort(reverse=True)
    return [k for _, k in strong[:limit]] + weak[: max(0, limit - len(strong))]


def related_source_boosts(
    graph: Optional[dict], term_keys: Sequence[str]
) -> Dict[str, float]:
    """``{chunk source_id: boost weight}`` for pages mentioned by the terms.

    Covers both SPEC-010 page-granular sources (``{artifact_id}::page::{pid}``)
    and legacy blob sources (``{artifact_id}``) of the same artifact.
    """
    if not graph or not term_keys:
        return {}
    wanted = set(term_keys)
    boosts: Dict[str, float] = {}
    for edge in graph["edges"].values():
        if edge.get("rel") != "mentions" or edge.get("src") not in wanted:
            continue
        node = graph["nodes"].get(edge.get("dst") or "")
        if not node or node.get("type") != "page":
            continue
        weight = float(edge.get("weight") or 0.0)
        for source in (page_source_id(node), str(node.get("artifact_id") or "")):
            if source:
                boosts[source] = max(boosts.get(source, 0.0), weight)
    return boosts


def subgraph(
    graph: Optional[dict], focus_term: Optional[str] = None, max_nodes: int = 300
) -> dict:
    """Whole graph (capped) or the 1-hop neighbourhood of ``focus_term``."""
    if not graph:
        return new_graph()
    nodes, edges = graph["nodes"], graph["edges"]
    if focus_term:
        # Direct key hit short-circuits; only a miss pays the full term
        # name/alias scan (S-17).
        target = term_key(focus_term)
        if target not in nodes:
            target = next(
                (
                    k for k, n in nodes.items()
                    if n.get("type") == "term"
                    and str(focus_term).lower() in {str(n.get("name") or "").lower()}
                    | {str(a).lower() for a in n.get("aliases") or []}
                ),
                None,
            )
        if target is None:
            return new_graph()
        keep = {target}
        related: Dict[str, dict] = {}
        for ekey, edge in edges.items():
            if target in (edge.get("src"), edge.get("dst")):
                related[ekey] = edge
                keep.add(edge["src"])
                keep.add(edge["dst"])
        return {"nodes": {k: nodes[k] for k in keep if k in nodes}, "edges": related}
    if len(nodes) <= max_nodes:
        return {"nodes": dict(nodes), "edges": dict(edges)}
    degree: Dict[str, int] = {}
    for edge in edges.values():
        degree[edge["src"]] = degree.get(edge["src"], 0) + 1
        degree[edge["dst"]] = degree.get(edge["dst"], 0) + 1
    keep = set(sorted(nodes, key=lambda k: -degree.get(k, 0))[:max_nodes])
    return {
        "nodes": {k: nodes[k] for k in keep},
        "edges": {
            e: dict(edge) for e, edge in edges.items()
            if edge["src"] in keep and edge["dst"] in keep
        },
    }


# ---------------------------------------------------------------------------
# Storage (knowledge_graphs row, TTL cache, lock-guarded merges)
# ---------------------------------------------------------------------------
def _load_graph_sync(product_id: str) -> Tuple[Optional[dict], int]:
    from api.db import SessionLocal
    from api.models import KnowledgeGraphORM

    with SessionLocal() as db:
        row = db.get(KnowledgeGraphORM, product_id)
        return (dict(row.graph) if isinstance(row and row.graph, dict) else None), int(
            row.version if row else 0
        )


def _save_graph_sync(product_id: str, graph: dict, prev_version: int) -> int:
    from api.db import SessionLocal
    from api.models import KnowledgeGraphORM

    with SessionLocal() as db:
        row = db.get(KnowledgeGraphORM, product_id)
        if row is None:
            row = KnowledgeGraphORM(product_id=product_id, graph=graph, version=1)
            db.add(row)
        else:
            row.graph = graph
            row.version = prev_version + 1
        db.commit()
        return row.version


async def load_product_graph(product_id: str, *, fresh: bool = False) -> Optional[dict]:
    """Read a product's graph (cached ``_GRAPH_CACHE_TTL``); None when absent."""
    if not product_id:
        return None
    if not fresh:
        cached = _graph_cache.get(product_id)
        if cached and time.monotonic() - cached[0] < _GRAPH_CACHE_TTL:
            return cached[1]
    try:
        graph, _version = await asyncio.to_thread(_load_graph_sync, product_id)
    except Exception as e:  # non-fatal: recall falls back to the plain path
        logger.warning("knowledge graph: load failed for %s: %s", product_id, e)
        return None
    _graph_cache[product_id] = (time.monotonic(), graph)
    return graph


async def mutate_product_graph(
    product_id: str, mutator: Callable[[dict], Any]
) -> bool:
    """Read-modify-write one product's graph under the process-wide lock.

    ``mutator(graph)`` mutates in place; a falsy return skips the save.
    """
    async with _graph_write_lock():
        graph, version = await asyncio.to_thread(_load_graph_sync, product_id)
        graph = graph or new_graph()
        try:
            if not mutator(graph):
                return False
            enforce_budget(graph)
            version = await asyncio.to_thread(_save_graph_sync, product_id, graph, version)
        except Exception as e:  # non-fatal: docgen/graph failures never break jobs
            logger.warning("knowledge graph: merge failed for %s: %s", product_id, e)
            return False
    _graph_cache[product_id] = (time.monotonic(), graph)
    logger.info(
        "knowledge graph: product %s updated to v%d (%d nodes, %d edges).",
        product_id, version, len(graph["nodes"]), len(graph["edges"]),
    )
    return True


def clear_graph_cache(product_id: Optional[str] = None) -> None:
    if product_id is None:
        _graph_cache.clear()
    else:
        _graph_cache.pop(product_id, None)


def graph_stats() -> List[Dict[str, Any]]:
    """Per-product node/edge counts for the admin Memory panel (best-effort)."""
    from api.db import SessionLocal
    from api.models import KnowledgeGraphORM

    try:
        with SessionLocal() as db:
            rows = db.query(KnowledgeGraphORM).order_by(KnowledgeGraphORM.product_id).all()
            return [
                {
                    "product_id": r.product_id,
                    "nodes": len((r.graph or {}).get("nodes") or {}),
                    "edges": len((r.graph or {}).get("edges") or {}),
                    "version": int(r.version or 0),
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning("knowledge graph: stats failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# LLM extraction (batched, budgeted, strict-JSON)
# ---------------------------------------------------------------------------
_EXTRACT_PROMPT = """Extract the key domain terms from the documentation pages below.

For every page list its domain terms (concepts, entities, acronyms, component
names — not common words). Then link related terms.
Rules:
- "term": canonical short name (1-4 words);
- "aliases": alternative spellings/abbreviations, if any;
- "description": one short sentence;
- "weight": 0.1-1.0 importance of the term on that page;
- "context": <=15-word quote showing the usage;
- "kind" for relations: synonym | part_of | related.

Return STRICT JSON only, no prose, exactly this shape:
{"pages": {"<page_id>": [{"term": str, "description": str, "aliases": [str], "weight": float, "context": str}]}, "relations": [{"from": str, "to": str, "kind": str}]}

Pages:
"""


def _extraction_prompt(entries: List[Tuple[str, str, str]]) -> str:
    parts = [
        f"### {page_id} — {title or page_id}\n{excerpt}" for page_id, title, excerpt in entries
    ]
    return _EXTRACT_PROMPT + "\n\n".join(parts)


def _parse_extraction(text: Optional[str]) -> Dict[str, Any]:
    """Tolerant strict-JSON parse (thin wrapper over ``_common.extract_json``;
    extraction answers are always objects, so non-dict hits are dropped)."""
    from api.docgen._common import extract_json

    data = extract_json(text)
    return data if isinstance(data, dict) else {}


async def extract_pages_terms(
    pages: Dict[str, Dict[str, Any]],
    llm: Any,
    *,
    batch_size: int,
    concurrency: int,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """Batched LLM extraction over ``{page_id: page dict}``.

    Returns ``({page_id: [term dicts]}, [relation dicts])``; pages whose batch
    failed simply contribute nothing (non-fatal).
    """
    entries = [
        (pid, str(p.get("title") or ""), (p.get("content") or "")[:_EXCERPT_CHARS])
        for pid, p in sorted(pages.items())
        if isinstance(p, dict) and (p.get("content") or "").strip()
    ]
    if not entries:
        return {}, []
    batches = [entries[i:i + max(1, batch_size)] for i in range(0, len(entries), max(1, batch_size))]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(batch: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        async with sem:
            try:
                raw = await llm.generate(_extraction_prompt(batch))
            except Exception as e:  # one failed batch must not kill the rest
                logger.warning("knowledge graph: extraction batch failed: %s", e)
                return {}
            return _parse_extraction(raw)

    results = await asyncio.gather(*[_one(b) for b in batches])
    page_terms: Dict[str, List[dict]] = {}
    relations: List[dict] = []
    known_ids = {pid for pid, _, _ in entries}
    for data in results:
        raw_pages = data.get("pages")
        if isinstance(raw_pages, dict):
            for pid, terms in raw_pages.items():
                if pid in known_ids and isinstance(terms, list):
                    page_terms[pid] = terms[:MAX_TERMS_PER_PAGE]
        raw_rel = data.get("relations")
        if isinstance(raw_rel, list):
            relations.extend(r for r in raw_rel if isinstance(r, dict))
    return page_terms, relations


def _indexable_pages(pages: Any) -> Dict[str, Dict[str, Any]]:
    """Non-empty page dicts of a ``pages`` JSON (SPEC-010 shape)."""
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(pages, dict):
        for pid, page in pages.items():
            if isinstance(page, dict) and (page.get("content") or "").strip():
                out[pid] = page
    return out


def _build_llm() -> Optional[Any]:
    from api.docgen._common import _resolve_docgen_model, _safe_build_llm

    return _safe_build_llm(*_resolve_docgen_model(None))


async def update_graph_from_pages(
    product_id: str,
    artifact_type: str,
    artifact_id: str,
    pages: Any,
    *,
    only: Optional[Set[str]] = None,
    llm: Optional[Any] = None,
) -> bool:
    """SPEC-010 post-indexation hook: merge extracted terms into the graph.

    Only ``only`` pages (changed/new ones) are re-extracted; the artifact's
    dead pages are pruned. Returns True when the graph was persisted.
    """
    if not product_id or not artifact_id:
        return False
    indexable = _indexable_pages(pages)
    if not indexable:
        return False
    scope = (
        {pid: indexable[pid] for pid in only or () if pid in indexable}
        if only is not None else indexable
    )
    from api.config.timeout import resolve_timeout, resolve_timeout_int

    owned = llm is None
    llm = llm or _build_llm()
    if llm is None:
        logger.info("knowledge graph: LLM unavailable; extraction skipped (%s).", artifact_id)
        return False
    try:
        page_terms, relations = {}, []
        try:
            page_terms, relations = await asyncio.wait_for(
                extract_pages_terms(
                    scope, llm,
                    batch_size=resolve_timeout_int("knowledge_graph_batch"),
                    concurrency=resolve_timeout_int("docgen_llm_concurrency"),
                ),
                timeout=resolve_timeout("knowledge_graph_build"),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "knowledge graph: extraction timed out for %s; graph unchanged.", artifact_id
            )
            return False

        def _merge(graph: dict) -> bool:
            for pid, page in indexable.items():
                upsert_page(
                    graph, artifact_type, artifact_id, pid, title=str(page.get("title") or "")
                )
            for pid in scope:
                set_page_terms(
                    graph,
                    page_key(artifact_type, artifact_id, pid),
                    page_terms.get(pid) or [],
                )
            add_term_relations(graph, relations)
            prune_artifact_pages(graph, artifact_type, artifact_id, set(indexable))
            prune_orphan_terms(graph)
            return True

        return await mutate_product_graph(product_id, _merge)
    finally:
        if owned:
            from api.docgen._common import _safe_aclose

            await _safe_aclose(llm)


async def rebuild_product_graph(
    product_id: str, *, llm: Optional[Any] = None
) -> Dict[str, Any]:
    """Full graph rebuild from all page-granular artifacts (admin action)."""
    from api.config.timeout import resolve_timeout, resolve_timeout_bool, resolve_timeout_int
    from api.db import SessionLocal
    from api.models import ProductORM
    from sqlalchemy.orm import selectinload

    if not resolve_timeout_bool("knowledge_graph_enabled"):
        return {"success": False, "message": "Knowledge graph extraction is disabled", "product_id": product_id}

    def _collect() -> List[Tuple[str, str, str, Dict[str, Dict[str, Any]]]]:
        with SessionLocal() as db:
            product = (
                db.query(ProductORM)
                .options(
                    selectinload(ProductORM.codebases),
                    selectinload(ProductORM.databases),
                )
                .filter(ProductORM.id == product_id)
                .first()
            )
            if product is None:
                return []
            out = []
            for attr, atype in (("codebases", "codebase"), ("databases", "database")):
                for art in getattr(product, attr) or []:
                    pages = _indexable_pages(getattr(art, "pages", None))
                    if pages:
                        out.append((atype, str(art.id), str(art.name or ""), pages))
            return out

    try:
        artifacts = await asyncio.to_thread(_collect)
    except Exception as e:
        logger.warning("knowledge graph: rebuild load failed for %s: %s", product_id, e)
        return {"success": False, "message": "Rebuild failed", "product_id": product_id}
    if not artifacts:
        return {
            "success": True,
            "product_id": product_id,
            "message": "No page-granular artifacts to index",
            "nodes": 0, "edges": 0, "pages": 0,
        }

    owned = llm is None
    llm = llm or _build_llm()
    if llm is None:
        return {"success": False, "message": "LLM unavailable", "product_id": product_id}
    try:
        graph = new_graph()
        try:
            for artifact_type, artifact_id, _name, pages in artifacts:
                page_terms, relations = await asyncio.wait_for(
                    extract_pages_terms(
                        pages, llm,
                        batch_size=resolve_timeout_int("knowledge_graph_batch"),
                        concurrency=resolve_timeout_int("docgen_llm_concurrency"),
                    ),
                    timeout=resolve_timeout("knowledge_graph_build"),
                )
                for pid, page in pages.items():
                    upsert_page(
                        graph, artifact_type, artifact_id, pid,
                        title=str(page.get("title") or ""),
                    )
                    set_page_terms(
                        graph,
                        page_key(artifact_type, artifact_id, pid),
                        page_terms.get(pid) or [],
                    )
                add_term_relations(graph, relations)
        except asyncio.TimeoutError:
            logger.warning(
                "knowledge graph: rebuild timed out for %s; keeping previous graph.",
                product_id,
            )
            return {"success": False, "message": "Rebuild timed out", "product_id": product_id}
        enforce_budget(graph)

        # Content replacement rides mutate_product_graph: the write lock plus
        # a FRESH version read, so a concurrent post-indexation merge between
        # extraction and save is neither lost nor clobbered (Q-4).
        def _replace(target: dict) -> bool:
            target.clear()
            target.update(graph)
            return True

        if not await mutate_product_graph(product_id, _replace):
            return {"success": False, "message": "Rebuild failed", "product_id": product_id}
        logger.info(
            "knowledge graph: rebuilt product %s (%d nodes, %d edges).",
            product_id, len(graph["nodes"]), len(graph["edges"]),
        )
        return {
            "success": True,
            "product_id": product_id,
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "pages": sum(len(p) for _, _, _, p in artifacts),
        }
    finally:
        if owned:
            from api.docgen._common import _safe_aclose

            await _safe_aclose(llm)


__all__ = [
    "MAX_NODES",
    "MAX_EDGES",
    "TERM_RELATIONS",
    "add_term_relations",
    "clear_graph_cache",
    "edge_key",
    "enforce_budget",
    "extract_pages_terms",
    "graph_stats",
    "load_product_graph",
    "mutate_product_graph",
    "new_graph",
    "normalize_term",
    "page_key",
    "page_source_id",
    "prune_artifact_pages",
    "prune_orphan_terms",
    "rebuild_product_graph",
    "related_source_boosts",
    "remove_page",
    "resolve_query_terms",
    "set_page_terms",
    "subgraph",
    "term_key",
    "update_graph_from_pages",
    "upsert_page",
    "upsert_term",
]
