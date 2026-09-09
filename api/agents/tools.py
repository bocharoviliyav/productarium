"""Expert-agent tools over product knowledge (LangChain/LangGraph Wave B).

Each tool factory takes ``product_id`` (plus an optional SQLAlchemy session
factory) and returns a list of LangChain tools bound to that product scope.
The LLM never sees or controls ``product_id`` — it is captured in a closure
at tool-construction time, so the agent physically cannot read another
product's data regardless of what it passes as tool arguments.

Tool contracts (all EN descriptions, per the Wave B requirements):
- ``knowledge_recall`` — semantic recall over the product's pgvector memory
  (``api.memory.query_memory``). Returns chunks with a citation header
  (source type/id + chunk id) per hit so the agent can cite sources.
- ``codebase_file_read`` — reads a file from a locally cloned codebase repo.
  Path traversal is blocked by construction: the requested path is resolved
  against the clone root and must remain INSIDE it (symlinks included).
  VCS metadata (``.git``) and credential-looking files (``.env*``, ``*.pem``,
  private keys) are never exposed to the agent. Local (non-http) clone roots
  must live under the managed clone root unless ``PRODUCTARIUM_ALLOW_LOCAL_CLONES``
  is set explicitly (local-dev escape hatch).
  Returns a polite in-result error when no clone exists.
- ``spec_read`` / ``link_read`` / ``node_read`` — read a spec / link set /
  knowledge node of the product from the DB by name or slug.

All tools return plain strings (LLM-friendly) and never raise into the agent
graph: errors are returned as in-result messages so the model can recover.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from sqlalchemy.orm import Session, sessionmaker

from api.utils.fs import open_read_nofollow

logger = logging.getLogger(__name__)

#: Default cap for knowledge-recall results handed to the agent.
DEFAULT_RECALL_TOP_K = 8
#: Maximum characters of a file body returned by ``codebase_file_read``.
MAX_FILE_CHARS = 200_000
#: Maximum characters of a spec/link/node body returned to the agent.
MAX_ENTITY_CHARS = 200_000

#: Indicators that a local path is not a safe clone root (UNC, rooted).
_UNSAFE_PATH_PREFIXES = ("//", "\\\\")

#: VCS metadata directories: never agent-readable (``.git/config`` persists
#: the token-authed clone URL).
_FORBIDDEN_SEGMENTS = frozenset({".git", ".hg", ".svn"})

#: Lowercase basenames that commonly hold credentials.
_SECRET_BASENAMES = frozenset({
    ".netrc", ".npmrc", ".pypirc",
    "credentials", "credentials.json", "credentials.yaml", "credentials.yml",
    "secrets.json", "secrets.yaml", "secrets.yml",
})

#: Lowercase basename prefixes of credential files (``.env``, ``.env.local``,
#: ``id_rsa``, ``id_ed25519``, ...).
_SECRET_PREFIXES = (".env", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials")

#: Extensions of key/certificate stores.
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore")

#: Env var that re-enables arbitrary LOCAL clone roots for dev setups.
ALLOW_LOCAL_CLONES_ENV = "PRODUCTARIUM_ALLOW_LOCAL_CLONES"


def _is_forbidden_agent_path(rel: str) -> bool:
    """True for VCS metadata / credential-looking paths (never agent-readable)."""
    parts = [p for p in rel.split("/") if p and p != "."]
    if not parts:
        return True
    if any(p.lower() in _FORBIDDEN_SEGMENTS for p in parts):
        return True
    name = parts[-1].lower()
    return (
        name.startswith(_SECRET_PREFIXES)
        or name in _SECRET_BASENAMES
        or name.endswith(_SECRET_SUFFIXES)
    )


@dataclass(frozen=True)
class KnowledgeHit:
    """A single recalled knowledge chunk with its citation info."""

    chunk_id: str
    source_type: str
    source_id: Optional[str]
    content: str
    score: Optional[float] = None


def _new_session_factory(
    session_factory: Optional[sessionmaker],
) -> Callable[[], Session]:
    """Return a zero-arg callable producing a short-lived DB session.

    Falls back to the global ``api.db.SessionLocal`` so tests can either pass
    their own factory or monkeypatch ``api.db.SessionLocal``.
    """
    if session_factory is not None:
        return session_factory
    from api.db import SessionLocal

    return SessionLocal


def _session_or_none() -> Optional[Session]:
    """Open a global SessionLocal; return None on failure (non-fatal)."""
    try:
        from api.db import SessionLocal

        return SessionLocal()
    except Exception as e:  # pragma: no cover - import/DB wiring failure
        logger.warning("agent tools: could not open a DB session: %s", e)
        return None


async def search_knowledge_hits(
    query: str,
    product_id: str,
    top_k: int = DEFAULT_RECALL_TOP_K,
) -> List[KnowledgeHit]:
    """Search product knowledge and return hits with citation metadata.

    Runs the SQL cosine-distance search over ``knowledge_chunks`` directly
    (the pgvector path). On non-Postgres backends (SQLite tests), falls back
    to the most recent chunks for the product so the tool still returns
    usable, citable evidence. Non-fatal: returns ``[]`` on any error.
    """
    if not query or not query.strip() or not product_id:
        return []
    try:
        return await _search_pgvector(query, product_id, top_k)
    except Exception as e:
        logger.debug(
            "agent tools: pgvector knowledge search failed (%s); "
            "falling back to recent chunks.",
            e,
        )
    try:
        return _recent_chunks(product_id, top_k)
    except Exception as e:  # pragma: no cover - DB unavailable
        logger.warning(
            "agent tools: knowledge search fallback failed for product %s: %s",
            product_id,
            e,
        )
        return []


async def _search_pgvector(
    query: str, product_id: str, top_k: int
) -> List[KnowledgeHit]:
    """Cosine search over knowledge_chunks (Postgres + pgvector only).

    Raises on any error (non-Postgres dialect, missing extension, DB down) so
    the caller can fall back; ``query_memory`` is used as the embedder/SQL
    driver through the pgvector-capable path only.
    """
    try:
        from api.memory.pgvector_backend import _embed_query, _is_pgvector_capable

        if not _is_pgvector_capable():
            raise RuntimeError("pgvector not available for the active DB")
    except Exception:
        raise
    qvec = await _embed_query(query)
    if not qvec:
        # Embedder unavailable: fall back to recent chunks instead of "".
        raise RuntimeError("query embedding unavailable")
    k = max(1, min(int(top_k or DEFAULT_RECALL_TOP_K), 100))
    from sqlalchemy import text

    sql = text(
        "SELECT id, source_type, source_id, content, "
        "1 - (embedding <=> CAST(:q AS vector)) AS score "
        "FROM knowledge_chunks "
        "WHERE product_id = :pid "
        "ORDER BY embedding <=> CAST(:q AS vector) "
        "LIMIT :k"
    )
    vec_literal = "[" + ",".join(str(float(x)) for x in qvec) + "]"
    conn = None
    try:
        from api.db import engine

        conn = engine.connect()
        rows = conn.execute(
            sql, {"pid": product_id, "q": vec_literal, "k": k}
        ).fetchall()
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - defensive
                pass
    hits: List[KnowledgeHit] = []
    for row in rows or []:
        content = row[3]
        if not content:
            continue
        score = row[4]
        hits.append(
            KnowledgeHit(
                chunk_id=str(row[0]),
                source_type=str(row[1] or "unknown"),
                source_id=str(row[2]) if row[2] else None,
                content=str(content),
                score=float(score) if score is not None else None,
            )
        )
    return hits


def _recent_chunks(product_id: str, top_k: int) -> List[KnowledgeHit]:
    """Most-recent chunks for a product (SQLite/fallback recall path)."""
    from api.models import KnowledgeChunkORM

    session = _session_or_none()
    if session is None:
        return []
    try:
        with session:
            rows = (
                session.query(KnowledgeChunkORM)
                .filter(KnowledgeChunkORM.product_id == product_id)
                .order_by(
                    KnowledgeChunkORM.created_at.desc(),
                    KnowledgeChunkORM.chunk_index,
                )
                .limit(max(1, min(int(top_k or DEFAULT_RECALL_TOP_K), 100)))
                .all()
            )
            return [
                KnowledgeHit(
                    chunk_id=str(r.id),
                    source_type=str(r.source_type or "unknown"),
                    source_id=str(r.source_id) if r.source_id else None,
                    content=str(r.content or ""),
                )
                for r in rows
            ]
    finally:
        session.close()


def format_hits_with_citations(hits: List[KnowledgeHit]) -> str:
    """Render recall hits as a citation-headed block for the agent.

    Each hit is prefixed with ``[source: <source_type>:<source_id> chunk=<id>]``
    so the model can attribute every fact it uses.
    """
    if not hits:
        return (
            "No indexed knowledge matched the query. Try other tools "
            "(spec_read, node_read, link_read, codebase_file_read) or state "
            "that no evidence was found."
        )
    parts: List[str] = []
    for i, hit in enumerate(hits, start=1):
        source = hit.source_type
        if hit.source_id:
            source += f":{hit.source_id}"
        score = f" score={hit.score:.3f}" if hit.score is not None else ""
        parts.append(
            f"[{i}] [source: {source} chunk={hit.chunk_id}{score}]\n{hit.content}"
        )
    return "\n\n".join(parts)


def _managed_repo_root() -> str:
    """Realpath of the managed clone root (``<DEFAULT_REPO_ROOT>/repos``)."""
    from api.repositories.documents import DEFAULT_REPO_ROOT

    return os.path.realpath(os.path.join(DEFAULT_REPO_ROOT, "repos"))


def _local_clones_allowed() -> bool:
    """True when arbitrary local clone roots are explicitly allowed (dev)."""
    return os.environ.get(ALLOW_LOCAL_CLONES_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _is_managed_clone(candidate: str) -> bool:
    """True when ``candidate`` resolves inside the managed clone root."""
    try:
        root = _managed_repo_root()
        return os.path.commonpath([root, os.path.realpath(candidate)]) == root
    except (ValueError, OSError):
        return False


def _clone_root_for(session: Session, codebase: Any) -> Optional[str]:
    """Resolve the local clone root directory for a codebase row.

    Returns ``None`` when the codebase has no usable ``repo_url``/``repo_type``
    or when a local (non-http) ``repo_url`` points outside the managed clone
    root — arbitrary host directories must not become agent-readable roots
    unless ``PRODUCTARIUM_ALLOW_LOCAL_CLONES`` is explicitly set.
    """
    from api.integrations._git_base import GitConnector
    from api.repositories.documents import DEFAULT_REPO_ROOT

    repo_url = (getattr(codebase, "repo_url", None) or "").strip()
    repo_type = (getattr(codebase, "repo_type", None) or "").strip() or None
    if not repo_url:
        return None
    # Local path clone: allowed only under the managed clone root (or via
    # the explicit dev escape hatch).
    if not repo_url.startswith(("http://", "https://")):
        if not (_is_managed_clone(repo_url) or _local_clones_allowed()):
            logger.warning(
                "agent tools: local repo_url outside the managed clone root "
                "is not readable (%s)",
                repo_url,
            )
            return None
        return repo_url if os.path.isdir(repo_url) else None
    repo_name = GitConnector.extract_repo_name(repo_url, repo_type or "")
    return os.path.join(DEFAULT_REPO_ROOT, "repos", repo_name)


def _resolve_inside_root(root: str, rel_path: str) -> Optional[str]:
    """Resolve ``rel_path`` under ``root``, or None if it escapes the root.

    Defense against path traversal (including ``..`` segments, absolute paths,
    and symlinks pointing outside the clone): the real path of the resolved
    target must be inside the real path of the root directory.
    """
    if not root or not rel_path:
        return None
    # Absolute paths and Windows drive letters are rejected outright: the
    # agent may only address paths RELATIVE to the clone root.
    if os.path.isabs(rel_path) or rel_path.startswith(_UNSAFE_PATH_PREFIXES):
        return None
    if len(rel_path) > 2 and rel_path[1] == ":":
        return None
    try:
        root_real = os.path.realpath(root)
        candidate = os.path.realpath(os.path.join(root_real, rel_path))
        # commonpath is the containment check (a prefix-string check would
        # wrongly accept ``clonex`` next to ``clone``).
        if os.path.commonpath([root_real, candidate]) != root_real:
            return None
        return candidate
    except (ValueError, OSError):
        return None


def _find_codebase_with_clone(session: Session, product_id: str) -> List[Any]:
    """Return the product's codebases that have a local clone root."""
    from api.models import CodebaseORM

    rows = (
        session.query(CodebaseORM)
        .filter(CodebaseORM.product_id == product_id)
        .all()
    )
    out = []
    for row in rows:
        root = _clone_root_for(session, row)
        if root and os.path.isdir(root):
            out.append(row)
    return out


async def _read_codebase_file(
    session_factory: Callable[[], Session],
    product_id: str,
    path: str,
) -> str:
    """Implementation of the codebase_file_read tool (never raises)."""
    if not path or not str(path).strip():
        return "Error: 'path' is required (a file path relative to the codebase clone root)."
    raw = str(path).strip()
    if raw.startswith(("/", "\\\\")) or (len(raw) > 2 and raw[1] == ":"):
        return (
            "Error: absolute paths are not allowed. Use a path relative to "
            "the codebase clone root (e.g. 'src/main.py')."
        )
    rel = raw.replace("\\", "/").lstrip("/")
    if _is_forbidden_agent_path(rel):
        return (
            "Error: this path is not readable by the agent. Git metadata "
            "and credential/key files are excluded from codebase reads."
        )

    session = session_factory()
    try:
        with session:
            codebases = _find_codebase_with_clone(session, product_id)
            if not codebases:
                return (
                    "No local codebase clone is available for this product. "
                    "The repository may not have been cloned yet (generate "
                    "documentation for the codebase first)."
                )
            for codebase in codebases:
                root = _clone_root_for(session, codebase)
                if not root:
                    continue
                resolved = _resolve_inside_root(root, rel)
                if resolved is None:
                    # Try the next clone; if none matches we report the
                    # traversal rejection below (per the last clone root).
                    continue
                if not os.path.isfile(resolved):
                    continue
                try:
                    # O_NOFOLLOW: a symlink swapped onto the final component
                    # after the realpath confinement check must not be
                    # followed outside the clone (TOCTOU hardening).
                    with open_read_nofollow(resolved, errors="replace") as f:
                        body = f.read(MAX_FILE_CHARS + 1)
                except OSError as e:
                    return f"Error: could not read the file ({e.__class__.__name__})."
                truncated = ""
                if len(body) > MAX_FILE_CHARS:
                    body = body[:MAX_FILE_CHARS]
                    truncated = "\n... (truncated)"
                name = getattr(codebase, "name", None) or "codebase"
                return (
                    f"[source: codebase {name} file={rel}]\n{body}{truncated}"
                )
        # Nothing matched: distinguish traversal rejection from missing file.
        for codebase in codebases:
            root = _clone_root_for(session, codebase)
            if root and _resolve_inside_root(root, rel) is None:
                return (
                    "Error: path escapes the codebase clone directory and was "
                    "rejected for security reasons. Use a relative path that "
                    "stays inside the repository."
                )
        return f"File not found in any cloned codebase of this product: {rel}"
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "agent tools: codebase_file_read failed for product %s: %s",
            product_id,
            e,
        )
        return f"Error: reading the codebase file failed ({e.__class__.__name__})."
    finally:
        session.close()


def _match_by_name(rows: List[Any], value: str) -> Optional[Any]:
    """Case-insensitive exact-then-prefix match on ``name``/``title``/``slug``."""
    target = value.strip().lower()
    if not target:
        return None
    for attr in ("name", "title"):
        for row in rows:
            if str(getattr(row, attr, "") or "").strip().lower() == target:
                return row
    for attr in ("slug", "name", "title"):
        for row in rows:
            if str(getattr(row, attr, "") or "").strip().lower().startswith(target):
                return row
    return None


def _cap_body(body: str) -> str:
    """Cap an entity body, marking truncation for the agent."""
    if len(body) <= MAX_ENTITY_CHARS:
        return body
    return body[:MAX_ENTITY_CHARS] + "\n... (truncated)"


def build_expert_tools(
    product_id: str,
    session_factory: Optional[sessionmaker] = None,
) -> List[Any]:
    """Build the expert agent's product-scoped tools.

    Args:
        product_id: The product every tool is hard-scoped to. Bound in a
            closure — the LLM cannot influence it.
        session_factory: Optional SQLAlchemy session factory (tests); defaults
            to the global ``api.db.SessionLocal``.

    Returns:
        A list of LangChain tool objects (``knowledge_recall``,
        ``codebase_file_read``, ``spec_read``, ``link_read``, ``node_read``).
    """
    if not product_id:
        raise ValueError("product_id is required to build expert tools")
    factory = _new_session_factory(session_factory)

    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _RecallArgs(BaseModel):
        query: str = Field(
            ..., description="The search query (a phrase or question)."
        )
        top_k: int = Field(
            8,
            ge=1,
            le=50,
            description="How many chunks to return (1-50, default 8).",
        )

    async def _recall(query: str, top_k: int = DEFAULT_RECALL_TOP_K) -> str:
        hits = await search_knowledge_hits(query, product_id, top_k)
        logger.info(
            "expert tool knowledge_recall: product=%s hits=%d", product_id, len(hits)
        )
        return format_hits_with_citations(hits)

    recall_tool = StructuredTool.from_function(
        coroutine=_recall,
        name="knowledge_recall",
        description=(
            "Semantic search over this product's indexed knowledge "
            "(generated codebase docs, specs, links, knowledge pages). "
            "Returns cited chunks; start here for most questions."
        ),
        args_schema=_RecallArgs,
    )

    class _FileReadArgs(BaseModel):
        path: str = Field(
            ...,
            description="File path relative to the codebase clone root, e.g. 'src/main.py'.",
        )

    async def _file_read(path: str) -> str:
        return await _read_codebase_file(factory, product_id, path)

    file_tool = StructuredTool.from_function(
        coroutine=_file_read,
        name="codebase_file_read",
        description=(
            "Read one source file from this product's locally cloned "
            "codebase repository. The path must be relative to the repository "
            "root; paths escaping the clone are rejected."
        ),
        args_schema=_FileReadArgs,
    )

    class _NameArgs(BaseModel):
        name: str = Field(
            ...,
            description="Exact (or unique prefix of the) spec name to read.",
        )

    async def _spec_read(name: str) -> str:
        session = factory()
        try:
            with session:
                from api.models import SpecORM

                rows = (
                    session.query(SpecORM)
                    .filter(SpecORM.product_id == product_id)
                    .all()
                )
                if not rows:
                    return "No specs are attached to this product."
                row = _match_by_name(rows, name)
                if row is None:
                    available = ", ".join(sorted(str(r.name) for r in rows))
                    return (
                        f"No spec named {name!r} was found in this product. "
                        f"Available specs: {available}."
                    )
                kind = getattr(row, "kind", None) or "openapi"
                content = getattr(row, "content", None) or ""
                return (
                    f"[source: spec {row.name} kind={kind}]\n{_cap_body(content)}"
                )
        finally:
            session.close()

    spec_tool = StructuredTool.from_function(
        coroutine=_spec_read,
        name="spec_read",
        description=(
            "Read an OpenAPI/AsyncAPI spec attached to this product by name. "
            "Returns the raw YAML/JSON content."
        ),
        args_schema=_NameArgs,
    )

    class _LinkArgs(BaseModel):
        name: str = Field(
            ...,
            description="Exact (or unique prefix of the) link set name to read.",
        )

    async def _link_read(name: str) -> str:
        session = factory()
        try:
            with session:
                from api.models import LinksORM

                rows = (
                    session.query(LinksORM)
                    .filter(LinksORM.product_id == product_id)
                    .all()
                )
                if not rows:
                    return "No link collections are attached to this product."
                row = _match_by_name(rows, name)
                if row is None:
                    available = ", ".join(sorted(str(r.name) for r in rows))
                    return (
                        f"No link collection named {name!r} was found in this "
                        f"product. Available: {available}."
                    )
                content = getattr(row, "content", None) or ""
                return (
                    f"[source: links {row.name}]\n{_cap_body(content)}"
                )
        finally:
            session.close()

    link_tool = StructuredTool.from_function(
        coroutine=_link_read,
        name="link_read",
        description=(
            "Read a collection of curated external links attached to this "
            "product by name. Returns the raw JSON content."
        ),
        args_schema=_LinkArgs,
    )

    class _NodeArgs(BaseModel):
        title: Optional[str] = Field(
            default=None,
            description="Knowledge node title to read (exact or unique prefix).",
        )
        slug: Optional[str] = Field(
            default=None,
            description="Knowledge node slug to read (exact or unique prefix).",
        )

    async def _node_read(title: Optional[str], slug: Optional[str]) -> str:
        if not (title or slug):
            return "Error: provide 'title' or 'slug' to identify the knowledge node."
        session = factory()
        try:
            with session:
                from api.models import KnowledgeNodeORM

                rows = (
                    session.query(KnowledgeNodeORM)
                    .filter(KnowledgeNodeORM.product_id == product_id)
                    .all()
                )
                if not rows:
                    return "No knowledge nodes exist for this product."
                row = None
                if slug:
                    target = slug.strip().lower()
                    for r in rows:
                        if str(r.slug or "").strip().lower() == target:
                            row = r
                            break
                if row is None and title:
                    row = _match_by_name(rows, title)
                if row is None:
                    titles = ", ".join(sorted(str(r.title) for r in rows))
                    return (
                        "No knowledge node matched the given title/slug in "
                        f"this product. Available nodes: {titles}."
                    )
                content = getattr(row, "content_md", None) or ""
                node_type = getattr(row, "node_type", None) or "page"
                return (
                    f"[source: knowledge node {row.title} type={node_type} "
                    f"slug={row.slug}]\n{_cap_body(content)}"
                )
        finally:
            session.close()

    node_tool = StructuredTool.from_function(
        coroutine=_node_read,
        name="node_read",
        description=(
            "Read a knowledge-tree page (Confluence-like node) of this "
            "product by title or slug. Returns the Markdown content."
        ),
        args_schema=_NodeArgs,
    )

    return [recall_tool, file_tool, spec_tool, link_tool, node_tool]


__all__ = [
    "DEFAULT_RECALL_TOP_K",
    "KnowledgeHit",
    "MAX_ENTITY_CHARS",
    "MAX_FILE_CHARS",
    "build_expert_tools",
    "format_hits_with_citations",
    "search_knowledge_hits",
]
