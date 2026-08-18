"""Internal HTTP endpoints that back the RLM retrieval tools.

The RLM agent runs inside a Pyodide REPL and cannot import host modules
(SQLAlchemy/psycopg) or touch the host filesystem directly. Its tools reach
product data by calling these localhost-only endpoints over HTTP via
``pyodide.http.open_url``. Each endpoint does the host-side DB / filesystem
work and returns JSON.

Security: every endpoint rejects any request whose Host header is not a
loopback address (``127.0.0.1`` / ``localhost`` / ``::1``), so the endpoints
are unreachable from the network even though they sit on the public FastAPI
app. No auth: the RLM REPL has no credentials, and the data is the product's
own indexed knowledge (already reachable via the authenticated public API).

Auto-discovered by ``api.routers.include_all_routers`` (module-level
``router = APIRouter(...)``).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from api.db import SessionLocal
from api.models import (
    CodebaseORM,
    KnowledgeNodeORM,
    LinksORM,
    ProductORM,
    SpecORM,
)
from api.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/internal/rlm", tags=["rlm-tools-internal"])

# Loopback hostnames/ips that are allowed to call these internal endpoints.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

# Cap the number of file-tree entries / search matches returned so a giant
# repo cannot produce a multi-MB JSON payload that overflows the Pyodide REPL.
_MAX_FILE_TREE = 5000
_MAX_SEARCH_MATCHES = 200
# Cap a single file's contents so a huge generated/bundled file does not
# overflow the REPL. The agent can still request other files.
_MAX_FILE_CHARS = 200_000


def _enforce_loopback(request: Request) -> None:
    """Reject any non-loopback caller. Returns 404 (not 403) to avoid leaking
    the endpoint's existence to network scanners."""
    client = (request.client.host if request.client else "") or ""
    # client.host is the TCP peer. For a forwarded/proxied request this would
    # be the proxy, but these endpoints are meant for same-process RLM calls,
    # so the peer must be loopback.
    host = client.lower()
    # Also check the Host header (defence in depth): a network request would
    # carry the external host, not localhost.
    host_header = (request.headers.get("host") or "").split(":")[0].lower()
    if host in _LOOPBACK_HOSTS and host_header in _LOOPBACK_HOSTS:
        return
    raise HTTPException(status_code=404, detail="Not Found")


def _resolve_repo_dir(codebase_id: str) -> Optional[str]:
    """Resolve the on-disk clone dir for a codebase by its id.

    Looks up the ``CodebaseORM.repo_url`` and derives the adalflow repo path
    (mirrors ``DatabaseManager._create_repo``'s path scheme:
    ``~/.adalflow/repos/{owner}_{repo}`` for GitHub/GitLab URLs).
    Returns ``None`` if the codebase/dir is missing.
    """
    try:
        with SessionLocal() as db:
            cb = db.get(CodebaseORM, codebase_id)
            if cb is None:
                return None
            repo_url = (getattr(cb, "repo_url", "") or "").strip()
            repo_type = getattr(cb, "repo_type", None) or "github"
        if not repo_url:
            return None
        try:
            from adalflow.utils import get_adalflow_default_root_path
        except Exception:  # pragma: no cover - adalflow optional in tests
            return None
        root = get_adalflow_default_root_path()
        url_parts = repo_url.rstrip("/").split("/")
        if repo_type in ("github", "gitlab") and len(url_parts) >= 5:
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        repo_dir = os.path.join(root, "repos", repo_name)
        if os.path.isdir(repo_dir):
            return repo_dir
        return None
    except Exception as e:
        logger.debug("rlm_tools: resolve_repo_dir failed for %s: %s", codebase_id, e)
        return None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/files")
def list_codebase_files(
    request: Request,
    codebase_id: str = Query(...),
):
    """Return the relative file tree of the cloned repo for ``codebase_id``."""
    _enforce_loopback(request)
    repo_dir = _resolve_repo_dir(codebase_id)
    if not repo_dir:
        return JSONResponse({"files": []})
    files: List[str] = []
    # Mirror the DEFAULT_EXCLUDED_DIRS used by read_all_documents so the tree
    # matches what was indexed (skip node_modules/.git/__pycache__/etc).
    excluded = {
        ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
        ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
    }
    try:
        for dirpath, dirnames, filenames in os.walk(repo_dir):
            dirnames[:] = [d for d in dirnames if d not in excluded]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, repo_dir)
                files.append(rel.replace(os.sep, "/"))
                if len(files) >= _MAX_FILE_TREE:
                    break
            if len(files) >= _MAX_FILE_TREE:
                break
    except Exception as e:
        logger.debug("rlm_tools: list_codebase_files walk failed: %s", e)
        return JSONResponse({"files": []})
    return JSONResponse({"files": sorted(files)})


@router.get("/file")
def read_codebase_file(
    request: Request,
    codebase_id: str = Query(...),
    path: str = Query(...),
):
    """Return the full contents of one file in the cloned repo."""
    _enforce_loopback(request)
    repo_dir = _resolve_repo_dir(codebase_id)
    if not repo_dir:
        return JSONResponse({"path": path, "content": "", "error": "repo not found"})
    # Prevent path traversal: normalize and ensure the resolved path stays
    # inside repo_dir.
    safe_path = os.path.normpath(os.path.join(repo_dir, path))
    if not safe_path.startswith(repo_dir + os.sep) and safe_path != repo_dir:
        return JSONResponse({"path": path, "content": "", "error": "path outside repo"})
    if not os.path.isfile(safe_path):
        return JSONResponse({"path": path, "content": "", "error": "file not found"})
    try:
        with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(_MAX_FILE_CHARS)
        return JSONResponse({"path": path, "content": content})
    except Exception as e:
        logger.debug("rlm_tools: read_codebase_file failed for %s: %s", path, e)
        return JSONResponse({"path": path, "content": "", "error": str(e)})


@router.get("/search_code")
def search_code(
    request: Request,
    codebase_id: str = Query(...),
    pattern: str = Query(...),
):
    """Grep ``pattern`` (regex) across the cloned repo; return matching lines."""
    _enforce_loopback(request)
    repo_dir = _resolve_repo_dir(codebase_id)
    if not repo_dir:
        return JSONResponse({"matches": []})
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return JSONResponse({"matches": [], "error": f"bad regex: {e}"})
    excluded = {
        ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
        ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
    }
    matches: List[Dict[str, Any]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(repo_dir):
            dirnames[:] = [d for d in dirnames if d not in excluded]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, repo_dir).replace(os.sep, "/")
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        for lineno, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append({
                                    "path": rel,
                                    "line": lineno,
                                    "text": line.rstrip("\n")[:500],
                                })
                                if len(matches) >= _MAX_SEARCH_MATCHES:
                                    return JSONResponse({"matches": matches})
                except Exception:
                    continue
    except Exception as e:
        logger.debug("rlm_tools: search_code walk failed: %s", e)
        return JSONResponse({"matches": []})
    return JSONResponse({"matches": matches})


@router.get("/knowledge")
async def search_knowledge(
    request: Request,
    product_id: str = Query(...),
    query: str = Query(...),
    top_k: int = Query(20, ge=1, le=100),
):
    """Semantic recall over the product's indexed knowledge (pgvector/cognee)."""
    _enforce_loopback(request)
    try:
        from api.memory import query_memory

        ctx = await query_memory(query, product_id, top_k=top_k)
        # query_memory returns joined text; return it as a single slice plus a
        # fallback flag so the agent knows whether to ask for artifact docs.
        return JSONResponse({
            "text": ctx or "",
            "sources": [],
            "empty": not bool(ctx),
        })
    except Exception as e:
        logger.debug("rlm_tools: search_knowledge failed: %s", e)
        return JSONResponse({"text": "", "sources": [], "empty": True, "error": str(e)})


def _product_or_404(product_id: str):
    """Load a product with its codebases/specs/links eagerly; None if missing."""
    try:
        from sqlalchemy.orm import selectinload

        with SessionLocal() as db:
            p = (
                db.query(ProductORM)
                .options(
                    selectinload(ProductORM.codebases),
                    selectinload(ProductORM.specs),
                    selectinload(ProductORM.links),
                )
                .filter(ProductORM.id == product_id)
                .first()
            )
            return p
    except Exception as e:
        logger.debug("rlm_tools: product lookup failed for %s: %s", product_id, e)
        return None


@router.get("/specs")
def get_specs(
    request: Request,
    product_id: str = Query(...),
):
    """Return all specs for a product (name, kind, content)."""
    _enforce_loopback(request)
    p = _product_or_404(product_id)
    if p is None:
        return JSONResponse({"specs": []})
    specs = []
    for s in p.specs:
        specs.append({
            "id": s.id,
            "name": s.name,
            "kind": getattr(s, "kind", None),
            "content": getattr(s, "content", "") or "",
        })
    return JSONResponse({"specs": specs})


@router.get("/links")
def get_links(
    request: Request,
    product_id: str = Query(...),
):
    """Return all links for a product (name, content JSON)."""
    _enforce_loopback(request)
    p = _product_or_404(product_id)
    if p is None:
        return JSONResponse({"links": []})
    links = []
    for l in p.links:
        links.append({
            "id": l.id,
            "name": l.name,
            "content": getattr(l, "content", "") or "",
        })
    return JSONResponse({"links": links})


@router.get("/nodes")
def get_knowledge_nodes(
    request: Request,
    product_id: str = Query(...),
):
    """Return all knowledge-tree nodes for a product (title, slug, type, content)."""
    _enforce_loopback(request)
    nodes = []
    try:
        with SessionLocal() as db:
            rows = (
                db.query(KnowledgeNodeORM)
                .filter(KnowledgeNodeORM.product_id == product_id)
                .all()
            )
            for n in rows:
                nodes.append({
                    "id": n.id,
                    "title": n.title,
                    "slug": getattr(n, "slug", None),
                    "node_type": getattr(n, "node_type", None),
                    "content_md": getattr(n, "content_md", "") or "",
                })
    except Exception as e:
        logger.debug("rlm_tools: get_knowledge_nodes failed: %s", e)
        return JSONResponse({"nodes": []})
    return JSONResponse({"nodes": nodes})


@router.get("/codebases")
def get_codebases(
    request: Request,
    product_id: str = Query(...),
):
    """Return all codebases for a product (id, name, repo_url, generated_docs)."""
    _enforce_loopback(request)
    p = _product_or_404(product_id)
    if p is None:
        return JSONResponse({"codebases": []})
    cbs = []
    for c in p.codebases:
        cbs.append({
            "id": c.id,
            "name": c.name,
            "repo_url": getattr(c, "repo_url", "") or "",
            "generated_docs": getattr(c, "generated_docs", "") or "",
        })
    return JSONResponse({"codebases": cbs})
