"""Public API router (contract J, plan section H).

API-token-authenticated endpoints for external integrations:

- ``GET  /api/public/products/{product_id}/knowledge`` — export VERIFIED
  knowledge as markdown (default) or json.
- ``POST /api/public/products/{product_id}/ask`` — reuse the expert agent to
  answer a query over a product (SSE stream).
- ``POST /api/public/products/{product_id}/push`` — push verified docs to a
  configured integration (Confluence / git).

All endpoints require a valid Bearer API token (``require_api_token``), which
also updates ``last_used_at``. Only verified content (``KnowledgeNode`` /
``Codebase``/``Spec``/``Links``/``Database`` with ``verified=True``) is exported
or pushed. Database exports carry only the MASKED DSN — the raw DSN never
leaves the server.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth.deps import require_api_token
from api.db import get_db
from api.utils.rate_limit import enforce_user_rate_limit
from api.models import (
    ApiTokenORM, CodebaseORM, DatabaseORM, KnowledgeNodeORM, LinksORM,
    ProductORM, SpecORM,
)
from api.config.settings import get_confluence_creds, get_git_creds

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/public", tags=["public"])

# Git hosts checked when picking a default push target from settings.
_GIT_HOSTS = ("github", "gitlab")


class AskRequest(BaseModel):
    query: str
    messages: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None


class PushRequest(BaseModel):
    target: Optional[str] = None
    space: Optional[str] = None
    host: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


# --- Verified-content queries ----------------------------------------------
def _verified_codebases(product_id: str, db: Session) -> List[CodebaseORM]:
    return (
        db.query(CodebaseORM)
        .filter(CodebaseORM.product_id == product_id, CodebaseORM.verified.is_(True))
        .all()
    )


def _verified_specs(product_id: str, db: Session) -> List[SpecORM]:
    return (
        db.query(SpecORM)
        .filter(SpecORM.product_id == product_id, SpecORM.verified.is_(True))
        .all()
    )


def _verified_links(product_id: str, db: Session) -> List[LinksORM]:
    return (
        db.query(LinksORM)
        .filter(LinksORM.product_id == product_id, LinksORM.verified.is_(True))
        .all()
    )


def _verified_databases(product_id: str, db: Session) -> List[DatabaseORM]:
    return (
        db.query(DatabaseORM)
        .filter(DatabaseORM.product_id == product_id, DatabaseORM.verified.is_(True))
        .all()
    )


def _verified_nodes(product_id: str, db: Session) -> List[KnowledgeNodeORM]:
    return (
        db.query(KnowledgeNodeORM)
        .filter(
            KnowledgeNodeORM.product_id == product_id,
            KnowledgeNodeORM.verified.is_(True),
        )
        .all()
    )


def _verified_meta(e: Any) -> str:
    """Build the ``> type | kind | verified_by`` meta line for markdown export."""
    meta: List[str] = []
    cls = type(e).__name__.replace("ORM", "").lower()
    meta.append(f"{cls}")
    kind = getattr(e, "kind", None)
    if kind:
        meta.append(f"kind: {kind}")
    if e.verified_by:
        meta.append(f"verified_by: {e.verified_by}")
    return "> " + " | ".join(meta)


def _knowledge_as_json(
    product: ProductORM,
    codebases: List[CodebaseORM],
    specs: List[SpecORM],
    links: List[LinksORM],
    nodes: List[KnowledgeNodeORM],
    *,
    databases: Optional[List[DatabaseORM]] = None,
) -> Dict[str, Any]:
    def _vmeta(e: Any) -> Dict[str, Any]:
        return {
            "verified_by": e.verified_by,
            "verified_at": e.verified_at.isoformat() if e.verified_at else None,
            "source": e.source,
        }

    return {
        "product": {"id": product.id, "name": product.name, "summary": product.summary},
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "verified_only": True,
        "codebases": [
            {"id": c.id, "name": c.name, "generated_docs": c.generated_docs, **_vmeta(c)}
            for c in codebases
        ],
        "specs": [
            {"id": s.id, "name": s.name, "kind": s.kind, "content": s.content, **_vmeta(s)}
            for s in specs
        ],
        "links": [
            {"id": l.id, "name": l.name, "content": l.content, **_vmeta(l)}
            for l in links
        ],
        "databases": [
            {
                "id": d.id,
                "name": d.name,
                "dsn_masked": d.dsn_masked,
                "generated_docs": d.generated_docs,
                **_vmeta(d),
            }
            for d in (databases or [])
        ],
        "nodes": [
            {
                "id": n.id, "parent_id": n.parent_id, "title": n.title, "slug": n.slug,
                "node_type": n.node_type, "content_md": n.content_md, **_vmeta(n),
            }
            for n in nodes
        ],
    }


def _knowledge_as_markdown(
    product: ProductORM,
    codebases: List[CodebaseORM],
    specs: List[SpecORM],
    links: List[LinksORM],
    nodes: List[KnowledgeNodeORM],
    *,
    databases: Optional[List[DatabaseORM]] = None,
) -> str:
    lines: List[str] = [f"# {product.name} — Verified Knowledge", ""]
    if product.summary:
        lines += [product.summary, ""]
    lines.append(f"_Exported {datetime.utcnow().isoformat()}Z — verified content only._")
    lines.append("")

    def _section(title: str, items: List[Any], content_attr: str) -> None:
        if not items:
            return
        lines.extend([f"## {title}", ""])
        for e in items:
            lines.extend([f"### {e.name}", "", _verified_meta(e), ""])
            content = getattr(e, content_attr, None)
            if content:
                lines.extend([content, ""])

    _section("Codebases", codebases, "generated_docs")
    _section("Databases", databases or [], "generated_docs")
    _section("Specifications", specs, "content")
    _section("Links", links, "content")

    if nodes:
        lines += ["## Knowledge Pages", ""]
        for n in sorted(nodes, key=lambda x: (x.parent_id or "", x.title)):
            lines += [f"### {n.title}", ""]
            if n.content_md:
                lines += [n.content_md, ""]

    return "\n".join(lines)


def _load_verified(product_id: str, db: Session):
    """(codebases, specs, links, nodes) — pre-Wave-E shape, kept for callers.

    ``api.mcp.inbound`` and older callers unpack this 4-tuple; the export /
    push endpoints additionally load databases via ``_verified_databases``.
    """
    return (
        _verified_codebases(product_id, db),
        _verified_specs(product_id, db),
        _verified_links(product_id, db),
        _verified_nodes(product_id, db),
    )


# --- GET /api/public/products ------------------------------------------------
@router.get("/products")
def list_public_products(
    tok: ApiTokenORM = Depends(require_api_token),
    db: Session = Depends(get_db),
):
    """List all products available for public knowledge export/chat."""
    products = db.query(ProductORM).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "summary": p.summary,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        }
        for p in products
    ]


# --- GET /api/public/products/{product_id}/knowledge ------------------------
@router.get("/products/{product_id}/knowledge")
def export_knowledge(
    product_id: str,
    format: str = Query("markdown", description="markdown | json"),
    tok: ApiTokenORM = Depends(require_api_token),
    db: Session = Depends(get_db),
):
    """Export VERIFIED knowledge for a product as markdown (default) or json."""
    fmt = (format or "markdown").lower()
    if fmt not in ("markdown", "json"):
        raise HTTPException(status_code=400, detail="format must be 'markdown' or 'json'")
    product = db.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    codebases, specs, links, nodes = _load_verified(product_id, db)
    databases = _verified_databases(product_id, db)
    if fmt == "json":
        return JSONResponse(content=_knowledge_as_json(
            product, codebases, specs, links, nodes, databases=databases
        ))
    md = _knowledge_as_markdown(
        product, codebases, specs, links, nodes, databases=databases
    )
    return Response(content=md, media_type="text/markdown; charset=utf-8")


# --- POST /api/public/products/{product_id}/ask -----------------------------
@router.post("/products/{product_id}/ask")
async def ask(
    product_id: str,
    body: AskRequest,
    tok: ApiTokenORM = Depends(require_api_token),
    db: Session = Depends(get_db),
):
    """Reuse the expert agent to answer a query over a product (SSE stream)."""
    # P1-17: per-user (token owner) bucket; tokens without a linked user are
    # keyed by the token id itself.
    enforce_user_rate_limit(
        tok.user_id or tok.id,
        setting_key="rate.public.per_user_min",
        env_name="RATE_PUBLIC_PER_USER_MINUTE",
        default_per_minute=30,
    )
    product = db.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    try:
        from api.expert.chat import run_expert_chat  # lazy: built in parallel
    except Exception as e:
        logger.warning("expert chat unavailable on the public ask path: %s", e)
        raise HTTPException(status_code=501, detail="Expert agent is not available")

    async def event_stream():
        try:
            agen = run_expert_chat(
                product_id=product_id,
                query=body.query,
                messages=body.messages or [],
                model=body.model,
                stream=True,
            )
            if hasattr(agen, "__aiter__"):
                async for chunk in agen:
                    payload = chunk if isinstance(chunk, str) else str(chunk)
                    yield f"data: {json.dumps({'delta': payload})}\n\n"
            else:
                result = await agen
                yield f"data: {json.dumps({'delta': str(result)})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:  # pragma: no cover - streamed error path
            # Generic message: the exception may embed internal endpoints.
            logger.warning("expert chat stream failed: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': 'internal error while streaming the answer'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- POST /api/public/products/{product_id}/push ----------------------------
def _default_push_target() -> str:
    """Pick a default push target from settings (Confluence first, then git)."""
    cconf = get_confluence_creds()
    if cconf.get("base_url") and cconf.get("token"):
        return "confluence"
    for host in _GIT_HOSTS:
        g = get_git_creds(host)
        if g.get("url") and g.get("token"):
            return host
    return "confluence"


@router.post("/products/{product_id}/push")
async def push(
    product_id: str,
    body: PushRequest,
    tok: ApiTokenORM = Depends(require_api_token),
    db: Session = Depends(get_db),
):
    """Push verified docs to a configured integration (Confluence or git)."""
    product = db.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    codebases, specs, links, nodes = _load_verified(product_id, db)
    databases = _verified_databases(product_id, db)
    target = (body.target or _default_push_target()).lower()

    try:
        from api.integrations import registry as _reg  # lazy: built in parallel
    except Exception as e:
        logger.warning("integrations unavailable on the public push path: %s", e)
        raise HTTPException(status_code=501, detail="Integrations are not available")
    getter = getattr(_reg, "get_connector", None)
    connector = getter(target) if callable(getter) else None
    if connector is None:
        raise HTTPException(status_code=501, detail=f"No connector registered for target '{target}'.")
    push_fn = getattr(connector, "push", None)
    if not callable(push_fn):
        push_fn = getattr(connector, "export", None)
    if not callable(push_fn):
        raise HTTPException(status_code=501, detail=f"Connector '{target}' does not support push/export.")

    md = _knowledge_as_markdown(
        product, codebases, specs, links, nodes, databases=databases
    )
    payload = {
        "product_id": product_id,
        "product_name": product.name,
        "markdown": md,
        "space": body.space,
        "host": body.host,
        "options": body.options or {},
        "user_id": tok.user_id,
    }
    try:
        # P1-13: sync push functions (Confluence/git connectors do blocking
        # network I/O) must not run on the event loop; native async ones are
        # awaited directly.
        if inspect.iscoroutinefunction(push_fn):
            result = await push_fn(payload)
        else:
            result = await asyncio.to_thread(push_fn, payload)
            if inspect.isawaitable(result):
                result = await result
    except Exception as e:
        # Push exceptions (Confluence/git HTTP stacks) can embed authenticated
        # URLs — log the details, keep the client message generic.
        logger.warning("public push to %r failed: %s", target, e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Push to '{target}' failed")
    if isinstance(result, dict):
        return result
    return {"success": True, "target": target, "message": "Pushed verified knowledge."}
