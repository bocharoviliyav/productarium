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
``Artifact`` with ``verified=True``) is exported or pushed.

The expert agent (``api.expert.chat``) and integrations (``api.integrations``)
are imported lazily: they are built in parallel and may not be present yet, so
a missing dependency degrades to a clear ``501`` instead of crashing the
router import.
"""

from __future__ import annotations

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
from api.models import ApiTokenORM, ArtifactORM, KnowledgeNodeORM, ProductORM
from api.settings_store import get_confluence_creds, get_git_creds

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/public", tags=["public"])

# Git hosts checked when picking a default push target from settings.
_GIT_HOSTS = ("github", "gitlab")


class AskRequest(BaseModel):
    query: str
    messages: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None


class PushRequest(BaseModel):
    # confluence | github | gitlab; default resolved from settings.
    target: Optional[str] = None
    space: Optional[str] = None  # Confluence space override
    host: Optional[str] = None  # git host override (kept for symmetry)
    options: Optional[Dict[str, Any]] = None


# --- Verified-content queries ----------------------------------------------
def _verified_artifacts(product_id: str, db: Session) -> List[ArtifactORM]:
    return (
        db.query(ArtifactORM)
        .filter(
            ArtifactORM.product_id == product_id,
            ArtifactORM.verified.is_(True),
        )
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


def _knowledge_as_json(
    product: ProductORM,
    artifacts: List[ArtifactORM],
    nodes: List[KnowledgeNodeORM],
) -> Dict[str, Any]:
    return {
        "product": {
            "id": product.id,
            "name": product.name,
            "summary": product.summary,
        },
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "verified_only": True,
        "artifacts": [
            {
                "id": a.id,
                "name": a.name,
                "type": a.type,
                "kind": a.kind,
                "generated_docs": a.generated_docs,
                "verified_by": a.verified_by,
                "verified_at": a.verified_at.isoformat() if a.verified_at else None,
                "source": a.source,
            }
            for a in artifacts
        ],
        "nodes": [
            {
                "id": n.id,
                "parent_id": n.parent_id,
                "title": n.title,
                "slug": n.slug,
                "node_type": n.node_type,
                "content_md": n.content_md,
                "verified_by": n.verified_by,
                "verified_at": n.verified_at.isoformat() if n.verified_at else None,
                "source": n.source,
            }
            for n in nodes
        ],
    }


def _knowledge_as_markdown(
    product: ProductORM,
    artifacts: List[ArtifactORM],
    nodes: List[KnowledgeNodeORM],
) -> str:
    lines: List[str] = []
    lines.append(f"# {product.name} — Verified Knowledge")
    lines.append("")
    if product.summary:
        lines.append(product.summary)
        lines.append("")
    lines.append(
        f"_Exported {datetime.utcnow().isoformat()}Z — verified content only._"
    )
    lines.append("")

    if artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for a in artifacts:
            lines.append(f"### {a.name}")
            lines.append("")
            meta = [f"type: {a.type}"]
            if a.kind:
                meta.append(f"kind: {a.kind}")
            if a.verified_by:
                meta.append(f"verified_by: {a.verified_by}")
            lines.append("> " + " | ".join(meta))
            lines.append("")
            if a.generated_docs:
                lines.append(a.generated_docs)
                lines.append("")

    if nodes:
        lines.append("## Knowledge Pages")
        lines.append("")
        for n in sorted(nodes, key=lambda x: (x.parent_id or "", x.title)):
            lines.append(f"### {n.title}")
            lines.append("")
            if n.content_md:
                lines.append(n.content_md)
                lines.append("")

    return "\n".join(lines)


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
    """Export VERIFIED knowledge for a product as markdown (default) or json.

    Only artifacts and knowledge nodes with ``verified=True`` are included.
    """
    fmt = (format or "markdown").lower()
    if fmt not in ("markdown", "json"):
        raise HTTPException(
            status_code=400, detail="format must be 'markdown' or 'json'"
        )
    product = db.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    artifacts = _verified_artifacts(product_id, db)
    nodes = _verified_nodes(product_id, db)

    if fmt == "json":
        return JSONResponse(content=_knowledge_as_json(product, artifacts, nodes))
    md = _knowledge_as_markdown(product, artifacts, nodes)
    return Response(content=md, media_type="text/markdown; charset=utf-8")


# --- POST /api/public/products/{product_id}/ask -----------------------------
@router.post("/products/{product_id}/ask")
async def ask(
    product_id: str,
    body: AskRequest,
    tok: ApiTokenORM = Depends(require_api_token),
    db: Session = Depends(get_db),
):
    """Reuse the expert agent to answer a query over a product (SSE stream).

    The expert agent (``api.expert.chat.run_expert_chat``) is imported lazily.
    If it is not available (e.g. not yet merged) the endpoint returns 501 with
    a clear message instead of failing at import time.
    """
    product = db.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    try:
        from api.expert.chat import run_expert_chat  # lazy: built in parallel
    except Exception as e:
        raise HTTPException(
            status_code=501, detail=f"Expert agent not available: {e}"
        )

    async def event_stream():
        try:
            agen = run_expert_chat(
                product_id=product_id,
                query=body.query,
                messages=body.messages or [],
                model=body.model,
                stream=True,
            )
            # Support both an async generator (streaming) and a coroutine
            # returning a complete string.
            if hasattr(agen, "__aiter__"):
                async for chunk in agen:
                    payload = chunk if isinstance(chunk, str) else str(chunk)
                    yield f"data: {json.dumps({'delta': payload})}\n\n"
            else:
                result = await agen
                yield f"data: {json.dumps({'delta': str(result)})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:  # pragma: no cover - streamed error path
            logger.warning("expert chat stream failed: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
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
    """Push verified docs to a configured integration (Confluence or git).

    Reads the target from the request body or, failing that, from the admin
    settings store (Confluence if configured, else a configured git host). The
    integration connector's ``push`` (or ``export``) method is called with a
    payload containing the verified-knowledge markdown. Returns 501 if no
    connector is registered or the connector does not support push.
    """
    product = db.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    artifacts = _verified_artifacts(product_id, db)
    nodes = _verified_nodes(product_id, db)
    target = (body.target or _default_push_target()).lower()

    try:
        from api.integrations import registry as _reg  # lazy: built in parallel
    except Exception as e:
        raise HTTPException(
            status_code=501, detail=f"Integrations not available: {e}"
        )
    getter = getattr(_reg, "get_connector", None)
    connector = getter(target) if callable(getter) else None
    if connector is None:
        raise HTTPException(
            status_code=501, detail=f"No connector registered for target '{target}'."
        )
    push_fn = getattr(connector, "push", None)
    if not callable(push_fn):
        push_fn = getattr(connector, "export", None)
    if not callable(push_fn):
        raise HTTPException(
            status_code=501,
            detail=f"Connector '{target}' does not support push/export.",
        )

    md = _knowledge_as_markdown(product, artifacts, nodes)
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
        result = push_fn(payload)
        if hasattr(result, "__await__"):
            result = await result
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Push to '{target}' failed: {e}"
        )
    if isinstance(result, dict):
        return result
    return {
        "success": True,
        "target": target,
        "message": "Pushed verified knowledge.",
    }
