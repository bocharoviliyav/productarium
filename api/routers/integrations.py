"""Integrations router (plan section G / contract J).

Exposes the scalable integrations framework:

- ``GET  /api/integrations``                              — list registered
  connectors + their configured status (any authenticated user).
- ``POST /api/integrations/{name}/test``                  — admin only; call a
  connector's ``test()`` to validate connectivity/credentials.
- ``GET  /api/integrations/{name}/spaces``                — any authenticated
  user; list the connector's pull sources (repos / spaces / MCP sources).
- ``POST /api/products/{product_id}/codebases/from-integration`` — git
  connectors only; pull a repo and create a ``CodebaseORM``.
- ``POST /api/products/{product_id}/knowledge/from-integration``  — any
  connector; pull a space/page and create one or more ``KnowledgeNodeORM``.

Pulled markdown is indexed into the product-scoped cognee dataset
``prod_{product_id}`` (background, non-fatal).

Router note: this router has NO prefix (a single prefix cannot host routes
under both ``/api/integrations`` and ``/api/products``); every endpoint
declares its full path. The tag stays ``["integrations"]``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.auth.deps import get_current_user, require_admin
from api.db import get_db
from api.integrations.registry import get_connector, list_connectors
from api.models import CodebaseORM, KnowledgeNodeORM, ProductORM
from api.schemas import Codebase, KnowledgeNode

logger = logging.getLogger(__name__)

router = APIRouter(tags=["integrations"])

# Connectors that produce a codebase (git clone).
_GIT_CONNECTORS = {"github", "gitlab"}


# --- Request / response models ----------------------------------------------
class FromIntegrationRequest(BaseModel):
    connector: str = Field(..., description="Connector name (e.g. github, confluence, mcp).")
    source_id: str = Field(..., description="Connector-specific source id (repo URL, page id, ...).")
    name: Optional[str] = Field(
        None, description="Name for the created codebase/node. Defaults to the pulled title."
    )
    opts: Optional[Dict[str, Any]] = Field(
        None, description="Per-pull options passed to the connector (e.g. {recursive: true})."
    )


class FromCodebaseRequest(FromIntegrationRequest):
    """Pull from a git connector and create a Codebase (repo_url/repo_type)."""


class FromNodeRequest(FromIntegrationRequest):
    """Pull from any connector and create KnowledgeNode(s)."""


# --- helpers -----------------------------------------------------------------
def _new_id(prefix: str) -> str:
    """Generate a frontend-compatible id: ``<prefix>_<base36 ts><6 random>``."""
    ts = format(int(time.time()), "x")
    rand = secrets.token_hex(3)  # 6 hex chars
    return f"{prefix}_{ts}{rand}"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", (title or "").lower()).strip("-")
    return slug or f"node-{secrets.token_hex(3)}"


def _codebase_pydantic(c: CodebaseORM) -> Codebase:
    return Codebase(
        id=c.id,
        name=c.name,
        repo_url=c.repo_url,
        repo_type=c.repo_type,
        token=c.token,
        generated_docs=c.generated_docs,
        pages=c.pages,
        verified=c.verified,
        verified_by=c.verified_by,
        verified_at=c.verified_at,
        source=c.source,
    )


def _knowledge_node_pydantic(n: KnowledgeNodeORM) -> KnowledgeNode:
    return KnowledgeNode(
        id=n.id,
        product_id=n.product_id,
        parent_id=n.parent_id,
        title=n.title,
        slug=n.slug,
        content_md=n.content_md,
        node_type=n.node_type,
        artifact_id=n.artifact_id,
        source=n.source,
        verified=n.verified,
        verified_by=n.verified_by,
        verified_at=n.verified_at,
        created_by=n.created_by,
        created_at=n.created_at,
        updated_at=n.updated_at,
    )


def _index_in_background(
    product_id: str,
    text: str,
    *,
    source_type: str = "integration",
    source_id: Optional[str] = None,
) -> None:
    """Fire-and-forget memory-backend indexing into the product-scoped dataset.

    Delegates to ``api.memory.index_document`` (active backend: pgvector by
    default, cognee alt) so the admin ``memory.backend`` switch governs this
    path too.
    """
    if not text:
        return
    dataset = f"prod_{product_id}"

    async def _run() -> None:
        try:
            from api.memory import index_document  # lazy: backend optional
            await index_document(
                text, product_id,
                source_type=source_type, source_id=source_id,
            )
        except Exception as e:  # pragma: no cover - non-fatal
            logger.warning("Background memory index for %s failed: %s", dataset, e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        try:
            asyncio.run(_run())
        except Exception as e:  # pragma: no cover - non-fatal
            logger.warning("Inline memory index for %s failed: %s", dataset, e)


def _compose_content(pulled: Dict[str, Any]) -> str:
    """Combine the pulled markdown + converted attachments into one MD blob."""
    parts = [pulled.get("markdown") or ""]
    for att in pulled.get("attachments") or []:
        if isinstance(att, dict):
            name = att.get("filename", "attachment")
            md = att.get("markdown", "")
            parts.append(f"\n\n---\n\n## Attachment: {name}\n\n{md}")
    return "\n".join(p for p in parts if p is not None)


# --- endpoints ---------------------------------------------------------------
@router.get("/api/integrations")
def list_registered_connectors(
    _user=Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List all registered integration connectors with their configured status."""
    return list_connectors()


@router.post("/api/integrations/{name}/test")
def test_connector(
    name: str,
    _admin=Depends(require_admin),
) -> Dict[str, Any]:
    """Admin-only: run a connector's connectivity/credentials test."""
    connector = get_connector(name)
    if connector is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown connector: {name}")
    try:
        return connector.test()
    except Exception as e:
        logger.warning("Connector %s test() raised: %s", name, e)
        return {"success": False, "message": f"Test failed: {e}"}


@router.get("/api/integrations/{name}/spaces")
def list_connector_spaces(
    name: str,
    _user=Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List the pull sources (repos / spaces / MCP sources) for a connector."""
    connector = get_connector(name)
    if connector is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown connector: {name}")
    try:
        return connector.list_spaces()
    except Exception as e:
        logger.warning("Connector %s list_spaces() raised: %s", name, e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.post(
    "/api/products/{product_id}/codebases/from-integration",
    response_model=Codebase,
    status_code=status.HTTP_201_CREATED,
)
def codebase_from_integration(
    product_id: str,
    body: FromCodebaseRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Pull a repo from a git connector and create a Codebase.

    Only git connectors (github/gitlab) are supported here; they set
    ``repo_url``/``repo_type`` on the created CodebaseORM.
    """
    if body.connector not in _GIT_CONNECTORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Connector '{body.connector}' does not produce a codebase; "
            f"use the knowledge pull endpoint instead.",
        )
    if db.get(ProductORM, product_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    connector = get_connector(body.connector)
    if connector is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown connector: {body.connector}",
        )

    try:
        pulled = connector.pull(body.source_id, body.opts)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.warning("Connector %s pull(%r) failed: %s", body.connector, body.source_id, e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Pull failed: {e}")

    name = body.name or pulled.get("title") or body.source_id
    codebase = CodebaseORM(
        id=_new_id("cb"),
        product_id=product_id,
        name=name,
        repo_url=pulled.get("repo_url"),
        repo_type=pulled.get("repo_type"),
        source="api",
    )
    db.add(codebase)
    try:
        db.commit()
        db.refresh(codebase)
    except Exception as e:
        db.rollback()
        logger.error("Failed to create Codebase from integration: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Persist failed")

    # Index any pulled markdown (rare for git, but some connectors attach docs).
    content = _compose_content(pulled)
    _index_in_background(product_id, content, source_type="codebase", source_id=codebase.id)
    return _codebase_pydantic(codebase)


@router.post(
    "/api/products/{product_id}/knowledge/from-integration",
    response_model=KnowledgeNode,
    status_code=status.HTTP_201_CREATED,
)
async def knowledge_from_integration(
    product_id: str,
    body: FromNodeRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Pull a space/page from any connector and create KnowledgeNode(s).

    Multi-page Confluence trees become a root node + children (parent links
    preserved). The pulled markdown (+ converted attachments) is indexed into
    the product-scoped cognee dataset in the background (non-fatal).
    """
    if db.get(ProductORM, product_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    connector = get_connector(body.connector)
    if connector is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown connector: {body.connector}",
        )

    try:
        pulled = connector.pull(body.source_id, body.opts)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.warning("Connector %s pull(%r) failed: %s", body.connector, body.source_id, e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Pull failed: {e}")

    title = pulled.get("title") or body.source_id
    name = body.name or title
    content = _compose_content(pulled)
    pulled_pages = pulled.get("pages") or []

    if isinstance(pulled_pages, list) and len(pulled_pages) > 1:
        # Multi-page tree: create root node + child nodes preserving parent links.
        created_nodes: List[KnowledgeNodeORM] = []
        indexed_texts: List[str] = []
        page_id_to_node_id: Dict[str, str] = {}
        for i, p in enumerate(pulled_pages):
            p_title = p.get("title") or f"Page {i+1}"
            p_html = p.get("html") or ""
            p_id = str(p.get("id") or i)
            p_parent = p.get("parent_id")

            node_id = _new_id("node")
            page_id_to_node_id[p_id] = node_id

            if i == 0:
                parent_node_id = None  # root page
            elif p_parent:
                parent_node_id = page_id_to_node_id.get(str(p_parent))
            else:
                parent_node_id = created_nodes[0].id if created_nodes else None

            knode = KnowledgeNodeORM(
                id=node_id,
                product_id=product_id,
                parent_id=parent_node_id,
                title=p_title,
                slug=_slugify(p_title),
                content_md=p_html or p_title,
                node_type="page",
                source="api",
                created_by=getattr(user, "id", None),
            )
            db.add(knode)
            created_nodes.append(knode)
            if p_html and p_html.strip():
                indexed_texts.append(p_html)

        try:
            db.commit()
            for kn in created_nodes:
                db.refresh(kn)
        except Exception as e:
            db.rollback()
            logger.error("Failed to create KnowledgeNode tree from integration: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Persist failed")

        if indexed_texts:
            _index_in_background(
                product_id, "\n\n".join(indexed_texts),
                source_type="integration",
                source_id=created_nodes[0].id if created_nodes else None,
            )
        return _knowledge_node_pydantic(created_nodes[0])

    # Single page node.
    node = KnowledgeNodeORM(
        id=_new_id("node"),
        product_id=product_id,
        parent_id=None,
        title=name,
        slug=_slugify(name),
        content_md=content,
        node_type="page",
        source="api",
        created_by=getattr(user, "id", None),
    )
    db.add(node)
    try:
        db.commit()
        db.refresh(node)
    except Exception as e:
        db.rollback()
        logger.error("Failed to create KnowledgeNode from integration: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Persist failed")
    _index_in_background(product_id, content, source_type="integration", source_id=node.id)
    return _knowledge_node_pydantic(node)


__all__ = ["router"]
