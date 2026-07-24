"""Integrations router (plan section G / contract J).

Exposes the scalable integrations framework:

- ``GET  /api/integrations``                              — list registered
  connectors + their configured status (any authenticated user).
- ``POST /api/integrations/{name}/test``                  — admin only; call a
  connector's ``test()`` to validate connectivity/credentials.
- ``GET  /api/integrations/{name}/spaces``                — any authenticated
  user; list the connector's pull sources (repos / spaces / MCP sources).
- ``POST /api/products/{product_id}/artifacts/from-integration`` — any
  authenticated user; pull a space/repo/page from a connector and create an
  Artifact (type=codebase for git, documentation for Confluence/MCP) OR a
  KnowledgeNode, then index the pulled markdown into the product-scoped cognee
  dataset ``prod_{product_id}`` (background, non-fatal).

Router note: a single ``APIRouter(prefix="/api/integrations")`` cannot also
host the ``/api/products/{product_id}/artifacts/from-integration`` route (the
prefix would be prepended), so this router has NO prefix and every endpoint
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
from api.cognee_manager import add_and_index_document
from api.db import get_db
from api.integrations.registry import get_connector, list_connectors
from api.models import (
    ARTIFACT_TYPES,
    ArtifactORM,
    KnowledgeNodeORM,
    ProductORM,
)
from api.schemas import Artifact, KnowledgeNode

logger = logging.getLogger(__name__)

router = APIRouter(tags=["integrations"])

# Connectors that produce a codebase artifact (vs documentation).
_GIT_CONNECTORS = {"github", "gitlab"}


# --- Request / response models (local to this router; contract J shapes) -----
class FromIntegrationRequest(BaseModel):
    connector: str = Field(..., description="Connector name (e.g. github, confluence, mcp).")
    source_id: str = Field(..., description="Connector-specific source id (repo URL, page id, ...).")
    artifact_name: Optional[str] = Field(
        None, description="Name for the created artifact/node. Defaults to the pulled title."
    )
    target: str = Field("artifact", description="artifact | node — what to create.")
    artifact_type: Optional[str] = Field(
        None,
        description=(
            "Override the artifact type. Defaults to codebase for git connectors, "
            "documentation otherwise. Must be one of: " + ", ".join(ARTIFACT_TYPES),
        ),
    )
    opts: Optional[Dict[str, Any]] = Field(
        None, description="Per-pull options passed to the connector (e.g. {recursive: true})."
    )


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


def _artifact_pydantic(a: ArtifactORM) -> Artifact:
    return Artifact(
        id=a.id,
        name=a.name,
        type=a.type,
        kind=a.kind,
        repo_url=a.repo_url,
        repo_type=a.repo_type,
        token=a.token,
        content=a.content,
        allure_url=a.allure_url,
        generated_docs=a.generated_docs,
        pages=a.pages,
        verified=a.verified,
        verified_by=a.verified_by,
        verified_at=a.verified_at,
        source=a.source,
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


def _index_in_background(product_id: str, text: str) -> None:
    """Fire-and-forget cognee indexing into the product-scoped dataset.

    cognee's ``add_and_index_document`` is async and non-fatal; we schedule it
    on the running loop and log any failure. Never blocks the HTTP response.
    """
    if not text:
        return
    dataset = f"prod_{product_id}"

    async def _run() -> None:
        try:
            await add_and_index_document(text, dataset_name=dataset)
        except Exception as e:  # pragma: no cover - non-fatal
            logger.warning("Background cognee index for %s failed: %s", dataset, e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        # No running loop (e.g. called from sync context in tests) — run inline.
        try:
            asyncio.run(_run())
        except Exception as e:  # pragma: no cover - non-fatal
            logger.warning("Inline cognee index for %s failed: %s", dataset, e)


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


@router.post("/api/products/{product_id}/artifacts/from-integration")
async def from_integration(
    product_id: str,
    body: FromIntegrationRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Pull a space/repo/page from a connector and create an Artifact or KnowledgeNode.

    The pulled markdown (plus converted attachments) is indexed into the
    product-scoped cognee dataset ``prod_{product_id}`` in the background
    (non-fatal). Git connectors create a ``codebase`` artifact (with
    ``repo_url``/``repo_type``); Confluence/MCP create a ``documentation``
    artifact unless ``artifact_type`` overrides it. When ``target == "node"``,
    a root ``KnowledgeNode`` is created instead.
    """
    # Validate product exists.
    product = db.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    connector = get_connector(body.connector)
    if connector is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown connector: {body.connector}",
        )

    # Pull from the connector.
    try:
        pulled = connector.pull(body.source_id, body.opts)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.warning("Connector %s pull(%r) failed: %s", body.connector, body.source_id, e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Pull failed: {e}")

    title = pulled.get("title") or body.source_id
    name = body.artifact_name or title
    content = _compose_content(pulled)

    # Resolve artifact type.
    if body.artifact_type and body.artifact_type not in ARTIFACT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"artifact_type must be one of {ARTIFACT_TYPES}",
        )
    artifact_type = body.artifact_type or (
        "codebase" if body.connector in _GIT_CONNECTORS else "documentation"
    )

    target = (body.target or "artifact").lower()
    if target not in ("artifact", "node"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target must be 'artifact' or 'node'",
        )

    if target == "node":
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
        _index_in_background(product_id, content)
        return _knowledge_node_pydantic(node)

    # target == "artifact"
    # repo_url/repo_type are only meaningful for codebase artifacts; propagate
    # them from the pulled payload when the (possibly overridden) type is
    # codebase, regardless of which connector produced it.
    is_codebase = artifact_type == "codebase"
    artifact = ArtifactORM(
        id=_new_id("art"),
        product_id=product_id,
        name=name,
        type=artifact_type,
        kind=None,
        repo_url=pulled.get("repo_url") if is_codebase else None,
        repo_type=pulled.get("repo_type") if is_codebase else None,
        content=content if not is_codebase else None,
        source="api",
    )
    db.add(artifact)
    try:
        db.commit()
        db.refresh(artifact)
    except Exception as e:
        db.rollback()
        logger.error("Failed to create Artifact from integration: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Persist failed")

    _index_in_background(product_id, content)
    return _artifact_pydantic(artifact)
