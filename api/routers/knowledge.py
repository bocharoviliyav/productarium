"""Knowledge tree router (contract J, section E).

Confluence-like tree of knowledge pages scoped to a Product, plus the Verified
toggle (item 5) and the AI product summary (item 4). All endpoints are mounted
under ``/api/products/{product_id}/...`` and reuse the foundation auth
dependencies (``get_current_user`` / ``require_admin``) and ``get_db``.

Endpoints:
- ``GET    /api/products/{product_id}/knowledge/tree``
- ``POST   /api/products/{product_id}/knowledge/nodes``
- ``GET    /api/products/{product_id}/knowledge/nodes/{node_id}``
- ``PUT    /api/products/{product_id}/knowledge/nodes/{node_id}``
- ``DELETE /api/products/{product_id}/knowledge/nodes/{node_id}``
- ``POST   /api/products/{product_id}/knowledge/nodes/{node_id}/upload``
- ``POST   /api/products/{product_id}/knowledge/nodes/{node_id}/verify``
- ``POST   /api/products/{product_id}/summary``

Design notes:
- Tree is built from a flat ``KnowledgeNodeORM`` list using ``parent_id``; the
  response is a list of root node dicts each carrying a ``children`` list (the
  ``KnowledgeNode`` Pydantic model has no ``children`` field by design).
- Subtree deletion is handled by the DB-level ``ON DELETE CASCADE`` on
  ``knowledge_nodes.parent_id``; we simply delete the node and commit.
- The markitdown upload imports ``api.formats.markitdown`` LAZILY inside the
  handler. On absence we degrade gracefully (text passthrough or 501).
- The summary uses ``api.docgen.summary.generate_product_summary`` (self-
  contained, decoupled from ``api.expert``) and stores the result onto
  ``ProductORM.summary``.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import importlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session, selectinload

from api.auth.deps import get_current_user
from api.db import get_db
from api.docgen.summary import generate_product_summary
from api.models import ArtifactORM, KnowledgeNodeORM, ProductORM
from api.schemas import (
    KnowledgeNode,
    KnowledgeNodeCreate,
    KnowledgeNodeUpdate,
)
from api.models import UserORM

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _new_node_id() -> str:
    import uuid
    return f"node_{uuid.uuid4().hex[:24]}"


_SLUG_NONALNUM = re.compile(r"[^a-z0-9-_]+")


def _slugify(title: str) -> str:
    """Derive a URL-safe slug from a title (ascii lowercase, hyphen-separated)."""
    s = (title or "").strip().lower()
    s = s.replace(" ", "-").replace("_", "-")
    s = _SLUG_NONALNUM.sub("", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "node"


def _orm_to_node(n: KnowledgeNodeORM) -> KnowledgeNode:
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


def _node_dict(n: KnowledgeNodeORM) -> Dict[str, Any]:
    """A serializable node dict (no children) for the tree response."""
    return {
        "id": n.id,
        "product_id": n.product_id,
        "parent_id": n.parent_id,
        "title": n.title,
        "slug": n.slug,
        "content_md": n.content_md,
        "node_type": n.node_type,
        "artifact_id": n.artifact_id,
        "source": n.source,
        "verified": n.verified,
        "verified_by": n.verified_by,
        "verified_at": n.verified_at,
        "created_by": n.created_by,
        "created_at": n.created_at,
        "updated_at": n.updated_at,
        "children": [],
    }


def build_tree(nodes: List[KnowledgeNodeORM]) -> List[Dict[str, Any]]:
    """Build a nested tree from a flat node list using ``parent_id``.

    Returns the list of root nodes (parent_id is None), each with a populated
    ``children`` list. Orphaned nodes (parent missing from the set) are treated
    as roots so a stale parent_id never hides a node.
    """
    by_id: Dict[str, Dict[str, Any]] = {n.id: _node_dict(n) for n in nodes}
    roots: List[Dict[str, Any]] = []
    for n in nodes:
        node_dict = by_id[n.id]
        pid = n.parent_id
        if pid and pid in by_id:
            by_id[pid]["children"].append(node_dict)
        else:
            roots.append(node_dict)
    return roots


def _load_product(db: Session, product_id: str) -> ProductORM:
    p = db.get(ProductORM, product_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return p


def _load_node(db: Session, product_id: str, node_id: str) -> KnowledgeNodeORM:
    n = db.get(KnowledgeNodeORM, node_id)
    if n is None or n.product_id != product_id:
        raise HTTPException(status_code=404, detail="Knowledge node not found")
    return n


def _is_owner(product: ProductORM, node: Optional[KnowledgeNodeORM], user: UserORM) -> bool:
    if user.role == "admin":
        return True
    if product.owner_id and product.owner_id == user.id:
        return True
    if node is not None and node.created_by and node.created_by == user.id:
        return True
    return False


# --------------------------------------------------------------------------- #
# Tree + CRUD
# --------------------------------------------------------------------------- #
@router.get("/api/products/{product_id}/knowledge/tree")
def get_knowledge_tree(
    product_id: str,
    db: Session = Depends(get_db),
    _user: UserORM = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Return the nested knowledge tree for a product (list of root nodes)."""
    _load_product(db, product_id)
    nodes = (
        db.query(KnowledgeNodeORM)
        .filter(KnowledgeNodeORM.product_id == product_id)
        .all()
    )
    return build_tree(nodes)


@router.post(
    "/api/products/{product_id}/knowledge/nodes",
    response_model=KnowledgeNode,
    status_code=status.HTTP_201_CREATED,
)
def create_node(
    product_id: str,
    body: KnowledgeNodeCreate,
    db: Session = Depends(get_db),
    user: UserORM = Depends(get_current_user),
) -> KnowledgeNode:
    """Create a knowledge node. Validates parent belongs to the same product."""
    _load_product(db, product_id)
    if body.parent_id:
        parent = db.get(KnowledgeNodeORM, body.parent_id)
        if parent is None or parent.product_id != product_id:
            raise HTTPException(
                status_code=400,
                detail="parent_id does not belong to this product",
            )
    node = KnowledgeNodeORM(
        id=_new_node_id(),
        product_id=product_id,
        parent_id=body.parent_id,
        title=body.title,
        slug=body.slug or _slugify(body.title),
        content_md=body.content_md,
        node_type=body.node_type or "page",
        artifact_id=body.artifact_id,
        source=body.source or "manual",
        created_by=user.id,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return _orm_to_node(node)


@router.get(
    "/api/products/{product_id}/knowledge/nodes/{node_id}",
    response_model=KnowledgeNode,
)
def get_node(
    product_id: str,
    node_id: str,
    db: Session = Depends(get_db),
    _user: UserORM = Depends(get_current_user),
) -> KnowledgeNode:
    """Return a single knowledge node (full content_md)."""
    node = _load_node(db, product_id, node_id)
    return _orm_to_node(node)


def _validate_parent_move(
    db: Session, product_id: str, node_id: str, parent_id: Optional[str]
) -> Optional[str]:
    """Normalize + validate a parent_id move for drag-and-drop reordering.

    Returns the resolved parent_id to store (None = root) or raises HTTPException
    on invalid moves. Rules:
    - ``""`` / absent -> treated as None (root).
    - parent must exist and belong to the same product.
    - parent must not be the node itself or one of its descendants (cycle).
    """
    # Empty string is the client's "move to root" signal; normalize to None.
    resolved = parent_id or None
    if resolved is None:
        return None
    parent = db.get(KnowledgeNodeORM, resolved)
    if parent is None or parent.product_id != product_id:
        raise HTTPException(
            status_code=400,
            detail="parent_id does not belong to this product",
        )
    if resolved == node_id:
        raise HTTPException(
            status_code=400,
            detail="A node cannot be its own parent",
        )
    # Walk up from the proposed parent; if we reach this node, the move would
    # create a cycle (node would become an ancestor of itself).
    cursor: Optional[str] = resolved
    seen: set = set()
    while cursor is not None and cursor not in seen:
        if cursor == node_id:
            raise HTTPException(
                status_code=400,
                detail="Cannot move a node under its own descendant (cycle)",
            )
        seen.add(cursor)
        anc = db.get(KnowledgeNodeORM, cursor)
        cursor = anc.parent_id if anc is not None else None
    return resolved


@router.put(
    "/api/products/{product_id}/knowledge/nodes/{node_id}",
    response_model=KnowledgeNode,
)
def update_node(
    product_id: str,
    node_id: str,
    body: KnowledgeNodeUpdate,
    db: Session = Depends(get_db),
    _user: UserORM = Depends(get_current_user),
) -> KnowledgeNode:
    """Update a node's title/slug/content_md/node_type/parent_id.

    WYSIWYG saves ``content_md``; drag-and-drop saves ``parent_id``. When
    ``content_md`` changes, the new text is re-indexed into the product's
    cognee dataset in the background (fire-and-forget, non-fatal) so expert
    Ask/summary stay in sync with manual page edits (and future MCP/Confluence/
    webhook-pulled content).
    """
    node = _load_node(db, product_id, node_id)
    data = body.model_dump(exclude_unset=True)
    content_changed = "content_md" in data and data["content_md"] is not None
    for field in ("title", "slug", "content_md", "node_type"):
        if field in data and data[field] is not None:
            setattr(node, field, data[field])
    if body.slug is not None and not body.slug.strip():
        # Re-derive slug from the (possibly updated) title when explicitly cleared.
        node.slug = _slugify(node.title)
    if "parent_id" in data:
        node.parent_id = _validate_parent_move(
            db, product_id, node_id, body.parent_id
        )
    db.commit()
    db.refresh(node)

    # Re-index edited page content into the per-product cognee dataset so the
    # expert agent / Ask recall user edits. Fire-and-forget; never fatal.
    if content_changed and node.content_md and node.content_md.strip():
        try:
            from api.docgen import _index_in_background
            _index_in_background(node.content_md, f"prod_{product_id}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Cognee re-index failed for node %s: %s", node_id, e)

    return _orm_to_node(node)


@router.delete("/api/products/{product_id}/knowledge/nodes/{node_id}")
def delete_node(
    product_id: str,
    node_id: str,
    db: Session = Depends(get_db),
    _user: UserORM = Depends(get_current_user),
) -> Dict[str, str]:
    """Delete a node; its subtree is removed by the DB ON DELETE CASCADE."""
    node = _load_node(db, product_id, node_id)
    db.delete(node)
    db.commit()
    return {"message": "Knowledge node deleted"}


# --------------------------------------------------------------------------- #
# markitdown upload
# --------------------------------------------------------------------------- #
def _convert_via_markitdown(data: bytes, filename: str) -> tuple:
    """Try api.formats.markitdown.convert_to_markdown with common conventions.

    Returns ``(ok, markdown_or_error)``. We import it lazily and tolerate
    either a path-based or bytes-based ``convert_to_markdown`` signature.
    """
    try:
        md = importlib.import_module("api.formats.markitdown")
    except Exception as e:  # module absent -> degrade
        return (False, f"markitdown unavailable: {e}")
    convert = getattr(md, "convert_to_markdown", None)
    if convert is None:
        return (False, "formats.markitdown.convert_to_markdown not found")

    suffix = os.path.splitext(filename or "")[1]
    tmp_path: Optional[str] = None
    try:
        # Persist bytes to a temp file so path-based wrappers work too.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp_path = tf.name
        for args, kwargs in (
            ((tmp_path,), {"filename": filename}),
            ((tmp_path,), {}),
            ((data,), {"filename": filename}),
            ((data,), {}),
        ):
            try:
                result = convert(*args, **kwargs)
            except TypeError:
                continue
            if result:
                return (True, str(result))
        return (False, "markitdown convert_to_markdown returned no output")
    except Exception as e:
        return (False, f"markitdown conversion failed: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@router.post(
    "/api/products/{product_id}/knowledge/nodes/{node_id}/upload",
    response_model=KnowledgeNode,
)
async def upload_node_content(
    product_id: str,
    node_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _user: UserORM = Depends(get_current_user),
) -> KnowledgeNode:
    """Convert an uploaded file to Markdown via markitdown and store as content_md."""
    node = _load_node(db, product_id, node_id)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    filename = file.filename or ""

    ok, output = _convert_via_markitdown(data, filename)
    if ok:
        node.content_md = output
    else:
        # Graceful degradation: if markitdown is unavailable, store UTF-8 text
        # when possible; otherwise 501 so the client knows conversion failed.
        logger.warning("markitdown conversion degraded: %s", output)
        try:
            node.content_md = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=(
                    "File could not be converted: markitdown is unavailable and "
                    "the upload is not UTF-8 text."
                ),
            )
    db.commit()
    db.refresh(node)
    return _orm_to_node(node)


# --------------------------------------------------------------------------- #
# Verified toggle (item 5) — owner or admin
# --------------------------------------------------------------------------- #
@router.post(
    "/api/products/{product_id}/knowledge/nodes/{node_id}/verify",
    response_model=KnowledgeNode,
)
def verify_node(
    product_id: str,
    node_id: str,
    db: Session = Depends(get_db),
    user: UserORM = Depends(get_current_user),
) -> KnowledgeNode:
    """Mark a node as verified (owner of the product/node or admin only)."""
    product = _load_product(db, product_id)
    node = _load_node(db, product_id, node_id)
    if not _is_owner(product, node, user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the product/node owner or an admin can verify",
        )
    node.verified = True
    node.verified_by = user.id
    node.verified_at = datetime.utcnow()
    db.commit()
    db.refresh(node)
    return _orm_to_node(node)


# --------------------------------------------------------------------------- #
# AI product summary (item 4)
# --------------------------------------------------------------------------- #
@router.post("/api/products/{product_id}/summary")
async def generate_summary(
    product_id: str,
    db: Session = Depends(get_db),
    _user: UserORM = Depends(get_current_user),
) -> Dict[str, Any]:
    """Generate an AI summary over the product's artifacts + knowledge nodes.

    Concatenates artifact ``generated_docs`` + node ``content_md``, asks the
    standard local LLM for a concise summary, stores it onto ``ProductORM.summary``
    and returns it.
    """
    product = (
        db.query(ProductORM)
        .options(selectinload(ProductORM.artifacts))
        .filter(ProductORM.id == product_id)
        .first()
    )
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    nodes = (
        db.query(KnowledgeNodeORM)
        .filter(KnowledgeNodeORM.product_id == product_id)
        .all()
    )
    summary = await generate_product_summary(product, product.artifacts, nodes)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Could not generate summary: no content available or the LLM is "
                "unreachable."
            ),
        )
    product.summary = summary
    db.commit()
    return {"product_id": product.id, "summary": summary}


__all__ = ["router", "build_tree"]
