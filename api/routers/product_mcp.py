"""Product ↔ MCP server bindings router (Wave C, contract 2).

Authenticated (any logged-in user) endpoints under ``/api/products``:

- ``GET    /api/products/{product_id}/mcp``             — list bindings
- ``POST   /api/products/{product_id}/mcp``             — bind a server
- ``PUT    /api/products/{product_id}/mcp/{binding_id}`` — update enabled/allowlist
- ``DELETE /api/products/{product_id}/mcp/{binding_id}`` — unbind

Each product can bind a server at most once — a duplicate POST is **409**
(documented deviation from "POST → 200 always"). ``allowed_tools`` is a list
of tool names (``null``/absent on PUT = ALL tools); on PUT an ABSENT field
keeps the current allowlist, an explicit ``null`` resets it to "all tools".

The binding view embeds the server's ``name``/``transport``/``status`` so the
UI can render rows without a second round-trip to the admin registry.
"""

from __future__ import annotations

import logging
import secrets as pysecrets
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.auth.deps import get_current_user
from api.db import get_db
from api.models import McpServerORM, ProductMcpServerORM, ProductORM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["product-mcp"])

_MAX_TOOLS = 128
_MAX_TOOL_NAME = 128


class McpBindingCreate(BaseModel):
    mcp_server_id: str = Field(min_length=1, max_length=64)
    enabled: bool = True
    allowed_tools: Optional[List[str]] = Field(default=None, max_length=_MAX_TOOLS)


class McpBindingUpdate(BaseModel):
    enabled: Optional[bool] = None
    allowed_tools: Optional[List[str]] = Field(default=None, max_length=_MAX_TOOLS)


class McpBindingOut(BaseModel):
    id: str
    mcp_server_id: str
    name: str
    transport: str
    enabled: bool
    allowed_tools: Optional[List[str]] = None
    status: str


def _validate_allowed_tools(tools: Optional[List[str]]) -> Optional[List[str]]:
    if tools is None:
        return None
    if len(tools) > _MAX_TOOLS:
        raise HTTPException(
            status_code=400, detail=f"allowed_tools must hold at most {_MAX_TOOLS} entries"
        )
    out: List[str] = []
    for t in tools:
        if not isinstance(t, str) or not t or len(t) > _MAX_TOOL_NAME:
            raise HTTPException(
                status_code=400,
                detail=f"allowed_tools entries must be strings of 1..{_MAX_TOOL_NAME} chars",
            )
        out.append(t)
    return out


def _load_product(db: Session, product_id: str) -> ProductORM:
    product = db.query(ProductORM).filter(ProductORM.id == product_id).first()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


def _load_binding(db: Session, product_id: str, binding_id: str) -> ProductMcpServerORM:
    binding = (
        db.query(ProductMcpServerORM)
        .filter(
            ProductMcpServerORM.id == binding_id,
            ProductMcpServerORM.product_id == product_id,
        )
        .first()
    )
    if binding is None:
        raise HTTPException(status_code=404, detail="Binding not found")
    return binding


def _binding_out(db: Session, b: ProductMcpServerORM) -> McpBindingOut:
    server = db.query(McpServerORM).filter(McpServerORM.id == b.mcp_server_id).first()
    return McpBindingOut(
        id=b.id,
        mcp_server_id=b.mcp_server_id,
        name=server.name if server else "(deleted)",
        transport=server.transport if server else "",
        enabled=bool(b.enabled),
        allowed_tools=list(b.allowed_tools) if isinstance(b.allowed_tools, list) else None,
        status=(server.status if server else "error") or "unknown",
    )


@router.get("/{product_id}/mcp", response_model=List[McpBindingOut])
async def list_product_mcp_bindings(
    product_id: str,
    db: Session = Depends(get_db),
    _user: Any = Depends(get_current_user),
):
    _load_product(db, product_id)
    bindings = (
        db.query(ProductMcpServerORM)
        .filter(ProductMcpServerORM.product_id == product_id)
        .order_by(ProductMcpServerORM.created_at)
        .all()
    )
    return [_binding_out(db, b) for b in bindings]


@router.post("/{product_id}/mcp", response_model=McpBindingOut)
async def create_product_mcp_binding(
    product_id: str,
    body: McpBindingCreate,
    db: Session = Depends(get_db),
    _user: Any = Depends(get_current_user),
):
    _load_product(db, product_id)
    server = (
        db.query(McpServerORM).filter(McpServerORM.id == body.mcp_server_id).first()
    )
    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    dup = (
        db.query(ProductMcpServerORM)
        .filter(
            ProductMcpServerORM.product_id == product_id,
            ProductMcpServerORM.mcp_server_id == body.mcp_server_id,
        )
        .first()
    )
    if dup is not None:
        raise HTTPException(
            status_code=409, detail="This MCP server is already bound to the product"
        )
    binding = ProductMcpServerORM(
        id=f"pmb_{pysecrets.token_hex(16)}",
        product_id=product_id,
        mcp_server_id=body.mcp_server_id,
        enabled=body.enabled,
        allowed_tools=_validate_allowed_tools(body.allowed_tools),
    )
    db.add(binding)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("product_mcp: DB commit failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Database error")
    db.refresh(binding)
    return _binding_out(db, binding)


@router.put("/{product_id}/mcp/{binding_id}", response_model=McpBindingOut)
async def update_product_mcp_binding(
    product_id: str,
    binding_id: str,
    body: McpBindingUpdate,
    db: Session = Depends(get_db),
    _user: Any = Depends(get_current_user),
):
    _load_product(db, product_id)
    binding = _load_binding(db, product_id, binding_id)
    fields = body.model_fields_set
    if "enabled" in fields and body.enabled is not None:
        binding.enabled = body.enabled
    if "allowed_tools" in fields:
        binding.allowed_tools = _validate_allowed_tools(body.allowed_tools)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("product_mcp: DB commit failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Database error")
    db.refresh(binding)
    return _binding_out(db, binding)


@router.delete("/{product_id}/mcp/{binding_id}")
async def delete_product_mcp_binding(
    product_id: str,
    binding_id: str,
    db: Session = Depends(get_db),
    _user: Any = Depends(get_current_user),
):
    _load_product(db, product_id)
    binding = _load_binding(db, product_id, binding_id)
    db.delete(binding)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("product_mcp: DB commit failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Database error")
    return {"message": "Binding deleted"}
