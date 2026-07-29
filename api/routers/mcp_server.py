"""Productarium Native MCP (Model Context Protocol) Server Router (MCP 2024-11-05 spec).

Exposes Productarium's product-centric AI-native Confluence knowledge base to
external AI agents (Claude Desktop, Cursor, Windsurf, custom agents) over HTTP
and SSE JSON-RPC 2.0.

Endpoints:
- ``GET  /api/mcp/sse``     — Open an SSE session stream for MCP client connections.
- ``POST /api/mcp/message`` — Handle incoming JSON-RPC 2.0 requests (or POST /api/mcp).

Tools exposed via MCP:
1. ``list_products``        — List all products in Productarium.
2. ``get_product_knowledge`` — Export verified product documentation as Markdown.
3. ``search_product_graph`` — Query cognee knowledge graph (prod_{product_id}) for facts & triplets.
4. ``ask_expert``           — Query Productarium's Expert Agent about a product.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import get_db
from api.models import ArtifactORM, KnowledgeNodeORM, ProductORM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])

_MCP_PROTOCOL_VERSION = "2024-11-05"
_MCP_SERVER_INFO = {"name": "productarium", "version": "1.0.0"}


class McpJsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[Dict[str, Any]] = None


# --- Tool definitions --------------------------------------------------------
_TOOLS = [
    {
        "name": "list_products",
        "description": "List all products in Productarium with their IDs, names, and summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_product_knowledge",
        "description": "Get full verified documentation and knowledge for a product as Markdown or JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product ID (prod_...)"},
                "format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown"},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "search_product_graph",
        "description": "Query cognee knowledge graph for a product (dataset prod_{product_id}) to retrieve facts, triplets, and relationships.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product ID (prod_...)"},
                "query": {"type": "string", "description": "The search query"},
                "top_k": {"type": "integer", "default": 20, "description": "Max triplets to retrieve"},
            },
            "required": ["product_id", "query"],
        },
    },
    {
        "name": "ask_expert",
        "description": "Ask Productarium's Expert Agent a question about a product, grounded in cognee knowledge graph & verified artifacts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product ID (prod_...)"},
                "query": {"type": "string", "description": "The question to ask"},
                "model": {"type": "string", "description": "Optional LLM model override"},
            },
            "required": ["product_id", "query"],
        },
    },
]


# --- Tool execution logic ----------------------------------------------------
def _mcp_list_products(db: Session) -> str:
    products = db.query(ProductORM).all()
    if not products:
        return "No products found in Productarium."
    out = ["# Productarium Products\n"]
    for p in products:
        summary = (p.summary or "No summary available.").strip()
        out.append(f"## {p.name} (`{p.id}`)\n{summary}\n")
    return "\n".join(out)


def _mcp_get_product_knowledge(product_id: str, fmt: str, db: Session) -> str:
    product = db.get(ProductORM, product_id)
    if product is None:
        return f"Error: Product {product_id!r} not found."

    from api.routers.public import _knowledge_as_json, _knowledge_as_markdown, _verified_artifacts, _verified_nodes

    artifacts = _verified_artifacts(product_id, db)
    nodes = _verified_nodes(product_id, db)

    # If no explicitly verified artifacts/nodes exist yet, export all product artifacts/nodes as draft knowledge
    if not artifacts:
        artifacts = db.query(ArtifactORM).filter(ArtifactORM.product_id == product_id).all()
    if not nodes:
        nodes = db.query(KnowledgeNodeORM).filter(KnowledgeNodeORM.product_id == product_id).all()

    if fmt == "json":
        return json.dumps(_knowledge_as_json(product, artifacts, nodes), ensure_ascii=False, indent=2)
    return _knowledge_as_markdown(product, artifacts, nodes)


async def _mcp_search_product_graph(product_id: str, query: str, top_k: int = 20) -> str:
    dataset_name = f"prod_{product_id}"
    try:
        from api.cognee_manager import query_cognee

        results = await query_cognee(query, dataset_name=dataset_name, top_k=top_k)
        if results:
            return f"# Knowledge Graph Results for {dataset_name}\n\n{results}"
        return f"No knowledge graph triplets found for query {query!r} in dataset {dataset_name}."
    except Exception as e:
        logger.warning("MCP search_product_graph failed: %s", e)
        return f"Error querying cognee knowledge graph: {e}"


async def _mcp_ask_expert(product_id: str, query: str, model: Optional[str] = None) -> str:
    try:
        from api.expert_agent import run_expert_chat

        ans = run_expert_chat(product_id=product_id, query=query, model=model, stream=False)
        if hasattr(ans, "__await__"):
            res = await ans
            return str(res) if res else "Expert agent returned empty answer."
        return str(ans)
    except Exception as e:
        logger.warning("MCP ask_expert failed: %s", e)
        return f"Error asking expert agent: {e}"


async def _handle_mcp_request(req: McpJsonRpcRequest, db: Session) -> Dict[str, Any]:
    method = req.method
    msg_id = req.id
    params = req.params or {}

    # 1. initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": _MCP_SERVER_INFO,
            },
        }

    # 2. notifications/initialized
    if method in ("notifications/initialized", "initialized"):
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    # 3. tools/list
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": _TOOLS},
        }

    # 4. tools/call
    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments") or {}

        text_result = ""
        if tool_name == "list_products":
            text_result = _mcp_list_products(db)
        elif tool_name == "get_product_knowledge":
            pid = args.get("product_id") or ""
            fmt = (args.get("format") or "markdown").lower()
            text_result = _mcp_get_product_knowledge(pid, fmt, db)
        elif tool_name == "search_product_graph":
            pid = args.get("product_id") or ""
            q = args.get("query") or ""
            tk = int(args.get("top_k") or 20)
            text_result = await _mcp_search_product_graph(pid, q, tk)
        elif tool_name == "ask_expert":
            pid = args.get("product_id") or ""
            q = args.get("query") or ""
            m = args.get("model")
            text_result = await _mcp_ask_expert(pid, q, m)
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name!r}"},
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": text_result,
                    }
                ]
            },
        }

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method!r}"},
    }


# --- Endpoints ---------------------------------------------------------------
@router.get("/sse")
async def mcp_sse_endpoint(request: Request):
    """Open an SSE connection for MCP clients."""
    session_id = str(uuid.uuid4())

    async def sse_stream():
        # Send endpoint event with session message target
        yield f"event: endpoint\ndata: /api/mcp/message?session_id={session_id}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@router.post("")
@router.post("/message")
async def mcp_message_endpoint(
    req: McpJsonRpcRequest,
    db: Session = Depends(get_db),
):
    """Handle incoming MCP JSON-RPC 2.0 requests."""
    res = await _handle_mcp_request(req, db)
    return JSONResponse(content=res)
