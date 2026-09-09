"""Inbound MCP server (Wave C, contract 3): Productarium AS an MCP server.

Exposes the platform over the Model Context Protocol at ``/api/mcp`` so
external MCP clients (Claude Desktop, IDEs, other agents) can consume
Productarium knowledge:

- ``list_products``              — all products
- ``get_product_knowledge``      — verified knowledge as markdown (fallback: all)
- ``search_knowledge``           — pgvector/recall search with citations
- ``ask_expert``                 — the expert agent over a product (bounded)

Transport: ``mcp.server.fastmcp.FastMCP`` streamable HTTP, **stateless** +
JSON responses (one JSON-RPC request per POST, no sessions) mounted into the
FastAPI app via ``api.api``. DNS-rebinding protection is disabled because the
server is mounted behind the main app (not bound to its own host).

Auth: a pure-ASGI wrapper (``_BearerAuthASGI``) around the mounted app checks
the ``Authorization: Bearer <token>`` header on EVERY http request against the
``api_tokens`` table (sha256, same store as ``require_api_token``) and stamps
``last_used_at``. Invalid/missing tokens get ``401`` + ``WWW-Authenticate``.

Sizing: knowledge exports are capped (``_MAX_KNOWLEDGE_CHARS``), expert
answers are capped (``_MAX_ANSWER_CHARS``) and the expert call is bounded by
``MCP_ASK_TIMEOUT_SECONDS`` (default 120 s).

All DB/service imports are lazy (inside the tools/wrapper) so import of this
module never touches the DB and tests can rebind ``api.db.SessionLocal``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- sizing / timing knobs -----------------------------------------------------
_MAX_KNOWLEDGE_CHARS = 100_000
_MAX_ANSWER_CHARS = 50_000
_DEFAULT_ASK_TIMEOUT = 120.0
_MAX_QUERY_CHARS = 20_000


def _ask_timeout() -> float:
    try:
        raw = float(os.environ.get("MCP_ASK_TIMEOUT_SECONDS", "") or _DEFAULT_ASK_TIMEOUT)
    except (TypeError, ValueError):
        return _DEFAULT_ASK_TIMEOUT
    return max(1.0, raw)


# --- bearer auth (ASGI) ----------------------------------------------------------
def _check_api_token(raw: str) -> bool:
    """Validate a Bearer token against ``api_tokens`` (sha256) + touch last_used_at.

    Lazy ``api.db`` import so tests rebinding ``SessionLocal`` are honored.
    Returns False on any failure (DB down == unauthenticated).
    """
    try:
        from datetime import datetime

        from api.db import SessionLocal
        from api.models import ApiTokenORM

        token_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        session = SessionLocal()
        try:
            tok = (
                session.query(ApiTokenORM)
                .filter(ApiTokenORM.token_hash == token_hash)
                .first()
            )
            if tok is None:
                return False
            tok.last_used_at = datetime.utcnow()
            try:
                session.commit()
            except Exception:
                session.rollback()
            return True
        finally:
            session.close()
    except Exception as e:
        logger.warning("mcp inbound: API token check failed: %s", e)
        return False


async def _send_unauthorized(send) -> None:
    """401 JSON response with the WWW-Authenticate challenge."""
    body = b'{"detail":"Missing or invalid API token"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b"Bearer"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class _BearerAuthASGI:
    """Pure-ASGI bearer-auth wrapper (no FastAPI dependency stack on this path)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):  # noqa: D401 - ASGI protocol
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        auth = headers.get("authorization", "")
        raw = ""
        if auth.lower().startswith("bearer "):
            raw = auth.split(" ", 1)[1].strip()
        # _check_api_token is sync SQLAlchemy — run it off the event loop so
        # concurrent MCP requests don't serialize (each blocking all others).
        if not raw or not await asyncio.to_thread(_check_api_token, raw):
            await _send_unauthorized(send)
            return
        await self.app(scope, receive, send)


# --- the MCP server --------------------------------------------------------------
def _build_mcp():
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    mcp = FastMCP(
        name="productarium",
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        # The app is mounted inside the main FastAPI app (not bound to its own
        # host), so the built-in localhost DNS-rebinding guard must be off.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )

    # ---- tools (contract 3) ----
    @mcp.tool()
    def list_products() -> List[Dict[str, Any]]:
        """List all Productarium products (id, name, summary)."""
        from api.db import SessionLocal
        from api.models import ProductORM

        session = SessionLocal()
        try:
            rows = session.query(ProductORM).order_by(ProductORM.created_at).all()
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "summary": (p.summary or "")[:2000] or None,
                }
                for p in rows
            ]
        finally:
            session.close()

    @mcp.tool()
    def get_product_knowledge(product_id: str) -> str:
        """Export a product's VERIFIED knowledge as Markdown.

        If nothing is verified yet, falls back to ALL product content so the
        tool stays useful during onboarding. Output is capped in length.
        """
        from api.db import SessionLocal
        from api.models import (
            CodebaseORM,
            KnowledgeNodeORM,
            LinksORM,
            ProductORM,
            SpecORM,
        )
        from api.routers.public import (
            _knowledge_as_markdown,
            _load_verified,
        )

        session = SessionLocal()
        try:
            product = session.get(ProductORM, product_id)
            if product is None:
                raise ValueError("Product not found")
            codebases, specs, links, nodes = _load_verified(product_id, session)
            if not (codebases or specs or links or nodes):
                # Nothing verified yet — expose everything (read-only tool).
                codebases = (
                    session.query(CodebaseORM).filter_by(product_id=product_id).all()
                )
                specs = session.query(SpecORM).filter_by(product_id=product_id).all()
                links = session.query(LinksORM).filter_by(product_id=product_id).all()
                nodes = (
                    session.query(KnowledgeNodeORM)
                    .filter_by(product_id=product_id)
                    .all()
                )
            md = _knowledge_as_markdown(product, codebases, specs, links, nodes)
            if len(md) > _MAX_KNOWLEDGE_CHARS:
                md = md[:_MAX_KNOWLEDGE_CHARS] + "\n\n…(truncated)"
            return md
        finally:
            session.close()

    @mcp.tool()
    async def search_knowledge(product_id: str, query: str, top_k: int = 8) -> str:
        """Search a product's indexed knowledge; returns cited evidence blocks.

        ``top_k`` is clamped to 1..20.
        """
        from api.agents.tools import (
            format_hits_with_citations,
            search_knowledge_hits,
        )

        query = (query or "")[:_MAX_QUERY_CHARS]
        if not query.strip():
            raise ValueError("query must not be empty")
        k = max(1, min(int(top_k or 8), 20))
        hits = await search_knowledge_hits(query, product_id, top_k=k)
        out = format_hits_with_citations(hits)
        return out[:_MAX_KNOWLEDGE_CHARS]

    @mcp.tool()
    async def ask_expert(product_id: str, query: str) -> str:
        """Ask the product's expert agent a question; returns the full answer."""
        query = (query or "")[:_MAX_QUERY_CHARS]
        if not query.strip():
            raise ValueError("query must not be empty")
        try:
            from api.expert.chat import run_expert_chat  # lazy: monkeypatch point
        except Exception as e:
            logger.warning("mcp inbound: expert chat unavailable: %s", e)
            raise ValueError("Expert agent is not available")

        agen = run_expert_chat(
            product_id=product_id, query=query, messages=[], model=None, stream=False
        )
        try:
            answer = await asyncio.wait_for(agen, timeout=_ask_timeout())
        except asyncio.TimeoutError:
            logger.warning(
                "mcp inbound: ask_expert timed out after %.0fs", _ask_timeout()
            )
            raise ValueError("Expert agent timed out")
        except ValueError:
            raise
        except Exception as e:
            logger.error("mcp inbound: ask_expert failed: %s", e, exc_info=True)
            raise ValueError("Expert agent failed")
        answer = answer or ""
        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS] + "…(truncated)"
        return answer

    return mcp


#: Module-level singleton (import-safe: FastMCP construction does no I/O).
_mcp = None
_mcp_app = None
_wrapped_app: Optional[_BearerAuthASGI] = None


def get_inbound_mcp():
    """The inbound FastMCP singleton (built on first use)."""
    global _mcp
    if _mcp is None:
        _mcp = _build_mcp()
    return _mcp


def get_inbound_mcp_app() -> Any:
    """The bearer-auth-wrapped streamable-HTTP ASGI app (mounted at /api/mcp)."""
    global _mcp_app, _wrapped_app
    if _wrapped_app is None:
        _mcp_app = get_inbound_mcp().streamable_http_app()
        _wrapped_app = _BearerAuthASGI(_mcp_app)
    return _wrapped_app


@asynccontextmanager
async def inbound_session_manager():
    """Run the inbound MCP session manager inside the app lifespan.

    Starlette does not propagate lifespans to mounted sub-apps, so the main
    app (``api.api``) must enter this context manager in its own lifespan.
    """
    async with get_inbound_mcp().session_manager.run():
        yield


__all__ = [
    "get_inbound_mcp",
    "get_inbound_mcp_app",
    "inbound_session_manager",
]
