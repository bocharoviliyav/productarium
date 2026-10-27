"""Inbound MCP server (Wave C, contract 3): Productarium AS an MCP server.

Exposes the platform over the Model Context Protocol at ``/api/mcp`` so
external MCP clients (Claude Desktop, IDEs, other agents) can consume
Productarium knowledge:

- ``list_products``              — all products
- ``get_product_knowledge``      — VERIFIED knowledge as markdown (strict)
- ``knowledge_search``           — SPEC-008 graph-aware search with citations
- ``knowledge_graph``            — SPEC-008 product term/page graph (sub)JSON
- ``ask_expert``                 — the expert agent over a product (bounded)

Transport: ``mcp.server.fastmcp.FastMCP`` streamable HTTP, **stateless** +
JSON responses (one JSON-RPC request per POST, no sessions) mounted into the
FastAPI app via ``api.api``. DNS-rebinding protection is disabled because the
server is mounted behind the main app (not bound to its own host).

Auth: a pure-ASGI wrapper (``_BearerAuthASGI``) around the mounted app checks
the ``Authorization: Bearer <token>`` header on EVERY http request against the
``api_tokens`` table (sha256, same store as ``require_api_token``) and stamps
``last_used_at``. Invalid/missing tokens get ``401`` + ``WWW-Authenticate``;
valid ones are per-token-owner rate-limited (the shared ``rate.public.per_user_min``
bucket of the public ask API; exhaustion → ``429`` + ``Retry-After``).

Sizing: knowledge exports are capped (``_MAX_KNOWLEDGE_CHARS``), expert
answers are capped (``_MAX_ANSWER_CHARS``) and the expert call is bounded by
``MCP_ASK_TIMEOUT_SECONDS`` (default 120 s).

All DB/service imports are lazy (inside the tools/wrapper) so import of this
module never touches the DB and tests can rebind ``api.db.SessionLocal``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

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
def _check_api_token(raw: str) -> str:
    """Validate a Bearer token against ``api_tokens`` (sha256) + touch last_used_at.

    Lazy ``api.db`` import so tests rebinding ``SessionLocal`` are honored.
    Returns the rate-limit identity (``user_id`` or the token id) on success,
    or the empty string on any failure (DB down == unauthenticated).
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
                return ""
            tok.last_used_at = datetime.utcnow()
            try:
                session.commit()
            except Exception:
                session.rollback()
            return str(tok.user_id or tok.id)
        finally:
            session.close()
    except Exception as e:
        logger.warning("mcp inbound: API token check failed: %s", e)
        return ""


def _auth_and_rate_limit(raw: str) -> str:
    """Token check + per-owner rate limit (P-4); raises HTTPException(429).

    Reuses the public ask bucket (``rate.public.per_user_min``) so one token
    cannot bypass the public API budget through the MCP surface.
    """
    from api.utils.rate_limit import enforce_user_rate_limit

    identity = _check_api_token(raw)
    if identity:
        enforce_user_rate_limit(
            identity,
            setting_key="rate.public.per_user_min",
            env_name="RATE_PUBLIC_PER_USER_MINUTE",
            default_per_minute=30,
        )
    return identity


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


async def _send_http_exception(send, exc: HTTPException) -> None:
    """Minimal JSON response for an HTTPException raised on the ASGI path."""
    body = json.dumps({"detail": exc.detail}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    headers.extend(
        (str(k).lower().encode("ascii"), str(v).encode("ascii"))
        for k, v in (exc.headers or {}).items()
    )
    await send(
        {
            "type": "http.response.start",
            "status": int(exc.status_code),
            "headers": headers,
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
        # Token check + rate limit are sync SQLAlchemy/settings reads — run
        # them off the event loop so concurrent MCP requests don't serialize.
        identity = ""
        if raw:
            try:
                identity = await asyncio.to_thread(_auth_and_rate_limit, raw)
            except HTTPException as exc:
                await _send_http_exception(send, exc)
                return
        if not identity:
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

        Strictly verified-only (P-5): unverified content is never exposed;
        nothing verified yet → an empty export. Output is capped in length.
        """
        from api.db import SessionLocal
        from api.models import ProductORM
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
            md = _knowledge_as_markdown(product, codebases, specs, links, nodes)
            if len(md) > _MAX_KNOWLEDGE_CHARS:
                md = md[:_MAX_KNOWLEDGE_CHARS] + "\n\n…(truncated)"
            return md
        finally:
            session.close()

    async def _search_cited(product_id: str, query: str, top_k: int) -> str:
        from api.agents.tools import (
            format_hits_with_citations,
            search_knowledge_hits,
        )

        if not (product_id or "").strip():
            raise ValueError("product_id is required")
        query = (query or "")[:_MAX_QUERY_CHARS]
        if not query.strip():
            raise ValueError("query must not be empty")
        k = max(1, min(int(top_k or 8), 20))
        hits = await search_knowledge_hits(query, product_id, top_k=k)
        out = format_hits_with_citations(hits)
        return out[:_MAX_KNOWLEDGE_CHARS]

    @mcp.tool()
    async def knowledge_search(product_id: str, query: str, top_k: int = 8) -> str:
        """Graph-aware knowledge search (SPEC-008): cited evidence blocks.

        Query terms are resolved in the product's knowledge graph when one
        exists (related pages' chunks get boosted); otherwise plain cosine +
        lexical recall. Scope is strictly one product. ``top_k`` 1..20.
        """
        return await _search_cited(product_id, query, top_k)

    @mcp.tool()
    async def knowledge_graph(product_id: str, focus_term: Optional[str] = None) -> str:
        """A product's knowledge graph (SPEC-008) as nodes/edges JSON.

        With ``focus_term`` — its 1-hop neighbourhood, otherwise the whole
        graph capped to the most connected nodes (shrunk further while the
        serialized JSON exceeds the size cap, so output is always valid JSON).
        Empty graph (no nodes) when the product has none yet.
        """
        from api.memory.knowledge_graph import load_product_graph, subgraph

        if not (product_id or "").strip():
            raise ValueError("product_id is required")
        graph = await load_product_graph(product_id)
        focus = (focus_term or "")[:_MAX_QUERY_CHARS] or None
        max_nodes = 300
        out = json.dumps(
            subgraph(graph, focus_term=focus, max_nodes=max_nodes),
            ensure_ascii=False, sort_keys=True,
        )
        # Over the cap: shrink the node budget and re-serialize — a raw
        # string cut would leave the client with invalid JSON (Q-7).
        while len(out) > _MAX_KNOWLEDGE_CHARS and max_nodes > 25:
            max_nodes = max(25, max_nodes // 2)
            out = json.dumps(
                subgraph(graph, focus_term=focus, max_nodes=max_nodes),
                ensure_ascii=False, sort_keys=True,
            )
        return out

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
