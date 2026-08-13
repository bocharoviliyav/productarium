"""Expert agent router (contract J, item 3 — replaces Long-context tasks).

Endpoints (prefix ``/api/products``, tags ``expert``):
- ``POST /api/products/{product_id}/ask``
    Stream an expert-chat answer as Server-Sent Events (SSE). Each event is
    ``data: {"content": "<chunk>"}\\n\\n``; the stream ends with ``data: [DONE]\\n\\n``.
    On error an event ``data: {"error": "..."}\\n\\n`` is emitted before [DONE].
- ``POST /api/products/{product_id}/ask/doc``
    Generate a self-contained Markdown document and return it as a downloadable
    file (``Content-Disposition: attachment``).

Both endpoints require an authenticated session (``get_current_user``). The
heavy lifting (cognee recall, RLM routing, LLM streaming) lives in the
``api.expert`` package (``api.expert.chat``); this router only does request
parsing, SSE framing, and file-response packaging so it stays thin and
file-disjoint from the foundation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from api.auth.deps import get_current_user
from api.expert.chat import run_expert_chat, run_expert_doc
from api.expert.types import ExpertStreamEvent
from api.models import UserORM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["expert"])


def _format_sse(chunk: object) -> str:
    """Map a stream chunk to an SSE ``data:`` frame.

    ``ExpertStreamEvent`` → typed frame (``{content}`` / ``{reasoning}`` /
    ``{status}`` / ``{error}``). Plain ``str`` → ``{"content": ...}``
    (backward compat with old test mocks that yield bare strings).
    """
    if isinstance(chunk, ExpertStreamEvent):
        return f"data: {json.dumps({chunk.type: chunk.content}, ensure_ascii=False)}\n\n"
    return f"data: {json.dumps({'content': str(chunk)}, ensure_ascii=False)}\n\n"


class ChatMessage(BaseModel):
    role: str
    content: str


class ExpertAskRequest(BaseModel):
    query: str
    messages: List[ChatMessage] = Field(default_factory=list)
    model: Optional[str] = None
    # Optional explicit LLM/RLM override. ``true`` forces RLM (with LLM
    # fallback), ``false`` forces the standard LLM, ``null`` (default) follows
    # the admin ``rlm.expert.mode`` setting (auto/rlm/llm).
    use_rlm: Optional[bool] = None


def _safe_filename(product_id: str) -> str:
    """Build a safe attachment filename for the doc download."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", product_id or "product") or "product"
    return f"productarium_{base}_expert.md"


@router.post("/{product_id}/ask")
async def expert_ask(
    product_id: str,
    body: ExpertAskRequest,
    user: UserORM = Depends(get_current_user),
):
    """Stream an expert-chat answer as SSE.

    Requires login. Emits ``data: {"content": "..."}`` events as the answer is
    produced and terminates with ``data: [DONE]``.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    async def event_stream():
        try:
            async for chunk in run_expert_chat(
                product_id, body.query, messages, body.model, stream=True,
                use_rlm=body.use_rlm,
            ):
                if chunk is None:
                    continue
                yield _format_sse(chunk)
        except Exception as e:  # pragma: no cover - defensive over streaming
            logger.error("expert /ask stream failed: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{product_id}/ask/doc")
async def expert_ask_doc(
    product_id: str,
    body: ExpertAskRequest,
    user: UserORM = Depends(get_current_user),
):
    """Generate a self-contained Markdown document and return it as a file.

    Requires login. Returns ``text/markdown`` with a ``Content-Disposition:
    attachment`` header. ``messages`` is accepted but ignored (doc generation is
    one-shot, not conversational).
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    try:
        md = await run_expert_doc(product_id, body.query, body.model, use_rlm=body.use_rlm)
    except Exception as e:  # pragma: no cover - defensive over generation
        logger.error("expert /ask/doc failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Expert document generation failed: {e}"
        )

    if not md:
        md = (
            f"# Expert document for {product_id}\n\n"
            "_(No content was generated. Ensure the product has indexed knowledge "
            "(cognee) or generated artifact docs, and that a local LLM is available.)_\n"
        )

    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={_safe_filename(product_id)}"
        },
    )


__all__ = ["router"]
