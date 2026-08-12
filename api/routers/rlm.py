"""RLM router — fast-rlm Recursive Language Model endpoint.

Endpoint (prefix ``/api``, tags ``rlm``):
- ``POST /api/rlm/run`` — run an RLM reasoning task.

Thin wrapper over ``api.rlm.runner``; the heavy work (Deno + Pyodide
bootstrap, long-context reasoning) lives there.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["rlm"])


class RLMRunRequest(BaseModel):
    query: str
    model: Optional[str] = None


@router.post("/rlm/run")
async def run_rlm_endpoint(request_data: RLMRunRequest):
    from api.rlm.runner import run_rlm_task
    try:
        result = await run_rlm_task(request_data.query, request_data.model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


__all__ = ["router"]
