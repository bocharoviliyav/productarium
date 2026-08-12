"""Misc router — small leaf endpoints that don't fit a dedicated domain.

Endpoints (no prefix, tags ``misc``):
- ``GET /lang/config`` — return the language config (used by the i18n context).
- ``GET /health``       — Docker/monitoring healthcheck.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from api.config import configs

router = APIRouter(tags=["misc"])


@router.get("/lang/config")
async def get_lang_config():
    return configs["lang_config"]


@router.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "productarium-api",
    }


__all__ = ["router"]
