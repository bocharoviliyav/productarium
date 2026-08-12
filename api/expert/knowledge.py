"""Expert knowledge retrieval + fallback + product-name lookup + history rendering.

Split out of the former ``api/expert_agent.py`` (Step 6). Owns:
- ``_retrieve_product_knowledge``: cognee recall over the product-scoped dataset
  ``prod_{product_id}`` with artifact-docs + live-Confluence fallbacks. Never
  raises; returns "" when nothing is available.
- ``_fallback_artifact_docs``: concatenates artifact ``generated_docs`` / page
  content when cognee is empty (own short-lived DB session, non-fatal).
- ``_product_name_by_id``: DB product-name lookup with id fallback (non-fatal).
- ``_format_history``: render prior conversation turns as a history block.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from api.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _product_name_by_id(product_id: str) -> str:
    """Look up a product name from the DB; fall back to the id. Non-fatal."""
    try:
        from api.db import SessionLocal
        from api.models import ProductORM

        with SessionLocal() as db:
            p = db.get(ProductORM, product_id)
            if p is not None and getattr(p, "name", None):
                return p.name
    except Exception as e:
        logger.debug("product name lookup failed for %r: %s", product_id, e)
    return product_id


def _fallback_artifact_docs(product_id: str) -> str:
    """Concatenate artifact generated_docs + page content when cognee is empty.

    Opens its own short-lived session (non-fatal: returns "" on any error or
    when the product/artifacts are missing).
    """
    try:
        from sqlalchemy.orm import selectinload

        from api.db import SessionLocal
        from api.models import ProductORM

        with SessionLocal() as db:
            p = (
                db.query(ProductORM)
                .options(selectinload(ProductORM.artifacts))
                .filter(ProductORM.id == product_id)
                .first()
            )
            if p is None:
                return ""
            parts: List[str] = []
            for a in p.artifacts:
                name = getattr(a, "name", None) or getattr(a, "id", "artifact")
                docs = getattr(a, "generated_docs", None) or ""
                if docs:
                    parts.append(f"## Artifact: {name}\n{docs}")
                pages = getattr(a, "pages", None) or {}
                if isinstance(pages, dict):
                    for page_id, page in pages.items():
                        content = ""
                        if isinstance(page, dict):
                            content = page.get("content") or ""
                        elif isinstance(page, str):
                            content = page
                        if content:
                            parts.append(
                                f"## Artifact: {name} / page {page_id}\n{content}"
                            )
            return "\n\n".join(parts)
    except Exception as e:
        logger.warning(
            "Fallback artifact docs failed for product %r: %s", product_id, e
        )
        return ""


async def _retrieve_product_knowledge(product_id: str, query: str) -> str:
    """Retrieve product knowledge from cognee (prod_{product_id}); fall back to
    concatenated artifact docs or live Confluence. Never raises; returns "" if nothing available.
    """
    dataset = f"prod_{product_id}"
    try:
        from api.cognee_manager import query_cognee

        ctx = await query_cognee(query, dataset_name=dataset, top_k=20)
        if ctx:
            logger.info(
                "Expert: cognee recall for %r returned %d chars.", dataset, len(ctx)
            )
            return ctx
        logger.info("Expert: cognee recall empty for %r; using artifact fallback.", dataset)
    except Exception as e:
        logger.warning("Expert: cognee recall failed for %r: %s", dataset, e)

    fallback_docs = _fallback_artifact_docs(product_id)
    if fallback_docs:
        return fallback_docs

    # Fallback to live Confluence (direct or MCP) if configured
    try:
        from api.integrations.registry import get_connector
        c_connector = get_connector("confluence")
        if c_connector and c_connector.is_configured():
            spaces = c_connector.list_spaces()
            if spaces:
                sp_id = spaces[0].get("key") or spaces[0].get("id")
                if sp_id:
                    pulled = c_connector.pull(sp_id, opts={"recursive": False})
                    if pulled and pulled.get("markdown"):
                        return pulled["markdown"]
    except Exception as e:
        logger.debug("Expert live Confluence fallback skipped for %s: %s", product_id, e)

    return ""


def _format_history(messages: List[Dict[str, Any]]) -> str:
    """Render prior conversation turns (user/assistant pairs) as a history block."""
    if not messages:
        return ""
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not role or not content:
            continue
        if role == "user":
            lines.append(f"<user>{content}</user>")
        elif role == "assistant":
            lines.append(f"<assistant>{content}</assistant>")
        else:
            lines.append(f"<{role}>{content}</{role}>")
    return "\n".join(lines)
