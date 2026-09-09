"""Agent tools for registered HTTP integrations (issue #3).

Each enabled :class:`api.models.HttpIntegrationORM` row becomes one LangChain
tool named after the (sanitized) integration name — so MCP servers and HTTP
integrations are both "available to agents by name" (the user's wording).
The tool performs a bounded read-only GET:

- ``url_template`` placeholders are substituted with the agent-supplied values
  (declared ``variables``) plus the implicit ``{product_name}``;
- ``headers`` are decrypted only here, on the way out;
- the call is bounded by the ``integration_http`` timeout from
  :mod:`api.config.timeout` and the result is capped at
  ``MCP_TOOL_RESULT_MAX_CHARS`` (same guardrail as MCP tool results).

Never raises out of the tool: a failed call returns a short error STRING so
the agent conversation stays alive.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

#: Max declared variables per integration (mirror of the router's cap).
_MAX_VARIABLES = 16

_VARIABLE_NAME_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def sanitize_tool_name(name: str) -> str:
    """Turn an integration name into a valid LangChain tool name."""
    cleaned = "".join(ch if ch in _VARIABLE_NAME_CHARS else "_" for ch in (name or "").strip())
    cleaned = cleaned.strip("_")[:64]
    if not cleaned:
        cleaned = "http_integration"
    if cleaned[0].isdigit():
        cleaned = "http_" + cleaned
    return cleaned


def normalize_variables(raw: Optional[List[Any]]) -> List[Dict[str, str]]:
    """Coerce a stored ``variables`` JSON list into ``[{name, description, default}]``."""
    out: List[Dict[str, str]] = []
    for v in raw or []:
        if not isinstance(v, dict):
            continue
        name = str(v.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": str(v.get("description") or ""),
                "default": str(v.get("default") or ""),
            }
        )
    return out[:_MAX_VARIABLES]


def _product_name(product_id: str) -> str:
    """Best-effort product name for the ``{product_name}`` placeholder."""
    try:
        from api.db import SessionLocal
        from api.models import ProductORM

        with SessionLocal() as db:
            p = db.get(ProductORM, product_id)
            if p is not None and getattr(p, "name", None):
                return p.name
    except Exception as e:  # pragma: no cover - non-fatal
        logger.debug("http_tools: product name lookup failed for %r: %s", product_id, e)
    return product_id


def build_http_integration_tool(row: Any, product_id: str) -> Any:
    """Build one LangChain tool for an ``HttpIntegrationORM`` row."""
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    from api.config.timeout import resolve_integration_http_timeout
    from api.mcp.manager import tool_result_max_chars
    from api.mcp.secrets import decrypt_secret_dict

    variables = normalize_variables(getattr(row, "variables", None))
    url_template = str(getattr(row, "url_template", "") or "")
    integration_name = str(getattr(row, "name", "") or "integration")
    headers = decrypt_secret_dict(getattr(row, "headers", None))
    product_name = _product_name(product_id)

    # Dynamic args schema: one optional string param per declared variable.
    fields: Dict[str, Any] = {}
    for v in variables:
        fields[v["name"]] = (
            Optional[str],
            Field(default=v["default"] or "", description=v["description"] or v["name"]),
        )
    args_model = create_model(f"{sanitize_tool_name(integration_name)}_args", **fields)

    description = (str(getattr(row, "description", "") or "")).strip()
    if not description:
        description = (
            f"HTTP integration '{integration_name}': GET {url_template}. "
            "Returns the raw response body (status line prefixed)."
        )

    async def _call(**kwargs: Any) -> str:
        import httpx

        values = {"product_name": product_name}
        for v in variables:
            supplied = kwargs.get(v["name"])
            raw = "" if supplied is None else str(supplied)
            if raw == "":
                raw = v["default"]
            values[v["name"]] = quote(raw, safe="")
        url = url_template
        for key, value in values.items():
            url = url.replace("{" + key + "}", value)

        limit = tool_result_max_chars()
        try:
            timeout = resolve_integration_http_timeout()
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers or None)
            body = f"HTTP {resp.status_code}\n{resp.text}"
        except Exception as e:  # never raise out of the tool
            logger.warning(
                "http integration %r call failed: %s", integration_name, e
            )
            return f"HTTP integration '{integration_name}' failed: {type(e).__name__}"
        if len(body) > limit:
            body = (
                body[:limit]
                + f"\n…[HTTP integration result truncated: {len(body)} chars total,"
                f" kept first {limit}]"
            )
        return body

    return StructuredTool.from_function(
        coroutine=_call,
        name=sanitize_tool_name(integration_name),
        description=description,
        args_schema=args_model,
    )


def build_http_integration_tools(
    product_id: str,
    session_factory: Optional[Any] = None,
) -> List[Any]:
    """Build the tools of all enabled HTTP integrations (best-effort, never raises).

    HTTP integrations are product-agnostic (a global admin registry), so every
    enabled row is exposed to every product's agent — the tool bodies carry
    the product context only via the ``{product_name}`` placeholder.
    """
    try:
        from api.models import HttpIntegrationORM

        factory = session_factory
        if factory is None:
            from api.db import SessionLocal as factory  # type: ignore[no-redef]

        with factory() as db:
            rows = (
                db.query(HttpIntegrationORM)
                .filter(HttpIntegrationORM.enabled.is_(True))
                .order_by(HttpIntegrationORM.created_at)
                .all()
            )
            return [build_http_integration_tool(row, product_id) for row in rows]
    except Exception as e:  # pragma: no cover - best-effort by contract
        logger.debug("http integration tools unavailable: %s", e)
        return []


__all__ = [
    "build_http_integration_tool",
    "build_http_integration_tools",
    "normalize_variables",
    "sanitize_tool_name",
]
