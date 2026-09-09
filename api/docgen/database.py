"""Database reverse-engineering flow (Wave E).

Documents a database artifact through the product's MCP servers:

1. **Introspection (deterministic, disk-cached)** — the flow resolves the
   product's MCP tools (``api.mcp.manager``): when the artifact pins
   ``mcp_server_id`` the tools of THAT server are used (it must be bound+enabled
   to the product, same visibility as the expert agent), otherwise all bound
   enabled servers contribute. KNOWN preset surfaces are handled by dedicated
   adapters BEFORE the generic heuristics: dbhub's ``search_objects``
   (schemas/tables listings + full per-table detail — uniform across
   PostgreSQL/MySQL/MariaDB/SQL Server/SQLite) and oracle-mcp-server's
   ``search_tables_schema``/``get_table_schema`` (arg names mapped at runtime).
   Everything else falls back to NAME heuristics (``list_schemas`` /
   ``list_tables`` / ``describe_table`` / ``get_table_ddl`` / …) because every
   other MCP database server names them differently. The walk is schemas →
   tables → per-table definitions, every
   call bounded by the manager's timeouts/caps (``MCP_TOOL_CALL_TIMEOUT_SECONDS``,
   ``MCP_TOOL_RESULT_MAX_CHARS``). The RESULT is cached on disk
   (``api.docgen.introspection_cache``, keyed by the MCP surface identity +
   walk budgets): a rerun or repair after an interruption — e.g. a failed LLM
   enrichment on a multi-hundred-MB schema — reuses the cached payload and
   never touches the database MCP server again (item 2.3b).
2. **Skeleton (deterministic)** — a markdown reference rendered from the
   collected schema (no LLM): overview, schemas, per-table definitions.
3. **Enrich** — the standard docgen LLM (``_llm_or_none`` over the
   ``database_doc.md`` prompt) turns the skeleton into full documentation.
   LLM failure degrades to the skeleton (the schema IS real evidence) — only
   a dead/unusable MCP surface is a hard error (an honest failed job, never
   empty docs).
4. **Guard (verification)** — mermaid repair loop, secret masking
   (``mask_secrets`` — the persisted docs never carry DSN passwords or
   tokens), an optional LLM judge (model-generated docs only; flags, never
   blocks) and per-page provenance (tools used, masking/judge verdict).
5. **Persist + index** — ``generated_docs`` + ``pages`` written onto the
   artifact (viewer contract) and the final markdown indexed into the active
   memory backend with ``source_type="database"``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from api.utils import setup_logging
from api.utils.llm_helpers import cap as _cap
from api.formats.mermaid import run_repair_loop
from api.prompts import LANGUAGE_NAMES, load_prompt_file
from api.docgen._common import (
    _close_owned_llm,
    _product_dataset,
    _index_in_background,
    _llm_or_none,
    _make_repair_llm,
    _persist_artifact,
    _resolve_docgen_model,
    emit_progress,
)
from api.docgen.corroborate import (
    filter_ungrounded_prose,
    grounding_from_introspection,
)
from api.docgen.fact_fold import fold, rank_split
from api.docgen.introspection_cache import (
    introspection_cache_key,
    load_introspection_cache,
    store_introspection_cache,
)
from api.docgen.verification import (
    build_section_provenance,
    judge_enabled,
    judge_section,
    mask_secrets,
)

setup_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Budgets (introspection walk)
# --------------------------------------------------------------------------- #
#: Max schemas introspected per database (extra schemas are noted + skipped).
MAX_SCHEMAS = 20
#: Max tables introspected per schema (extra tables are listed by name only).
MAX_TABLES_PER_SCHEMA = 100
#: Max characters of one table definition kept in the doc/evidence.
MAX_DEFINITION_CHARS = 8_000
#: Max characters of the raw schema dump handed to the LLM.
MAX_SCHEMA_DUMP_CHARS = 120_000
#: Overall wall-clock budget for one introspection walk (DoS guard, review
#: #4: per-call limits alone let a slow MCP surface occupy a docgen worker
#: for hours). Env-overridable; default 15 minutes.
def _env_float(name: str, default: float) -> float:
    """Import-time env parsing that never crashes on garbage values."""
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw)
    except ValueError:
        if raw:
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


INTROSPECTION_TIMEOUT_SECONDS = _env_float("DB_INTROSPECTION_TIMEOUT_SECONDS", 900.0)
#: Hard cap on MCP tool calls during one walk (schema listings + per-table
#: definitions combined) so huge schemas cannot loop unboundedly.
MAX_TOOL_CALLS = 2_000


# --------------------------------------------------------------------------- #
# MCP tool resolution (pinned server vs all bound enabled servers)
# --------------------------------------------------------------------------- #
async def _tools_for_pinned_server(
    product_id: str, mcp_server_id: str
) -> List[Any]:
    """Tools of the pinned server, honoring the product binding allowlist.

    Raises ValueError (honest job failure) when the server is gone, not bound
    to the product, or unreachable — the flow refuses to document from an
    MCP surface the product is not entitled to.
    """
    from api.db import SessionLocal
    from api.models import McpServerORM, ProductMcpServerORM
    from api.mcp.manager import discovery_timeout, get_mcp_manager

    session = SessionLocal()
    try:
        server = session.get(McpServerORM, mcp_server_id)
        if server is None:
            raise ValueError(
                f"Pinned MCP server {mcp_server_id!r} no longer exists; "
                "update the database artifact's mcp_server_id."
            )
        binding = (
            session.query(ProductMcpServerORM)
            .filter(
                ProductMcpServerORM.product_id == product_id,
                ProductMcpServerORM.mcp_server_id == mcp_server_id,
            )
            .first()
        )
        if binding is None or not binding.enabled or not server.enabled:
            raise ValueError(
                f"Pinned MCP server {server.name!r} is not bound+enabled to "
                f"the product {product_id}; bind it via the product MCP panel."
            )
        allow = binding.allowed_tools if isinstance(binding.allowed_tools, list) else None
        allow_set = {str(t) for t in allow} if allow is not None else None
    finally:
        session.close()

    manager = get_mcp_manager()
    try:
        tools = await asyncio.wait_for(
            manager.discover_tools(server), timeout=discovery_timeout() + 5.0
        )
    except Exception as e:
        raise ValueError(
            f"MCP server {server.name!r} is unreachable "
            f"({type(e).__name__}); reverse-engineering aborted."
        ) from e
    if allow_set is not None:
        tools = [t for t in tools if (getattr(t, "name", "") or "") in allow_set]
    return list(tools)


async def _resolve_mcp_tools(entity: Any, product_id: str) -> List[Any]:
    """MCP tools for the artifact: pinned server or all bound enabled servers."""
    mcp_server_id = getattr(entity, "mcp_server_id", None)
    if mcp_server_id:
        return await _tools_for_pinned_server(product_id, mcp_server_id)
    from api.mcp.manager import gather_mcp_agent_tools

    tools = await gather_mcp_agent_tools(product_id)
    return list(tools or [])


# --------------------------------------------------------------------------- #
# Introspection disk cache key (2.3b)
# --------------------------------------------------------------------------- #
def _cache_budgets() -> Dict[str, Any]:
    """Walk-budget components of the cache key: a budget change reshapes the
    introspection payload, so it must invalidate cached entries."""
    return {
        "max_schemas": MAX_SCHEMAS,
        "max_tables_per_schema": MAX_TABLES_PER_SCHEMA,
        "max_definition_chars": MAX_DEFINITION_CHARS,
        "max_schema_dump_chars": MAX_SCHEMA_DUMP_CHARS,
        "max_tool_calls": MAX_TOOL_CALLS,
    }


def _cache_binding_inputs(
    product_id: str, pinned_server_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """DB inputs of the introspection cache key, mirroring tool visibility.

    The pinned path is keyed by (server id, binding allowlist) — exactly what
    ``_tools_for_pinned_server`` filters on; the gather path by the product's
    enabled binding ids (server enabled too), which is what
    ``gather_mcp_agent_tools`` walks. ``None`` (DB error, empty product id,
    or a pin that is not bound+enabled) BYPASSES the cache — the normal path
    then either introspects fresh or raises its honest error, unchanged.
    """
    if not product_id:
        return None
    try:
        from api.db import SessionLocal
        from api.models import McpServerORM, ProductMcpServerORM

        session = SessionLocal()
        try:
            rows = (
                session.query(ProductMcpServerORM)
                .join(McpServerORM, ProductMcpServerORM.mcp_server_id == McpServerORM.id)
                .filter(
                    ProductMcpServerORM.product_id == product_id,
                    ProductMcpServerORM.enabled.is_(True),
                    McpServerORM.enabled.is_(True),
                )
                .order_by(ProductMcpServerORM.id)
                .all()
            )
        finally:
            session.close()
    except Exception as e:  # pragma: no cover - cache bookkeeping is never fatal
        logger.debug("cache binding inputs unavailable (cache bypassed): %s", e)
        return None
    if pinned_server_id:
        allow: Optional[List[str]] = None
        for row in rows:
            if row.mcp_server_id == pinned_server_id:
                allow = sorted({str(t) for t in (row.allowed_tools or [])})
                break
        if allow is None:
            # Pin not bound+enabled — the resolve step below raises the honest
            # error; caching must not mask (nor serve) that state.
            return None
        return {"mcp_server_id": str(pinned_server_id), "allowlist": allow}
    return {"binding_ids": [row.id for row in rows]}


def _introspection_cache_key(entity: Any, product_id: str) -> Optional[str]:
    """Cache key for this artifact's introspection surface (None = bypass)."""
    inputs = _cache_binding_inputs(
        product_id, getattr(entity, "mcp_server_id", None)
    )
    if inputs is None:
        return None
    return introspection_cache_key(budgets=_cache_budgets(), **inputs)


# --------------------------------------------------------------------------- #
# Tool classification + bounded invocation
# --------------------------------------------------------------------------- #
def _classify_introspection_tools(tools: List[Any]) -> Dict[str, List[Any]]:
    """Split discovered MCP tools into introspection roles by name heuristics.

    Database MCP servers name their introspection tools differently
    (``list_schemas``/``get_schemas``, ``list_tables``/``search_tables``,
    ``describe_table``/``get_table_info``/``get_table_ddl`` …), so the roles
    are matched on name tokens rather than exact names. First match per tool
    wins; unrelated tools (query execution, health, …) are ignored.
    """
    roles: Dict[str, List[Any]] = {"schemas": [], "tables": [], "describe": [], "ddl": []}
    for tool in tools or []:
        name = (getattr(tool, "name", "") or "").lower()
        if not name:
            continue
        has_table = "table" in name
        if has_table and ("ddl" in name or "definition" in name or "create" in name):
            roles["ddl"].append(tool)
        elif has_table and (
            "describe" in name
            or "info" in name
            or "detail" in name
            or "column" in name
            or "structure" in name
        ):
            roles["describe"].append(tool)
        elif (
            "schema" in name
            and not has_table
            and ("list" in name or "show" in name or "get" in name or "all" in name)
        ):
            roles["schemas"].append(tool)
        elif has_table and ("list" in name or "search" in name or "show" in name or "get" in name):
            roles["tables"].append(tool)
    return roles


def _tool_arg_names(tool: Any) -> List[str]:
    """Declared argument names of a LangChain tool (best-effort)."""
    try:
        args = getattr(tool, "args", None)
        if isinstance(args, dict):
            return [str(a) for a in args.keys() if str(a) not in ("kwargs",)]
    except Exception:  # pragma: no cover - defensive over tool schemas
        pass
    return []


def _build_tool_args(
    tool: Any,
    *,
    schema: Optional[str] = None,
    table: Optional[str] = None,
) -> Dict[str, Any]:
    """Map (schema, table) onto the tool's declared argument names.

    Server-specific tools use different argument spellings; match on name
    tokens (``schema``/``database``-ish params get the schema, ``table``/
    ``name``-ish params get the table). Unmappable required args make the
    call fail — recorded per table and skipped (best-effort by design).
    """
    out: Dict[str, Any] = {}
    for arg in _tool_arg_names(tool):
        low = arg.lower()
        if schema is not None and (
            "schema" in low or "database" in low or low == "db"
        ):
            out[arg] = schema
        elif table is not None and (
            "table" in low or low in ("name", "object", "object_name", "entity")
        ):
            out[arg] = table
    return out


def _block_text(block: Any) -> str:
    """Text payload of ONE content block (MCP dict or object form)."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        text = block.get("text")
        if isinstance(text, str):
            return text
        if "content" in block:
            return _result_text(block["content"]) or ""
        try:
            return json.dumps(block, ensure_ascii=False, default=str)
        except Exception:  # pragma: no cover - defensive
            return str(block)
    text = getattr(block, "text", None)  # mcp.types.TextContent and lookalikes
    if isinstance(text, str):
        return text
    content = getattr(block, "content", None)  # ToolMessage-like wrapper
    if content is not None:
        return _result_text(content) or ""
    return ""


def _result_text(result: Any) -> Optional[str]:
    """Best-effort TEXT of a tool result (``None`` = not text-extractable).

    MCP tool results arrive in several shapes depending on the client stack:
    a plain string, a ``CallToolResult`` (``.content`` block list), a block
    list (``[{'type': 'text', 'text': …}, …]`` or its object form), or a
    ToolMessage-like wrapper. Every text block is unwrapped and joined; the
    caller's ``json.dumps`` fallback stays for non-text payloads.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        parts = [p for p in (_block_text(item).strip() for item in result) if p]
        return "\n".join(parts) if parts else None
    if isinstance(result, dict):
        return _block_text(result) or None
    content = getattr(result, "content", None)  # CallToolResult / ToolMessage
    if content is not None:
        return _result_text(content)
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return None


async def _call_tool(tool: Any, args: Dict[str, Any]) -> str:
    """One bounded MCP tool call: manager timeout + result cap; text result.

    Never raises — errors surface as ``ERROR:`` strings the caller records.
    """
    from api.mcp.manager import tool_call_timeout, tool_result_max_chars

    name = getattr(tool, "name", "") or "mcp-tool"
    try:
        result = await asyncio.wait_for(
            tool.ainvoke(args), timeout=tool_call_timeout()
        )
    except asyncio.TimeoutError:
        logger.warning("database docgen: MCP tool %r timed out; skipped", name)
        return f"ERROR: MCP tool {name!r} timed out and was aborted."
    except Exception as e:  # pragma: no cover - depends on live server
        logger.warning("database docgen: MCP tool %r failed: %s", name, e)
        return f"ERROR: MCP tool {name!r} failed ({type(e).__name__})."
    text = _result_text(result)
    if text is None:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:  # pragma: no cover - defensive
            text = str(result)
    limit = tool_result_max_chars()
    if len(text) > limit:
        text = text[:limit] + f"\n…[truncated: {len(text)} chars total]"
    return text


# --------------------------------------------------------------------------- #
# Preset MCP adapters (dbhub / oracle-mcp-server)
# --------------------------------------------------------------------------- #
class _AdaptedTool:
    """Synthetic role tool: canonical (schema, table) surface over a real tool.

    ``_introspect`` drives tools through ``_build_tool_args`` (name-token
    mapping over ``tool.args``) + ``tool.ainvoke``; an adapter declares
    CANONICAL arg names so the generic mapper fills them, then translates
    the call into the underlying tool's real payload inside ``ainvoke``.
    """

    def __init__(self, name: str, description: str, declared_args: Dict[str, Any], invoke):
        self.name = name
        self.description = description
        self.args = dict(declared_args)
        self._invoke = invoke

    async def ainvoke(self, tool_args: Dict[str, Any]) -> Any:
        return await self._invoke(dict(tool_args or {}))


#: dbhub ``search_objects`` result cap (server max is 1000).
_SEARCH_OBJECTS_LIMIT = 1000


def _render_search_full(text: str) -> str:
    """Render dbhub ``search_objects`` ``detail_level=full`` JSON as text.

    Keeps the evidence readable inside the per-table definition block:
    a header line, a column table and index lines. Any parse failure
    returns the raw text unchanged (best-effort by design).
    """
    try:
        data = json.loads((text or "").strip())
    except (ValueError, TypeError):
        return text or ""
    rows = _unwrap_name_collection(data, "results", "tables", "rows")
    if not isinstance(rows, list) or not rows:
        return text or ""
    out: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        header = str(row.get("name") or "?")
        if row.get("schema"):
            header += f" ({row['schema']})"
        meta = []
        if isinstance(row.get("column_count"), int):
            meta.append(f"{row['column_count']} columns")
        if row.get("row_count") is not None:
            meta.append(f"~{row['row_count']} rows")
        out.append(f"table {header}" + (f" — {', '.join(meta)}" if meta else ""))
        if row.get("comment"):
            out.append(f"comment: {row['comment']}")
        columns = row.get("columns")
        if isinstance(columns, list) and columns:
            out.append("| column | type | null | default |")
            out.append("| --- | --- | --- | --- |")
            for c in columns:
                if not isinstance(c, dict):
                    continue
                out.append("| {} | {} | {} | {} |".format(
                    c.get("name") or "?",
                    c.get("type") or "?",
                    "YES" if c.get("nullable") else "NO",
                    "-" if c.get("default") is None else str(c.get("default")),
                ))
                if c.get("description"):
                    out.append(f"  - comment: {c['description']}")
        indexes = row.get("indexes")
        if isinstance(indexes, list) and indexes:
            for i in indexes:
                if not isinstance(i, dict):
                    continue
                cols_raw = i.get("columns")
                if isinstance(cols_raw, str):
                    # Postgres array literal ("{id}") — strip the braces.
                    cols_i = cols_raw.strip("{}")
                else:
                    cols_i = ", ".join(str(x) for x in (cols_raw or []))
                flags = "UNIQUE" if i.get("unique") else ""
                if i.get("primary"):
                    flags = (flags + " PRIMARY").strip()
                out.append(
                    f"index {i.get('name') or '?'} ({cols_i})"
                    + (f" {flags}" if flags else "")
                )
        out.append("")
    return "\n".join(out).strip() if out else (text or "")


def _build_dbhub_roles(tools: List[Any]) -> Optional[Dict[str, List[Any]]]:
    """Roles over dbhub's ``search_objects`` (works for ALL five engines).

    ``search_objects`` is dbhub's unified object browser (schemas / tables /
    columns with pattern matching and detail levels) — a much better
    introspection surface than ``execute_sql`` wrappers: no engine-specific
    SQL, no injection surface, uniform JSON output. Returns ``None`` when
    the tool list has no ``search_objects`` (not a dbhub surface).
    """
    search = next(
        (t for t in tools if (getattr(t, "name", "") or "").lower() == "search_objects"),
        None,
    )
    if search is None:
        return None

    async def _call(payload: Dict[str, Any]) -> str:
        return await _call_tool(search, payload)

    async def _list_schemas(_args: Dict[str, Any]) -> str:
        # SQLite has no schemas — an empty listing is a normal outcome there.
        return await _call({
            "object_type": "schema",
            "detail_level": "names",
            "limit": _SEARCH_OBJECTS_LIMIT,
        })

    async def _list_tables(args: Dict[str, Any]) -> str:
        payload: Dict[str, Any] = {
            "object_type": "table",
            "detail_level": "names",
            "limit": _SEARCH_OBJECTS_LIMIT,
        }
        if args.get("schema"):
            payload["schema"] = args["schema"]
        return await _call(payload)

    async def _describe(args: Dict[str, Any]) -> str:
        payload: Dict[str, Any] = {
            "object_type": "table",
            "pattern": args.get("table") or "%",
            "detail_level": "full",
            "limit": 5,
        }
        if args.get("schema"):
            payload["schema"] = args["schema"]
        return _render_search_full(await _call(payload))

    return {
        "schemas": [_AdaptedTool(
            "search_objects[schemas]",
            "List database schemas via dbhub search_objects",
            {}, _list_schemas,
        )],
        "tables": [_AdaptedTool(
            "search_objects[tables]",
            "List tables via dbhub search_objects",
            {"schema": {"type": "string"}}, _list_tables,
        )],
        "describe": [_AdaptedTool(
            "search_objects[describe]",
            "Full table structure via dbhub search_objects",
            {"schema": {"type": "string"}, "table": {"type": "string"}},
            _describe,
        )],
        "ddl": [],
    }


def _oracle_arg_tokens(tool: Any) -> Dict[str, str]:
    """Map semantic tokens onto the oracle tool's declared argument names."""
    tokens: Dict[str, str] = {}
    for arg in _tool_arg_names(tool):
        low = arg.lower()
        if "pattern" in low:
            tokens.setdefault("pattern", arg)
        elif "table" in low or low in ("name", "object_name"):
            tokens.setdefault("table", arg)
        elif "schema" in low or "owner" in low:
            tokens.setdefault("schema", arg)
    return tokens


async def _call_oracle(
    tool: Any,
    *,
    pattern: Optional[str] = None,
    table: Optional[str] = None,
    schema: Optional[str] = None,
) -> str:
    """One oracle-mcp-server call with runtime-mapped argument names."""
    tokens = _oracle_arg_tokens(tool)
    payload: Dict[str, Any] = {}
    if pattern is not None:
        if "pattern" in tokens:
            payload[tokens["pattern"]] = pattern
        elif "table" in tokens:
            payload[tokens["table"]] = pattern
        else:
            declared = _tool_arg_names(tool)
            if declared:
                payload[declared[0]] = pattern
    if table is not None and "table" in tokens:
        payload[tokens["table"]] = table
    if schema is not None and "schema" in tokens:
        payload[tokens["schema"]] = schema
    return await _call_tool(tool, payload)


def _build_oracle_roles(tools: List[Any]) -> Optional[Dict[str, List[Any]]]:
    """Roles over oracle-mcp-server's schema tools (single connected schema).

    The server caches the connected schema (``TARGET_SCHEMA`` or the user's
    schema), so no schemas role is exposed — the walk runs in the default
    scope. Table listing prefers ``search_tables_schema`` (pattern ``%``);
    per-table definitions use ``get_table_schema`` (falling back to an exact
    ``search_tables_schema`` lookup). Returns ``None`` when none of the three
    known tools is present.
    """
    by_name = {(getattr(t, "name", "") or "").lower(): t for t in tools}
    search = by_name.get("search_tables_schema")
    lookup = by_name.get("get_table_schema")
    multi = by_name.get("get_tables_schema")
    if search is None and lookup is None and multi is None:
        return None
    lister = search if search is not None else multi

    async def _list_tables(args: Dict[str, Any]) -> str:
        if search is not None:
            return await _call_oracle(search, pattern="%")
        return await _call_oracle(multi)  # bulk dump; names parsed best-effort

    async def _describe(args: Dict[str, Any]) -> str:
        table = args.get("table")
        if lookup is not None:
            return await _call_oracle(lookup, table=table)
        if search is not None:
            return await _call_oracle(search, pattern=table)
        return "ERROR: no per-table oracle schema tool available"

    lister_name = getattr(lister, "name", "oracle_tool")
    describer_name = getattr(
        lookup if lookup is not None else search, "name", "oracle_tool"
    )
    return {
        "schemas": [],
        "tables": [_AdaptedTool(
            f"{lister_name}[tables]",
            "List oracle tables via oracle-mcp-server",
            {}, _list_tables,
        )],
        "describe": [_AdaptedTool(
            f"{describer_name}[describe]",
            "Table schema via oracle-mcp-server",
            {"table": {"type": "string"}}, _describe,
        )],
        "ddl": [],
    }


def preset_adapter_roles(
    tools: List[Any], db_type: Optional[str] = None
) -> Optional[Dict[str, List[Any]]]:
    """Roles for KNOWN preset tool surfaces (dbhub / oracle-mcp-server).

    Returns ``None`` when the tool list matches no known surface — the caller
    then falls back to the generic name heuristics. ``db_type`` (the preset
    flow's engine key) is accepted for future per-engine dispatch but the
    adapters key off ACTUAL tool names, so a manually registered dbhub or
    oracle-mcp-server works identically to the preset flow. Oracle wins over
    dbhub when both surfaces are bound (the walk documents one surface).
    """
    if not tools:
        return None
    roles = _build_oracle_roles(tools)
    if roles is not None:
        return roles
    return _build_dbhub_roles(tools)


# --------------------------------------------------------------------------- #
# Result parsing (JSON / dict shapes / quoted-string fallback)
# --------------------------------------------------------------------------- #
_NAME_KEYS = ("name", "schema_name", "table_name", "tableName", "object_name", "qualname")


def _unwrap_name_collection(data: Any, *collection_keys: str) -> Any:
    """Drill into result envelopes down to the actual name collection.

    Follows explicit collection keys (``tables`` / ``results`` / …) at any
    depth and the dbhub envelope ``{"success": true, "data": {…}}``; a dict
    holding exactly ONE list (e.g. ``{"public": ["t1", …]}`` keyed by
    schema) unwraps to that list. Bounded drill — anything else is returned
    unchanged.
    """
    for _ in range(3):
        if not isinstance(data, dict):
            return data
        for key in collection_keys:
            value = data.get(key)
            if isinstance(value, list):
                return value
        inner = data.get("data")
        if isinstance(inner, dict):
            data = inner
            continue
        lists = [v for v in data.values() if isinstance(v, list)]
        return lists[0] if len(lists) == 1 else data
    return data


def _parse_names(text: str, *collection_keys: str) -> List[str]:
    """Extract a de-duplicated name list from a tool result (best-effort)."""
    if not text or text.startswith("ERROR:"):
        return []
    data: Any = None
    try:
        data = json.loads(text.strip())
    except (ValueError, TypeError):
        pass
    if isinstance(data, dict):
        data = _unwrap_name_collection(data, *collection_keys)
    names: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                for key in _NAME_KEYS:
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        names.append(value)
                        break
    elif isinstance(data, str):
        names.append(data)
    names = [n.strip() for n in names if n and n.strip()]
    if not names:
        # Fallback: quoted strings in the raw text (capped).
        names = re.findall(r'"([^"\n]{1,128})"', text)[:100]
    seen: set = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# --------------------------------------------------------------------------- #
# Introspection walk: schemas → tables → per-table definitions
# --------------------------------------------------------------------------- #
async def _introspect(roles: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Run the deterministic introspection walk over classified tools.

    Returns ``{"schemas": [...], "tables": {name: {...}}, "tools_used": {...}}``.
    Raises ValueError when the surface cannot produce a schema at all (no
    table-listing tool, or every listing call failed) — an honest failure
    instead of empty docs.
    """
    tools_used: Dict[str, str] = {}
    schema_names: List[str] = []
    calls = 0

    async def _bounded_call(tool: Any, args: Dict[str, Any]) -> str:
        """Call counter around :func:`_call_tool` (hard cap, review #4)."""
        nonlocal calls
        if calls >= MAX_TOOL_CALLS:
            return (
                f"ERROR: introspection tool-call budget ({MAX_TOOL_CALLS}) "
                "exceeded; call skipped."
            )
        calls += 1
        return await _call_tool(tool, args)

    schema_tool = roles["schemas"][0] if roles["schemas"] else None
    if schema_tool is not None:
        tools_used["schemas"] = getattr(schema_tool, "name", "")
        raw = await _bounded_call(schema_tool, _build_tool_args(schema_tool))
        schema_names = _parse_names(raw, "schemas", "databases")[:MAX_SCHEMAS]

    table_tool = roles["tables"][0] if roles["tables"] else None
    if table_tool is None:
        raise ValueError(
            "No table-listing MCP tool found (expected a tool named like "
            "'list_tables'/'search_tables'); cannot introspect the database."
        )
    tools_used["tables"] = getattr(table_tool, "name", "")

    describer = roles["describe"][0] if roles["describe"] else None
    ddl_tool = roles["ddl"][0] if roles["ddl"] else None
    if describer is not None:
        tools_used["describe"] = getattr(describer, "name", "")
    if ddl_tool is not None:
        tools_used["ddl"] = getattr(ddl_tool, "name", "")

    scopes: List[Optional[str]] = schema_names or [None]
    tables: Dict[str, Any] = {}
    errors: List[str] = []
    for schema in scopes:
        raw = await _bounded_call(
            table_tool, _build_tool_args(table_tool, schema=schema)
        )
        table_names = _parse_names(raw, "tables", "results", "rows")[:MAX_TABLES_PER_SCHEMA]
        if not table_names:
            errors.append(f"no tables listed for schema {schema or '(default)'}")
            continue
        for table_name in table_names:
            full = f"{schema}.{table_name}" if schema else table_name
            definition = ""
            if describer is not None:
                definition = await _bounded_call(
                    describer,
                    _build_tool_args(describer, schema=schema, table=table_name),
                )
            if (not definition or definition.startswith("ERROR:")) and ddl_tool is not None:
                definition = await _bounded_call(
                    ddl_tool,
                    _build_tool_args(ddl_tool, schema=schema, table=table_name),
                )
            tables[full] = {
                "schema": schema,
                "table": table_name,
                "definition": _cap(definition or "", MAX_DEFINITION_CHARS),
            }
    if not tables:
        detail = "; ".join(errors[:3]) or "unknown reason"
        raise ValueError(
            f"Database introspection produced no tables ({detail}); "
            "check the MCP server connection and tool arguments."
        )
    return {"schemas": schema_names, "tables": tables, "tools_used": tools_used}


# --------------------------------------------------------------------------- #
# Deterministic skeleton
# --------------------------------------------------------------------------- #
# How many table blocks stay visible in the Tables page; the rest go under
# a <details> disclosure (fold, «не резать, а прятать") — a big schema can
# carry thousands of tables and no analyst reads them in a row.
_TABLES_VISIBLE = 12


def _table_block(full: str, info_t: Dict[str, Any]) -> List[str]:
    """Markdown block for ONE table (heading + DDL + blank separator)."""
    block: List[str] = [f"### `{full}`", ""]
    definition = (info_t.get("definition") or "").strip()
    if definition and not definition.startswith("ERROR:"):
        block.append("```sql")
        block.append(definition)
        block.append("```")
    elif definition.startswith("ERROR:"):
        block.append(f"_{definition}_")
    else:
        block.append("_(no definition available from MCP tools)_")
    block.append("")
    return block


def _render_skeleton(entity: Any, info: Dict[str, Any]) -> Tuple[str, str, str]:
    """Render (overview_md, schema_md, tables_md) from the introspected schema."""
    name = getattr(entity, "name", None) or "Database"
    dsn_masked = getattr(entity, "dsn_masked", None) or ""
    schemas = info.get("schemas") or []
    tables: Dict[str, Any] = info.get("tables") or {}

    overview: List[str] = [f"# Database: {name}", ""]
    if dsn_masked:
        overview.append(f"**Connection (masked):** `{dsn_masked}`")
        overview.append("")
    overview.append(f"**Schemas:** {', '.join(schemas) if schemas else '(default)'}")
    overview.append(f"**Tables introspected:** {len(tables)}")
    overview.append("")

    schema_md: List[str] = ["## Schemas", ""]
    if schemas:
        for s in schemas:
            count = sum(1 for t in tables.values() if t.get("schema") == s)
            schema_md.append(f"- `{s}` — {count} table(s)")
    else:
        schema_md.append("- Single default schema (no schema namespace).")
    schema_md.append("")

    # Fact fold: first _TABLES_VISIBLE tables (alphabetical) stay visible,
    # the rest go under ONE disclosure in the same order — every table is
    # still on the page, none is cut.
    table_blocks = [_table_block(full, tables[full]) for full in sorted(tables.keys())]
    visible, hidden = rank_split(table_blocks, key=lambda b: 0, keep=_TABLES_VISIBLE)
    tables_md: List[str] = ["## Tables", ""]
    for block in visible:
        tables_md.extend(block)
    if hidden:
        hidden_lines = [line for block in hidden for line in block]
        tables_md.extend(fold("Remaining tables", hidden_lines, count=len(hidden)))
    return "\n".join(overview), "\n".join(schema_md), "\n".join(tables_md)


def _schema_dump(info: Dict[str, Any]) -> str:
    """Raw evidence blob for the LLM/judge (JSON, capped)."""
    try:
        dump = json.dumps(info, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        dump = str(info)
    return _cap(dump, MAX_SCHEMA_DUMP_CHARS)


# --------------------------------------------------------------------------- #
# Enrichment prompt
# --------------------------------------------------------------------------- #
_DATABASE_DOC_FALLBACK = (
    "You are a database documentation expert. Document the database "
    "`{database_name}` for engineers, based EXCLUSIVELY on the schema "
    "evidence collected from its introspection tools below.\n\n"
    "<masked_connection>\n{dsn_masked}\n</masked_connection>\n\n"
    "<deterministic_skeleton>\n{skeleton}\n</deterministic_skeleton>\n\n"
    "<schema_evidence>\n{schema_dump}\n</schema_evidence>\n\n"
    "Produce complete, well-structured Markdown documentation covering: an "
    "overview of the database's purpose, the schema layout, every table with "
    "its columns and keys, inferred relationships between tables, and notable "
    "design decisions (indexes, constraints, naming conventions). Do NOT "
    "invent tables, columns, or constraints that are not present in the "
    "evidence; mark inferences explicitly as inferences. Write the "
    "documentation in {language_name} (technical terms in English). Your "
    "final message must contain ONLY the finished document."
)


def build_database_doc_prompt(
    *,
    database_name: str,
    dsn_masked: str,
    skeleton: str,
    schema_dump: str,
    language: str = "ru",
) -> str:
    """Build the enrichment prompt; ``language`` selects the OUTPUT language.

    Mirrors the spec flow: the ``{language_name}`` placeholder is substituted
    per-request, so the UI's ``language`` request field ("en"/"ru") actually
    reaches the model instead of being pinned to the config default
    (review #4 LOW: the parameter used to be accepted and dropped).
    """
    template = load_prompt_file("database_doc.md", _DATABASE_DOC_FALLBACK)
    for var, value in (
        ("database_name", database_name),
        ("dsn_masked", dsn_masked or "(not provided)"),
        ("skeleton", skeleton or "(unavailable)"),
        ("schema_dump", schema_dump or "(unavailable)"),
        ("language_name", LANGUAGE_NAMES.get(language, language)),
    ):
        template = template.replace("{" + var + "}", str(value))
    return template


# --------------------------------------------------------------------------- #
# Page assembly + provenance
# --------------------------------------------------------------------------- #
_DATABASE_PAGE_TITLES = {
    "overview": "Overview",
    "schema": "Schemas",
    "tables": "Tables",
    "documentation": "Documentation (AI)",
}


def _database_pages(
    sections: Dict[str, str], provenance: Dict[str, Any]
) -> Dict[str, Any]:
    pages: Dict[str, Any] = {}
    for sid, content in sections.items():
        page_id = f"page_{sid}"
        page: Dict[str, Any] = {
            "id": page_id,
            "title": _DATABASE_PAGE_TITLES.get(sid, sid),
            "content": content or "",
            "filePaths": [],
            "importance": "high" if sid in ("overview", "tables") else "medium",
            "relatedPages": [],
        }
        if provenance:
            page["provenance"] = dict(provenance)
        pages[page_id] = page
    return pages


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
async def generate_database_docs(
    entity: Any,
    product: Any,
    model: Optional[str] = None,
    language: str = "ru",
    progress: Optional[Any] = None,
) -> str:
    """Reverse-engineer a database via MCP tools and document it.

    Raises ValueError (→ a FAILED docgen job, never empty docs) when the MCP
    surface is unusable: no bound servers/tools, a dead pinned server, no
    table-listing tool, or an introspection walk that produced nothing —
    except when a fresh disk-cached introspection payload for the same MCP
    surface is available (2.3b): the walk is the expensive stage (a schema of
    hundreds of MB takes tens of minutes), so reruns/repairs after an
    interruption reuse the cache and do not re-validate the live surface.
    """
    product_id = (
        getattr(product, "id", None) or getattr(product, "product_id", None) or ""
    )
    name = getattr(entity, "name", None) or "Database"
    dsn_masked = getattr(entity, "dsn_masked", None) or ""

    emit_progress(progress, phase="planning")
    # 1-2) Introspection, CACHE-FIRST (2.3b). The walk over a huge schema is
    # the most expensive, least reproducible stage of the run; when a fresh
    # cached payload for the same MCP surface exists, the live surface is not
    # touched at all (the tool-resolution ValueErrors below are deliberately
    # skipped on a hit — that is what makes a repair-after-crash resilient).
    cache_key: Optional[str] = None
    try:
        cache_key = _introspection_cache_key(entity, product_id)
    except Exception as e:  # pragma: no cover - key computation is never fatal
        logger.debug("introspection cache key unavailable: %s", e)
    info: Optional[Dict[str, Any]] = None
    cache_state = "miss"
    if cache_key is not None:
        info = load_introspection_cache(cache_key)
        if info is not None:
            cache_state = "hit"

    if info is None:
        # 1) MCP tools (pinned server or all bound enabled servers).
        tools = await _resolve_mcp_tools(entity, product_id)
        if not tools:
            raise ValueError(
                "No MCP tools are available for this product; bind a database "
                "MCP server before generating documentation."
            )

        # 2) Classify + deterministic introspection walk. Known preset
        # surfaces (dbhub / oracle-mcp-server) get dedicated adapters before
        # the generic name heuristics — the heuristics misclassify their
        # tools (``search_objects`` matches nothing, ``get_tables_schema``
        # would be misused as a table lister).
        roles = preset_adapter_roles(tools, db_type=getattr(entity, "db_type", None))
        if roles is None:
            roles = _classify_introspection_tools(tools)
        if not (roles["tables"] or roles["describe"] or roles["ddl"]):
            available = ", ".join(sorted(filter(None, (
                getattr(t, "name", "") for t in tools
            )))) or "(none)"
            raise ValueError(
                "None of the product's MCP tools look like database "
                "introspection tools (list_schemas/list_tables/describe_table/"
                f"get_table_ddl…). Available tools: {available}"
            )
        # Overall wall-clock budget for the whole walk (review #4): per-call
        # timeouts bound ONE call, this bounds the walk; a timeout is an honest
        # job failure (ValueError), never an empty-docs success.
        emit_progress(
            progress, phase="sections", sections_total=1,
            current_section="introspection",
        )
        t_introspect = time.monotonic()
        try:
            info = await asyncio.wait_for(
                _introspect(roles), timeout=INTROSPECTION_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            raise ValueError(
                "Database introspection exceeded the overall time budget of "
                f"{int(INTROSPECTION_TIMEOUT_SECONDS)}s; the MCP surface is too "
                "slow — pin a faster server or narrow the scope and retry."
            )
        emit_progress(
            progress, section_done="introspection",
            section_seconds=time.monotonic() - t_introspect,
        )
        if cache_key is not None:
            store_introspection_cache(cache_key, info)
    else:
        logger.info(
            "Database introspection served from the disk cache (key=%s…): "
            "the MCP surface is not re-walked this run.",
            cache_key[:12],
        )
        emit_progress(
            progress, phase="sections", sections_total=1,
            section_done="introspection", section_seconds=0.0,
        )

    # 3) Deterministic skeleton (overview / schemas / tables).
    overview_md, schema_md, tables_md = _render_skeleton(entity, info)
    skeleton = "\n\n".join(part for part in (overview_md, schema_md, tables_md) if part)

    # 4) LLM enrichment (standard docgen path; empty → skeleton).
    r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    model = model or r_model
    dump = _schema_dump(info)
    prompt = build_database_doc_prompt(
        database_name=name,
        dsn_masked=dsn_masked,
        skeleton=skeleton,
        schema_dump=dump,
        language=language,
    )
    enrich_source = "skeleton"
    docs = skeleton
    llm_text = await _llm_or_none(
        prompt, model, base_url=r_base_url, api_key=r_api_key
    )
    if llm_text:
        docs = llm_text
        enrich_source = "standard-llm"

    # 5) Guard: mermaid repair → secret masking → judge (flags only).
    emit_progress(progress, phase="verifying")
    repair_llm = None
    try:
        repair_llm = _make_repair_llm(model, base_url=r_base_url, api_key=r_api_key)
        docs, _mstats = await run_repair_loop(docs, repair_llm)
    except Exception as e:  # pragma: no cover - verifier must never break gen
        logger.warning("Mermaid repair loop failed for database doc: %s", e)
    finally:
        await _close_owned_llm(repair_llm)

    docs, findings = mask_secrets(docs or "")
    if findings:
        logger.warning(
            "database docgen guard: masked %d secret-like value(s) (%s)",
            len(findings), ", ".join(sorted(set(findings))),
        )

    # 3.1: corroborate filter — model-generated sentences naming identifiers
    # that are NOT in the introspected schema (invented tables/columns/
    # enums/functions) are dropped BEFORE the judge, so the judged text and
    # the persisted text agree. Skeleton output is deterministic evidence
    # and is never filtered. Fail-open per the module contract.
    corroborate_removed: List[str] = []
    if enrich_source == "standard-llm":
        try:
            grounding = grounding_from_introspection(info)
            if grounding:
                docs, corrob = filter_ungrounded_prose(docs, grounding)
                if corrob.emptied:
                    logger.warning(
                        "database corroborate filter would empty the docs; "
                        "original kept"
                    )
                elif corrob.touched:
                    corroborate_removed = list(corrob.removed_identifiers)
                    logger.warning(
                        "database docgen corroborate: dropped %d sentence(s) "
                        "naming ungrounded identifier(s): %s",
                        corrob.sentences_removed,
                        ", ".join(corrob.removed_identifiers[:15]),
                    )
        except Exception as e:  # pragma: no cover - guard must never break gen
            logger.warning("Database corroborate filter failed: %s", e)

    judge_verdict = None
    if judge_enabled() and enrich_source == "standard-llm":
        try:
            judge_verdict = await judge_section(
                "database", docs, skeleton + "\n\n" + dump, model=model
            )
            if judge_verdict.verdict != "consistent" and judge_verdict.issues:
                logger.warning(
                    "database docgen judge: %s — %s",
                    judge_verdict.verdict,
                    "; ".join(judge_verdict.issues[:3]),
                )
        except Exception as e:  # pragma: no cover - judge must never break gen
            logger.warning("database docgen judge failed: %s", e)

    provenance = build_section_provenance(
        "database",
        model=model,
        prompt_file="database_doc.md",
        source_files=[],
        fingerprint=None,
        citations={"resolved": [], "unresolved": []},
        judge=judge_verdict,
        regen="legacy-fallback" if enrich_source == "standard-llm" else "skeleton",
        secrets_masked=len(findings),
    )
    provenance["generator"] = enrich_source
    provenance["tools_used"] = dict(info.get("tools_used") or {})
    provenance["schema_fingerprint_source"] = "mcp_introspection"
    provenance["introspection_cache"] = cache_state
    if corroborate_removed:
        provenance["corroborate"] = {"removed": corroborate_removed}

    # 6) Persist (viewer contract) + background indexing.
    emit_progress(progress, phase="indexing")
    sections = {
        "overview": overview_md,
        "schema": schema_md,
        "tables": tables_md,
        "documentation": docs if enrich_source == "standard-llm" else "",
    }
    _persist_artifact(entity, docs, _database_pages(sections, provenance))
    _index_in_background(
        docs,
        _product_dataset(product),
        source_type="database",
        source_id=getattr(entity, "id", None),
    )
    return docs


__all__ = [
    "INTROSPECTION_TIMEOUT_SECONDS",
    "MAX_SCHEMAS",
    "MAX_TABLES_PER_SCHEMA",
    "MAX_TOOL_CALLS",
    "build_database_doc_prompt",
    "classify_introspection_tools",
    "generate_database_docs",
    "preset_adapter_roles",
]
