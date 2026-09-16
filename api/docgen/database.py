"""Database reverse-engineering flow (Wave E, DB-RE restructure).

Documents a database artifact through the product's MCP servers into a
ROOT-PAGE + PER-ENTITY-SUBPAGE tree (viewer contract: ``pages`` dict with
``parent`` nesting):

- ``Overview`` — deterministic facts (masked DSN, schemas, object counts)
  + one LLM call (Обзор / Раскладка схем / Замечания по дизайну).
- ``Tables`` (root) — short descriptions, ``Relationships`` (FK evidence)
  and a Mermaid ``erDiagram`` derived from the FK graph; every table gets a
  CHILD subpage (structure / indexes / constraints / relations / triggers /
  DDL), ranked by FK-degree → column count and capped by
  ``DB_DOCGEN_MAX_SUBPAGES`` (surplus stays folded on the root, never cut).
- ``Views`` / ``Triggers`` / ``Procedures & Functions`` / ``Sequences`` /
  ``Types`` — category roots appear ONLY with introspection evidence, with
  per-object child subpages inside the same cap.

Stages:

1. **Introspection (deterministic, disk-cached)** — resolves the product's
   MCP tools (pinned server or all bound enabled). KNOWN preset surfaces
   get dedicated adapters BEFORE the generic name heuristics: dbhub's
   ``search_objects`` (schemas/tables/views/routines + full per-table
   detail, uniform across PostgreSQL/MySQL/MariaDB/SQL Server/SQLite) and
   oracle-mcp-server's ``search_tables_schema``/``get_table_schema`` +
   ``get_pl_sql_objects``/``get_object_source``/constraints/indexes/types.
   Oracle engines with a sql tool take a cross-schema BULK walk first
   (ALL_* owners → columns/comments per schema) — the pinned server caches
   one schema, which misses the enterprise multi-schema layout.
   When the engine is detectable (``db_type``, masked DSN scheme, or the
   oracle tool surface) AND a read-only ``sql`` tool exists, a fixed pack
   of SELECT-only catalog queries (``information_schema``/``pg_catalog``
   for PostgreSQL, ``ALL_*`` views for Oracle) fills the gaps generic
   tools leave: FK edges, indexes, triggers, sequences, materialized
   views, composite types and routine sources. Every call is bounded by
   the manager's timeouts/caps; the RESULT is cached on disk
   (``api.docgen.introspection_cache``) keyed by the MCP surface identity
   + walk budgets, so reruns/repairs never re-walk the live database.
2. **Enrich (batched, admin-tunable)** — ONE overview call, then table
   descriptions as strict-JSON BATCHES of ``DB_DOCGEN_ENRICH_BATCH``
   tables (not per page: 61 tables cost 2 calls, not 61; an Oracle
   monolith costs dozens, not thousands), category descriptions one call
   each, and — only when introspection found NO foreign keys — an
   explicitly-marked inference pass. Subpages stay deterministic; every
   LLM sentence is corroborated against the introspected identifiers.
   Product knowledge from the semantic memory (codebase docs, specs,
   Confluence) is attached as SUPPLEMENTARY context. LLM failure degrades
   to the deterministic skeleton — only a dead/unusable MCP surface is a
   hard error (an honest failed job, never empty docs).
3. **Guard (verification)** — mermaid repair loop on pages carrying
   diagrams, secret masking on EVERY page (the persisted docs never carry
   DSN passwords or tokens), the corroborate filter on the LLM overview,
   an optional LLM judge (model-generated overview only; flags, never
   blocks) and per-page provenance (tools used, fingerprint, caps,
   masking/judge verdicts).
4. **Persist + index** — ``generated_docs`` + ``pages`` written onto the
   artifact and the assembled markdown indexed into the active memory
   backend with ``source_type="database"``. A compact ``db_context``
   digest (schemas + top tables by FK-degree) is stored in the Tables
   page provenance so ``product_database_context`` can feed the CODEBASE
   flow's brief — code docs grounded in the schema, DB docs grounded in
   the code (both verifiable).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from api.utils import setup_logging
from api.utils.llm_helpers import cap as _cap
from api.formats.mermaid import run_repair_loop
from api.prompts import LANGUAGE_NAMES, load_prompt_file
from api.docgen._common import (
    _carry_page_verify_flags,
    _check_cancel,
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
# Budgets (introspection walk + render/enrich knobs)
# --------------------------------------------------------------------------- #
#: Max schemas introspected per database (enterprise multitenant
#: databases carry hundreds; extras past the cap are skipped).
MAX_SCHEMAS = 200
#: Max tables introspected per schema (extra tables are listed by name only).
MAX_TABLES_PER_SCHEMA = 100
#: Max characters of one table definition kept in the doc/evidence.
MAX_DEFINITION_CHARS = 8_000
#: Max characters of the raw schema dump handed to the LLM.
MAX_SCHEMA_DUMP_CHARS = 120_000
#: Max characters of the product-knowledge context block inside prompts.
_MAX_PRODUCT_CONTEXT_CHARS = 8_000


def _env_float(name: str, default: float) -> float:
    """Import-time env parsing that never crashes on garbage values."""
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw)
    except ValueError:
        if raw:
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


#: Overall wall-clock budget for one introspection walk (DoS guard, review
# #4: per-call limits alone let a slow MCP surface occupy a docgen worker
# for hours). Env-overridable; default 30 minutes — a multi-schema Oracle
# monolith legitimately needs tens of paged catalog queries.
INTROSPECTION_TIMEOUT_SECONDS = _env_float("DB_INTROSPECTION_TIMEOUT_SECONDS", 1800.0)
#: Hard cap on MCP tool calls during one walk (schema listings + per-table
#: definitions + category probes combined) so huge schemas cannot loop
#: unboundedly.
MAX_TOOL_CALLS = 2_000


# Admin-tunable render/enrich knobs over the timeout registry (admin store >
# env > default; read per generation so an admin change applies to the next
# run without a restart). Module-level wrappers are the monkeypatch seams.
def _enrich_batch_size() -> int:
    """Tables per strict-JSON description batch."""
    from api.config.timeout import resolve_db_docgen_enrich_batch

    return resolve_db_docgen_enrich_batch()


def _subpage_cap() -> int:
    """Max entity subpages rendered per database (surplus stays folded)."""
    from api.config.timeout import resolve_db_docgen_max_subpages

    return resolve_db_docgen_max_subpages()


def _max_descriptions() -> int:
    """Max objects that get an LLM description per database run."""
    from api.config.timeout import resolve_db_docgen_max_descriptions

    return resolve_db_docgen_max_descriptions()


def _fk_evidence_tables() -> int:
    """Max tables probed for constraints/indexes/relations (walk budget)."""
    from api.config.timeout import resolve_db_fk_evidence_tables

    return resolve_db_fk_evidence_tables()


def _source_objects() -> int:
    """Max non-table objects with fetched source/definition (walk budget)."""
    from api.config.timeout import resolve_db_source_objects

    return resolve_db_source_objects()


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
    introspection payload, so it must invalidate cached entries. Only
    WALK-affecting knobs participate; the render/enrich knobs
    (batch size, subpage cap, description cap) do not reshape the walk."""
    return {
        "max_schemas": MAX_SCHEMAS,
        "max_tables_per_schema": MAX_TABLES_PER_SCHEMA,
        "max_definition_chars": MAX_DEFINITION_CHARS,
        "max_schema_dump_chars": MAX_SCHEMA_DUMP_CHARS,
        "max_tool_calls": MAX_TOOL_CALLS,
        "fk_evidence_tables": _fk_evidence_tables(),
        "source_objects": _source_objects(),
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
#: Every role the walk knows about. Preset adapters and the generic
#: classifier both return full dicts over these keys so the walk can index
#: any role without KeyError.
_ROLE_KEYS = (
    "schemas", "tables", "describe", "ddl",
    "indexes", "constraints", "relationships",
    "views", "routines", "triggers", "sequences", "types",
    "sql", "source",
)


def _empty_roles() -> Dict[str, List[Any]]:
    return {key: [] for key in _ROLE_KEYS}


def _classify_introspection_tools(tools: List[Any]) -> Dict[str, List[Any]]:
    """Split discovered MCP tools into introspection roles by name heuristics.

    Database MCP servers name their introspection tools differently
    (``list_schemas``/``get_schemas``, ``list_tables``/``search_tables``,
    ``describe_table``/``get_table_info``/``get_table_ddl`` …), so the roles
    are matched on name tokens rather than exact names. First match per tool
    wins; unrelated tools (query execution, health, …) are ignored.
    """
    roles = _empty_roles()
    for tool in tools or []:
        name = (getattr(tool, "name", "") or "").lower()
        if not name:
            continue
        has_table = "table" in name
        has_lister = (
            "list" in name or "search" in name or "show" in name
            or "get" in name
        )
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
        elif "related" in name:
            roles["relationships"].append(tool)
        elif has_table and "constraint" in name and has_lister:
            roles["constraints"].append(tool)
        elif has_table and "index" in name:
            roles["indexes"].append(tool)
        elif "view" in name and has_lister:
            roles["views"].append(tool)
        elif (
            ("procedure" in name or "function" in name or "routine" in name)
            and has_lister
        ):
            roles["routines"].append(tool)
        elif "trigger" in name and has_lister:
            roles["triggers"].append(tool)
        elif "sequence" in name and has_lister:
            roles["sequences"].append(tool)
        elif "type" in name and not has_table:
            roles["types"].append(tool)
        elif "sql" in name and ("execute" in name or "run" in name or "query" in name):
            roles["sql"].append(tool)
        elif has_table and has_lister:
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


def _sql_args(
    tool: Any, query: str, max_rows: Optional[int] = None
) -> Dict[str, Any]:
    """Map a SQL string (and an optional row cap) onto the sql tool's args."""
    out: Dict[str, Any] = {}
    sql_arg: Optional[str] = None
    for arg in _tool_arg_names(tool):
        low = arg.lower()
        if low in ("sql", "query", "statement", "q"):
            sql_arg = sql_arg or arg
        elif max_rows is not None and ("max_row" in low or low == "limit"):
            out[arg] = max_rows
    if sql_arg is None:
        declared = _tool_arg_names(tool)
        sql_arg = declared[0] if declared else "sql"
    out[sql_arg] = query
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
# Read-only SQL guard + catalog packs (PostgreSQL / Oracle)
# --------------------------------------------------------------------------- #
# The packs are CONSTANT select-only statements — no identifier is ever
# interpolated, so there is no injection surface to validate. The guard
# below is defense in depth: it rejects anything the pack constants are
# not (and anything a future editor might paste in).
_READONLY_FORBIDDEN_RE = re.compile(
    r"\b("
    r"insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+|alter\s+|"
    r"create\s+(or\s+replace\s+)?(table|index|view|trigger|procedure|"
    r"function|sequence|type|materialized)|grant\s|revoke\s|truncate\s|"
    r"merge\s+into|call\s|exec(ute)?\s|commit\b|rollback\b|savepoint\b|"
    r"lock\s+(table|in)|for\s+update"
    r")",
    re.IGNORECASE,
)


def _assert_readonly_sql(query: str) -> bool:
    """True only for single read-only SELECT/WITH statements."""
    q = (query or "").strip()
    if not q:
        return False
    if not re.match(r"(?is)^(with|select)\b", q):
        return False
    if _READONLY_FORBIDDEN_RE.search(q):
        return False
    body = q.rstrip()
    if body.endswith(";"):
        body = body[:-1]
    return ";" not in body  # one statement only


_PG_SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")
_ORACLE_SYSTEM_OWNERS = (
    "SYS", "SYSTEM", "XDB", "MDSYS", "CTXSYS", "ORDSYS", "OUTLN", "DBSNMP",
    "WMSYS", "EXFSYS", "OLAPSYS", "ORDDATA", "LBACSYS", "AUDSYS",
    "GSMADMIN_INTERNAL", "OJVMSYS", "DBSFWUSER", "DVSYS", "APPQOSSYS",
    "GSMCATUSER", "GSMUSER", "DIP", "ORACLE_OCM", "MDDATA",
    "REMOTE_SCHEDULER_AGENT", "SYSBACKUP", "SYSDG", "SYSKM", "SYSRAC",
    "DVF", "DV_OWNER", "DV_ACCTMGR",
    # Maintenance accounts present on legacy monoliths (Statspack, APEX,
    # Workspace, OEM, OWB…).
    "ANONYMOUS", "APEX_PUBLIC_USER", "FLOWS_FILES", "MGMT_VIEW", "SYSMAN",
    "OWBSYS", "OWBSYS_AUDIT", "WKSYS", "WK_TEST", "WKPROXY",
    "SI_INFORMTN_SCHEMA", "XS$NULL", "TSMSYS", "PERFSTAT",
)
#: APEX releases ship version-prefixed schemas (APEX_040200…) — a fixed
#: tuple cannot cover them.
_ORA_SYSTEM_OWNER_PREFIXES = ("APEX_", "FLOWS_")


def _pg_not_system(column: str) -> str:
    return f"{column} NOT IN (" + ", ".join(f"'{s}'" for s in _PG_SYSTEM_SCHEMAS) + ")"


def _ora_not_system(column: str) -> str:
    return f"{column} NOT IN (" + ", ".join(f"'{s}'" for s in _ORACLE_SYSTEM_OWNERS) + ")"


def _ora_user_owner(owner: str) -> bool:
    """True for USER Oracle owners — maintenance accounts are not docs.

    Case-insensitive; authoritative counterpart of the SQL-side ``NOT IN``
    (covers the version-prefixed APEX_/FLOWS_ schemas). An empty owner
    means "no owner info" and passes — there is nothing to filter on.
    """
    up = (owner or "").strip().upper()
    if not up:
        return True
    return up not in _ORACLE_SYSTEM_OWNERS and not up.startswith(
        _ORA_SYSTEM_OWNER_PREFIXES
    )


def _pg_user_schema(name: str) -> bool:
    """True for user-visible PG schemas (system namespaces are noise)."""
    low = (name or "").strip().lower()
    return bool(low) and not low.startswith("pg_") and low != "information_schema"


#: PostgreSQL catalog pack: FK edges, indexes, triggers, sequences,
#: materialized views, composite types and routine sources. Pure SELECTs.
_PG_SQL_PACK: Dict[str, str] = {
    "fk_edges": (
        "SELECT tc.table_schema AS table_schema, tc.table_name AS table_name, "
        "tc.constraint_name AS constraint_name, kcu.column_name AS column_name, "
        "ccu.table_schema AS foreign_schema, ccu.table_name AS foreign_table, "
        "ccu.column_name AS foreign_column "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "ON kcu.constraint_name = tc.constraint_name "
        "AND kcu.table_schema = tc.table_schema "
        "JOIN information_schema.constraint_column_usage ccu "
        "ON ccu.constraint_name = tc.constraint_name "
        "AND ccu.table_schema = tc.table_schema "
        "WHERE tc.constraint_type = 'FOREIGN KEY'"
    ),
    "indexes": (
        "SELECT schemaname AS table_schema, tablename AS table_name, "
        "indexname AS index_name, indexdef AS index_def "
        "FROM pg_indexes WHERE " + _pg_not_system("schemaname")
    ),
    "triggers": (
        "SELECT n.nspname AS schema_name, c.relname AS table_name, "
        "t.tgname AS trigger_name, p.proname AS function_name, "
        "pg_get_triggerdef(t.oid) AS definition "
        "FROM pg_trigger t "
        "JOIN pg_class c ON c.oid = t.tgrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_proc p ON p.oid = t.tgfoid "
        "WHERE NOT t.tgisinternal AND " + _pg_not_system("n.nspname") + " "
        # Extension-owned trigger functions (TimescaleDB, …) are not user code.
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d "
        "WHERE d.objid = p.oid AND d.deptype = 'e')"
    ),
    "sequences": (
        "SELECT schemaname AS schema_name, sequencename AS sequence_name, "
        "start_value, minimum_value, maximum_value, increment "
        "FROM pg_sequences WHERE " + _pg_not_system("schemaname")
    ),
    "matviews": (
        "SELECT schemaname AS schema_name, matviewname AS view_name, "
        "definition AS view_definition FROM pg_matviews "
        "WHERE " + _pg_not_system("schemaname")
    ),
    "types": (
        "SELECT n.nspname AS schema_name, t.typname AS type_name, "
        "string_agg(a.attname || ' ' || "
        "format_type(a.atttypid, a.atttypmod), ', ' ORDER BY a.attnum) "
        "AS attributes "
        "FROM pg_type t "
        "JOIN pg_namespace n ON n.oid = t.typnamespace "
        "JOIN pg_class c ON c.reltype = t.oid "
        "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 "
        "AND NOT a.attisdropped "
        "WHERE c.relkind = 'c' AND " + _pg_not_system("n.nspname") + " "
        # Extension-shipped composite types (pgvector …) are not user types.
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d "
        "WHERE d.objid = t.oid AND d.deptype = 'e') "
        "GROUP BY n.nspname, t.typname"
    ),
    "routines": (
        "SELECT n.nspname AS schema_name, p.proname AS routine_name, "
        "CASE p.prokind WHEN 'p' THEN 'PROCEDURE' ELSE 'FUNCTION' END AS kind, "
        "pg_get_functiondef(p.oid) AS source "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE " + _pg_not_system("n.nspname") + " "
        # Extension-shipped functions (pgvector lives in public) are noise:
        # only pg_depend membership separates them from user code.
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d "
        "WHERE d.objid = p.oid AND d.deptype = 'e')"
    ),
}

#: Page size for Oracle catalog queries. The manager hard-caps every tool
#: result (``MCP_TOOL_RESULT_MAX_CHARS``, 100k chars) and a rendered
#: pipe-table row is ~60-100 chars, so ~1_200 rows/page keeps each page
#: under the cap. The old single 5_000-row ask was silently chopped
#: mid-table by the cap and the parser lost everything after the cut.
_ORA_PAGE_ROWS = 1_200
#: Sanity ceiling on pages per query (72k rows) — bounds worst-case walk
#: time; anything wider is documentation noise anyway.
_ORA_MAX_PAGES = 60
#: Owner names interpolated into the bulk queries below are catalog-derived
#: (all_tables.owner), never user input; this charset regex additionally
#: excludes quotes/semicolons, so the interpolated literal is inert.
_ORA_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_$#]{1,128}$")

#: Oracle bulk walk over the ALL_* catalog: one owners query, then columns +
#: comments per owner (BIN$ = recyclebin leftovers, never documentation).
#: Aliases are QUOTED lowercase — Oracle uppercases unquoted aliases, and
#: the rendered pipe-table headers must match the parser's row keys.
#: Deliberate, guarded exception to the constants-only packs: ``{owner}`` is
#: filled from the catalog itself (regex above) and re-checked by the
#: read-only guard before every call.
#: NULL CELLS ARE FORBIDDEN in every selected expression: the pinned
#: oracle-mcp-server's formatter crashes on ANY None cell (its _escape
#: returns a bare string for None while the caller unpacks a 2-tuple), so
#: the whole result becomes "Unexpected error executing query: …" and
#: parses to zero rows. Hence NVL() wherever a catalog column is nullable
#: (data_default — a LONG on top of that — is simply not selected).
_ORA_OWNERS_QUERY = (
    'SELECT owner AS "owner" FROM all_tables WHERE ' + _ora_not_system("owner")
    + " GROUP BY owner ORDER BY owner"
)
_ORA_COLUMNS_QUERY = (
    'SELECT c.owner AS "table_schema", c.table_name AS "table_name", '
    'c.column_id AS "column_id", c.column_name AS "column_name", '
    'c.data_type AS "data_type", c.nullable AS "nullable", '
    'NVL(t.num_rows, 0) AS "num_rows" '
    "FROM all_tab_columns c "
    "JOIN all_tables t ON t.owner = c.owner AND t.table_name = c.table_name "
    "WHERE c.owner = '{owner}' AND t.table_name NOT LIKE 'BIN$%' "
    "ORDER BY c.table_name, c.column_id"
)
_ORA_COMMENTS_QUERY = (
    'SELECT table_name AS "table_name", comments AS "comments" '
    "FROM all_tab_comments "
    "WHERE owner = '{owner}' AND comments IS NOT NULL"
)

#: The pinned server's effective schema (TARGET_SCHEMA or the login user):
#: ``get_object_source`` resolves every object against it (DBMS_METADATA
#: has no cross-schema lookup there), so sources can only be fetched for
#: objects whose owner matches it.
_ORA_SESSION_SCHEMA_QUERY = (
    "SELECT SYS_CONTEXT('USERENV', 'SESSION_USER') AS \"owner\" FROM dual"
)

#: Oracle catalog pack over the ALL_* views (works with any role's grants).
#: Routine/trigger sources arrive LINE-based and are regrouped by the walk.
# Aliases are QUOTED lowercase so the rendered pipe-table headers match
# the parser's row keys (Oracle uppercases unquoted aliases).
_ORACLE_SQL_PACK: Dict[str, str] = {
    "fk_edges": (
        'SELECT ac.owner AS "table_schema", ac.table_name AS "table_name", '
        'a.constraint_name AS "constraint_name", ac.column_name AS "column_name", '
        'rc.table_name AS "foreign_table", rc.column_name AS "foreign_column" '
        "FROM all_constraints a "
        "JOIN all_cons_columns ac ON ac.owner = a.owner "
        "AND ac.constraint_name = a.constraint_name "
        "JOIN all_cons_columns rc ON rc.owner = a.r_owner "
        "AND rc.constraint_name = a.r_constraint_name "
        "AND rc.position = ac.position "
        "WHERE a.constraint_type = 'R' AND " + _ora_not_system("a.owner")
    ),
    "indexes": (
        'SELECT i.table_owner AS "table_schema", i.table_name AS "table_name", '
        'i.index_name AS "index_name", c.column_name AS "column_name", '
        'c.column_position AS "column_position", i.uniqueness AS "uniqueness" '
        "FROM all_indexes i "
        "JOIN all_ind_columns c ON c.index_owner = i.owner "
        "AND c.index_name = i.index_name "
        "WHERE " + _ora_not_system("i.table_owner") + " "
        "ORDER BY i.table_owner, i.table_name, i.index_name, c.column_position"
    ),
    "triggers": (
        'SELECT owner AS "schema_name", trigger_name AS "trigger_name", '
        'table_name AS "table_name", '
        'triggering_event AS "triggering_event", trigger_type AS "trigger_type", '
        'NVL(status, "?") AS "enabled" '
        "FROM all_triggers WHERE " + _ora_not_system("owner")
    ),
    "sequences": (
        'SELECT sequence_owner AS "schema_name", sequence_name AS "sequence_name", '
        'NVL(TO_CHAR(min_value), "-") AS "min_value", '
        'NVL(TO_CHAR(max_value), "-") AS "max_value", '
        'increment_by AS "increment_by", cycle_flag AS "cycle_flag" '
        "FROM all_sequences WHERE " + _ora_not_system("sequence_owner")
    ),
    # all_mviews.query is a LONG — NVL is illegal on LONG, so a NULL query
    # there crashes the pinned formatter and the matview pack degrades to
    # empty ("sql:matviews" unavailable; non-fatal by design).
    "matviews": (
        'SELECT owner AS "schema_name", mview_name AS "view_name", '
        'query AS "view_definition" FROM all_mviews '
        "WHERE " + _ora_not_system("owner")
    ),
    # Names only: all_views.TEXT is a LONG column (no SUBSTR/aggregation) —
    # definitions come from the capped per-object source fetch.
    "views": (
        'SELECT owner AS "schema_name", view_name AS "name" '
        "FROM all_views WHERE " + _ora_not_system("owner")
    ),
    "types": (
        'SELECT owner AS "schema_name", type_name AS "type_name", '
        'NVL(typecode, "?") AS "typecode" '
        "FROM all_types WHERE " + _ora_not_system("owner")
    ),
    "routines": (
        'SELECT owner AS "schema_name", name AS "object_name", '
        'type AS "object_type", line AS "line", '
        # Blank ALL_SOURCE lines can be NULL — NVL or the whole pack dies.
        'NVL(text, " ") AS "line_text" '
        "FROM all_source "
        "WHERE type IN ('FUNCTION', 'PROCEDURE', 'PACKAGE', "
        "'PACKAGE BODY', 'TRIGGER', 'TYPE', 'TYPE BODY') "
        "AND " + _ora_not_system("owner") + " "
        "ORDER BY owner, type, name, line"
    ),
}


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
    views / procedures / columns with pattern matching and detail levels) —
    a much better introspection surface than ``execute_sql`` wrappers: no
    engine-specific SQL, no injection surface, uniform JSON output. Views
    are listed with FULL detail (definitions included); routines as names.
    Returns ``None`` when the tool list has no ``search_objects``.
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

    async def _list_views(_args: Dict[str, Any]) -> str:
        # Full detail: the row carries the view definition (create stmt).
        return await _call({
            "object_type": "view",
            "pattern": "%",
            "detail_level": "full",
            "limit": _SEARCH_OBJECTS_LIMIT,
        })

    async def _list_routines(_args: Dict[str, Any]) -> str:
        rows: List[Dict[str, Any]] = []
        for obj_type in ("procedure", "function"):
            raw = await _call({
                "object_type": obj_type,
                "detail_level": "names",
                "limit": _SEARCH_OBJECTS_LIMIT,
            })
            rows.extend(
                {"name": n, "kind": obj_type.upper()}
                for n in _parse_names(raw, "results", "rows")
            )
        return json.dumps(rows)

    sql_tool = next(
        (t for t in tools if (getattr(t, "name", "") or "").lower() == "execute_sql"),
        None,
    )

    async def _sql(args: Dict[str, Any]) -> str:
        query = str(args.get("sql") or "")
        if not _assert_readonly_sql(query):
            return "ERROR: rejected non-read-only SQL (only SELECT is allowed)."
        return await _call({"sql": query})

    roles = _empty_roles()
    roles["schemas"] = [_AdaptedTool(
        "search_objects[schemas]",
        "List database schemas via dbhub search_objects",
        {}, _list_schemas,
    )]
    roles["tables"] = [_AdaptedTool(
        "search_objects[tables]",
        "List tables via dbhub search_objects",
        {"schema": {"type": "string"}}, _list_tables,
    )]
    roles["describe"] = [_AdaptedTool(
        "search_objects[describe]",
        "Full table structure via dbhub search_objects",
        {"schema": {"type": "string"}, "table": {"type": "string"}},
        _describe,
    )]
    roles["views"] = [_AdaptedTool(
        "search_objects[views]",
        "List views with definitions via dbhub search_objects",
        {}, _list_views,
    )]
    roles["routines"] = [_AdaptedTool(
        "search_objects[routines]",
        "List procedures/functions via dbhub search_objects",
        {}, _list_routines,
    )]
    if sql_tool is not None:
        roles["sql"] = [_AdaptedTool(
            "execute_sql[sql]",
            "Read-only catalog queries via dbhub execute_sql",
            {"sql": {"type": "string"}}, _sql,
        )]
    return roles


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


async def _oracle_typed_call(tool: Any, object_type: str) -> str:
    """Call an oracle tool that filters by ``object_type`` (pattern ``%``)."""
    payload: Dict[str, Any] = {}
    for arg in _tool_arg_names(tool):
        low = arg.lower()
        if "type" in low:
            payload[arg] = object_type
        elif "pattern" in low:
            payload[arg] = "%"
    return await _call_tool(tool, payload)


def _build_oracle_roles(tools: List[Any]) -> Optional[Dict[str, List[Any]]]:
    """Roles over oracle-mcp-server's schema tools (single connected schema).

    The server caches the connected schema (``TARGET_SCHEMA`` or the user's
    schema), so no schemas role is exposed — the walk runs in the default
    scope. Table listing prefers ``search_tables_schema`` (pattern ``%``);
    per-table definitions use ``get_table_schema`` (falling back to an exact
    ``search_tables_schema`` lookup). The DB-RE restructure adds the
    category/evidence roles when the corresponding tools are present:
    ``get_pl_sql_objects`` (views/triggers/sequences/routines by
    object_type), ``get_user_defined_types``, ``get_table_constraints``,
    ``get_table_indexes``, ``get_related_tables``, ``get_object_source``
    (per-object source) and ``run_sql_query`` (read-only catalog pack).
    Returns ``None`` when none of the three table tools is present.
    """
    by_name = {(getattr(t, "name", "") or "").lower(): t for t in tools}
    search = by_name.get("search_tables_schema")
    lookup = by_name.get("get_table_schema")
    multi = by_name.get("get_tables_schema")
    if search is None and lookup is None and multi is None:
        return None
    lister = search if search is not None else multi

    plsql = by_name.get("get_pl_sql_objects")
    udt = by_name.get("get_user_defined_types")
    constraints_t = by_name.get("get_table_constraints")
    indexes_t = by_name.get("get_table_indexes")
    related_t = by_name.get("get_related_tables")
    source_t = by_name.get("get_object_source")
    sql_t = by_name.get("run_sql_query")

    async def _list_tables(args: Dict[str, Any]) -> str:
        if search is not None:
            raw = await _call_oracle(search, pattern="%")
            if _parse_names(raw, "tables", "results", "rows"):
                return raw
            # The pinned server answers with PROSE ("Found N tables …\nTable:
            # X\nColumns: …"), capped at 20 — scrape the table names and
            # re-emit them as the JSON shape the walk's parser expects.
            names = re.findall(r"^Table:\s+(\S+)", _unwrap_untrusted(raw), re.MULTILINE)
            if names:
                return json.dumps({"tables": [{"table_name": n} for n in names]})
            return raw
        return await _call_oracle(multi)  # bulk dump; names parsed best-effort

    async def _describe(args: Dict[str, Any]) -> str:
        table = args.get("table")
        if lookup is not None:
            return await _call_oracle(lookup, table=table)
        if search is not None:
            return await _call_oracle(search, pattern=table)
        return "ERROR: no per-table oracle schema tool available"

    def _plsql_names_lister(obj_types: Tuple[str, ...]):
        async def _lister(_args: Dict[str, Any]) -> str:
            rows: List[Dict[str, Any]] = []
            for obj_type in obj_types:
                raw = await _oracle_typed_call(plsql, obj_type)
                rows.extend(
                    {"name": n, "kind": obj_type}
                    for n in _parse_names(raw, "objects", "results", "rows")
                )
            return json.dumps(rows)
        return _lister

    async def _list_types(_args: Dict[str, Any]) -> str:
        payload: Dict[str, Any] = {}
        for arg in _tool_arg_names(udt):
            if "pattern" in arg.lower():
                payload[arg] = "%"
        raw = await _call_tool(udt, payload)
        return json.dumps([{"name": n} for n in _parse_names(raw, "types", "results")])

    def _oracle_table_call(tool: Any):
        async def _call(args: Dict[str, Any]) -> str:
            return await _call_oracle(
                tool, table=args.get("table"), schema=args.get("schema")
            )
        return _call

    async def _source(args: Dict[str, Any]) -> str:
        payload: Dict[str, Any] = {}
        for arg in _tool_arg_names(source_t):
            low = arg.lower()
            if "type" in low:
                payload[arg] = args.get("object_type")
            elif "name" in low or "object" in low:
                payload[arg] = args.get("object_name")
        return await _call_tool(source_t, payload)

    async def _sql(args: Dict[str, Any]) -> str:
        query = str(args.get("sql") or "")
        if not _assert_readonly_sql(query):
            return "ERROR: rejected non-read-only SQL (only SELECT is allowed)."
        return await _call_tool(
            sql_t, _sql_args(sql_t, query, max_rows=args.get("max_rows"))
        )

    lister_name = getattr(lister, "name", "oracle_tool")
    describer_name = getattr(
        lookup if lookup is not None else search, "name", "oracle_tool"
    )
    roles = _empty_roles()
    roles["tables"] = [_AdaptedTool(
        f"{lister_name}[tables]",
        "List oracle tables via oracle-mcp-server",
        {}, _list_tables,
    )]
    roles["describe"] = [_AdaptedTool(
        f"{describer_name}[describe]",
        "Table schema via oracle-mcp-server",
        {"table": {"type": "string"}}, _describe,
    )]
    if plsql is not None:
        roles["views"] = [_AdaptedTool(
            f"{plsql.name}[views]",
            "List oracle views/materialized views",
            {}, _plsql_names_lister(("VIEW", "MATERIALIZED VIEW")),
        )]
        roles["triggers"] = [_AdaptedTool(
            f"{plsql.name}[triggers]",
            "List oracle triggers",
            {}, _plsql_names_lister(("TRIGGER",)),
        )]
        roles["sequences"] = [_AdaptedTool(
            f"{plsql.name}[sequences]",
            "List oracle sequences",
            {}, _plsql_names_lister(("SEQUENCE",)),
        )]
        roles["routines"] = [_AdaptedTool(
            f"{plsql.name}[routines]",
            "List oracle procedures/functions/packages",
            {}, _plsql_names_lister(("PROCEDURE", "FUNCTION", "PACKAGE")),
        )]
    if udt is not None:
        roles["types"] = [_AdaptedTool(
            f"{udt.name}[types]",
            "List oracle user-defined types",
            {}, _list_types,
        )]
    if constraints_t is not None:
        roles["constraints"] = [_AdaptedTool(
            f"{constraints_t.name}[constraints]",
            "Table constraints via oracle-mcp-server",
            {"table": {"type": "string"}, "schema": {"type": "string"}},
            _oracle_table_call(constraints_t),
        )]
    if indexes_t is not None:
        roles["indexes"] = [_AdaptedTool(
            f"{indexes_t.name}[indexes]",
            "Table indexes via oracle-mcp-server",
            {"table": {"type": "string"}, "schema": {"type": "string"}},
            _oracle_table_call(indexes_t),
        )]
    if related_t is not None:
        roles["relationships"] = [_AdaptedTool(
            f"{related_t.name}[relationships]",
            "Related tables (FK adjacency) via oracle-mcp-server",
            {"table": {"type": "string"}, "schema": {"type": "string"}},
            _oracle_table_call(related_t),
        )]
    if source_t is not None:
        roles["source"] = [_AdaptedTool(
            f"{source_t.name}[source]",
            "Object source via oracle-mcp-server",
            {"object_name": {"type": "string"}, "object_type": {"type": "string"}},
            _source,
        )]
    if sql_t is not None:
        roles["sql"] = [_AdaptedTool(
            f"{sql_t.name}[sql]",
            "Read-only catalog queries via oracle-mcp-server",
            {"sql": {"type": "string"}, "max_rows": {"type": "integer"}}, _sql,
        )]
    return roles


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
# Engine detection (drives the read-only SQL catalog packs)
# --------------------------------------------------------------------------- #
def _detect_engine(entity: Any, tools: Optional[List[Any]] = None) -> Optional[str]:
    """Best-effort engine key: ``postgresql`` | ``oracle`` | None.

    Precedence: the preset ``db_type`` attribute > the masked DSN scheme >
    oracle-specific tool names. Only the two focus engines have SQL packs;
    everything else returns None (adapter/name heuristics still run).
    """
    db_type = (getattr(entity, "db_type", None) or "").strip().lower()
    if db_type:
        if db_type in ("postgres", "postgresql"):
            return "postgresql"
        if db_type == "oracle":
            return "oracle"
        return None
    dsn = (getattr(entity, "dsn_masked", None) or "").strip().lower()
    if dsn.startswith("jdbc:oracle"):
        return "oracle"
    if "://" in dsn:
        scheme = dsn.split("://", 1)[0]
        if scheme in ("postgres", "postgresql"):
            return "postgresql"
        if scheme == "oracle":
            return "oracle"
    if tools:
        names = {(getattr(t, "name", "") or "").lower() for t in tools}
        if "get_pl_sql_objects" in names or "search_tables_schema" in names:
            return "oracle"
    return None


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
    if not names and data is None:
        # Fallback for NON-JSON prose results: quoted strings (capped). A
        # successfully parsed but empty JSON collection ("{\"objects\": []}")
        # must stay empty — scraping quotes there would return the JSON key
        # ("objects") as a phantom object name.
        names = re.findall(r'"([^"\n]{1,128})"', text)[:100]
    seen: set = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _parse_object_rows(text: str, *collection_keys: str) -> List[Dict[str, Any]]:
    """Parse a listing result into normalized row dicts (``name`` + extras).

    Unlike :func:`_parse_names` this keeps the whole row (schema, kind,
    definition, meta fields) so category collections keep their evidence.
    """
    if not text or text.startswith("ERROR:"):
        return []
    try:
        data = json.loads(text.strip())
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        data = _unwrap_name_collection(
            data, *(collection_keys or ("results", "rows", "objects", "items"))
        )
    rows: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                rows.append({"name": item})
            elif isinstance(item, dict):
                for key in _NAME_KEYS:
                    if isinstance(item.get(key), str) and item[key]:
                        rows.append(item)
                        break
    return rows


def _rows_from_sql_result(text: str) -> List[Dict[str, Any]]:
    """Normalize one SQL-tool result into a row-dict list (best-effort).

    Handles the dbhub envelope ``{"success": true, "data": {"columns":
    [...], "rows": [[...], ...]}}`` (columns zipped onto each row), a bare
    ``{"columns": [...], "rows": [...]}`` dict, a plain list of dicts, and —
    as the text fallback — a rendered pipe table.
    """
    if not text or text.startswith("ERROR:"):
        return []
    try:
        data = json.loads(text.strip())
    except (ValueError, TypeError):
        return _parse_pipe_table(text)
    if isinstance(data, dict):
        inner = data.get("data") if isinstance(data.get("data"), dict) else data
        columns = inner.get("columns")
        rows = inner.get("rows")
        if isinstance(columns, list) and isinstance(rows, list):
            names = [str(c) for c in columns]
            out = []
            for row in rows:
                if isinstance(row, (list, tuple)):
                    out.append(dict(zip(names, [str(v) for v in row])))
                elif isinstance(row, dict):
                    out.append(row)
            return out
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


# The pinned oracle-mcp-server wraps EVERY run_sql_query result (tables,
# empty results, errors) in an anti-injection envelope; the payload proper
# lives between the <untrusted-data-{uid}> tags with < / > HTML-escaped.
_UNTRUSTED_RE = re.compile(
    r"<untrusted-data-[0-9a-f-]+>\n?(.*?)\n?</untrusted-data-", re.DOTALL
)


def _unwrap_untrusted(text: str) -> str:
    """Strip the oracle-mcp-server untrusted-data envelope (no-op elsewhere)."""
    m = _UNTRUSTED_RE.search(text or "")
    return m.group(1).strip() if m else (text or "")


_PIPE_SPLIT_RE = re.compile(r"(?<!\\)\|")


def _split_pipe_row(line: str) -> List[str]:
    """Cells of one rendered table row, ``\\|`` escapes unescaped.

    The oracle-mcp-server formatter escapes ``|`` inside cells (PL/SQL
    ``||`` concatenation, defaults); a naive split shreds such rows.
    """
    cells = _PIPE_SPLIT_RE.split(line.strip())
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [c.strip().replace("\\|", "|") for c in cells]


def _parse_pipe_table(text: str) -> List[Dict[str, Any]]:
    """Rows from a rendered ``| a | b |`` table (text tool fallback)."""
    lines = [l for l in (text or "").splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = _split_pipe_row(lines[0])
    rows: List[Dict[str, Any]] = []
    for line in lines[1:]:
        cells = _split_pipe_row(line)
        if cells and set("".join(cells)) <= {"-", " ", ":"}:
            continue  # separator row
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Strict JSON array of objects from an LLM batch reply (best-effort).

    Tolerates a wrapping code fence and stray prose around the array; any
    parse failure returns [] (the caller keeps the deterministic render).
    """
    if not text:
        return []
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3]
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(t[start:end + 1])
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


# --------------------------------------------------------------------------- #
# Parsed table structure (re-parse of the controlled renders / JSON detail)
# --------------------------------------------------------------------------- #
_COL_ROW_RE = re.compile(r"^\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]*)\|$")
_INDEX_LINE_RE = re.compile(r"^index\s+(\S+?)\s*\(([^)]*)\)\s*(.*)$")
_TABLE_HEADER_RE = re.compile(r"^table\s+\S+(?:\s+\([^)]*\))?\s*(?:—\s*(.*))?$")
_ROW_COUNT_RE = re.compile(r"~(\d+)\s*rows")


def _parse_table_definition(definition: str) -> Dict[str, Any]:
    """Best-effort structured meta from one table definition text.

    Accepts either a JSON object with ``columns`` (dbhub raw detail,
    oracle JSON describe) or the controlled ``_render_search_full`` render
    (header line + pipe column table + ``index …`` lines). Returns a dict
    with only the non-empty pieces (``columns`` / ``indexes`` / ``comment``
    / ``row_count``).
    """
    out: Dict[str, Any] = {}
    text = (definition or "").strip()
    if not text or text.startswith("ERROR:"):
        return out
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            columns = data.get("columns")
            if isinstance(columns, list) and columns:
                parsed = []
                for c in columns:
                    if isinstance(c, dict) and c.get("name"):
                        parsed.append({
                            "name": str(c.get("name")),
                            "type": str(c.get("type") or "?"),
                            "nullable": bool(c.get("nullable", True)),
                            "default": c.get("default"),
                        })
                if parsed:
                    out["columns"] = parsed
            if isinstance(data.get("comment"), str) and data["comment"]:
                out["comment"] = data["comment"]
            if isinstance(data.get("row_count"), int):
                out["row_count"] = data["row_count"]
            indexes = data.get("indexes")
            if isinstance(indexes, list) and indexes:
                parsed_idx = _parse_index_dicts(indexes)
                if parsed_idx:
                    out["indexes"] = parsed_idx
            if out:
                return out

    columns: List[Dict[str, Any]] = []
    indexes: List[Dict[str, Any]] = []
    comment = ""
    row_count: Optional[int] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = _TABLE_HEADER_RE.match(line)
        if header:
            m = _ROW_COUNT_RE.search(header.group(1) or "")
            if m:
                row_count = int(m.group(1))
            continue
        idx = _INDEX_LINE_RE.match(line)
        if idx:
            flags = (idx.group(3) or "").upper()
            indexes.append({
                "name": idx.group(1),
                "columns": [c.strip() for c in idx.group(2).split(",") if c.strip()],
                "unique": "UNIQUE" in flags,
                "primary": "PRIMARY" in flags,
            })
            continue
        if line.startswith("comment:"):
            comment = line[len("comment:"):].strip()
            continue
        col = _COL_ROW_RE.match(line)
        if col:
            cells = [col.group(i).strip() for i in (1, 2, 3, 4)]
            if cells[0].lower() == "column" or set(cells[0]) <= {"-", ":", " "}:
                continue  # header / separator row
            columns.append({
                "name": cells[0],
                "type": cells[1],
                "nullable": cells[2].upper().startswith("YES"),
                "default": None if cells[3] == "-" else cells[3],
            })
    if columns:
        out["columns"] = columns
    if indexes:
        out["indexes"] = indexes
    if comment:
        out["comment"] = comment
    if row_count is not None:
        out["row_count"] = row_count
    return out


def _parse_index_dicts(indexes: List[Any]) -> List[Dict[str, Any]]:
    """Normalize JSON index dicts (dbhub/oracle shapes) into one render form."""
    out: List[Dict[str, Any]] = []
    for i in indexes:
        if not isinstance(i, dict) or not (i.get("name") or i.get("index_name")):
            continue
        cols_raw = i.get("columns") or i.get("column_name") or []
        if isinstance(cols_raw, str):
            cols = [c.strip() for c in cols_raw.strip("{}").split(",") if c.strip()]
        elif isinstance(cols_raw, list):
            cols = [str(c) for c in cols_raw]
        else:
            cols = []
        unique = bool(i.get("unique") or (str(i.get("uniqueness") or "").upper() == "UNIQUE"))
        out.append({
            "name": str(i.get("name") or i.get("index_name")),
            "columns": cols,
            "unique": unique,
            "primary": bool(i.get("primary")),
            "ddl": i.get("index_def") or i.get("indexdef") or i.get("ddl") or "",
        })
    return out


def _qualify_name(schema: Optional[str], name: str, tables: Dict[str, Any]) -> str:
    """Schema-qualify ``name`` against the known table keys when possible."""
    if schema:
        full = f"{schema}.{name}"
        if full in tables:
            return full
    if name in tables:
        return name
    if schema:
        low = name.lower()
        for full in tables:
            if full.lower() == f"{schema}.{low}":
                return full
    return f"{schema}.{name}" if schema else name


def _edges_from_fk_rows(
    rows: List[Dict[str, Any]], tables: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Group catalog FK rows (one row per column pair) into edge dicts."""
    groups: Dict[Tuple[str, str, str], Dict[str, List[str]]] = {}
    for row in rows:
        t_schema = row.get("table_schema") or row.get("schema") or row.get("owner")
        t_name = row.get("table_name") or row.get("table") or row.get("name")
        r_schema = row.get("foreign_schema") or row.get("r_owner")
        r_table = row.get("foreign_table") or row.get("r_table") or row.get("referenced_table")
        if not (isinstance(t_name, str) and t_name and isinstance(r_table, str) and r_table):
            continue
        frm = _qualify_name(str(t_schema) if t_schema else None, t_name, tables)
        to = _qualify_name(str(r_schema) if r_schema else None, r_table, tables)
        constraint = row.get("constraint_name") or row.get("constraint") or ""
        key = (frm, to, str(constraint))
        group = groups.setdefault(key, {"from_cols": [], "to_cols": []})
        col = row.get("column_name") or row.get("column") or row.get("from_column")
        rcol = row.get("foreign_column") or row.get("r_column") or row.get("referenced_column")
        if isinstance(col, str) and col and col not in group["from_cols"]:
            group["from_cols"].append(col)
        if isinstance(rcol, str) and rcol and rcol not in group["to_cols"]:
            group["to_cols"].append(rcol)
    edges: List[Dict[str, Any]] = []
    for (frm, to, constraint), group in sorted(groups.items()):
        edges.append({
            "from": frm,
            "from_cols": group["from_cols"],
            "to": to,
            "to_cols": group["to_cols"],
            "constraint": constraint,
            "kind": "fk",
        })
    return edges


# --------------------------------------------------------------------------- #
# Category collections (views / triggers / routines / sequences / types)
# --------------------------------------------------------------------------- #
_CATEGORIES = ("views", "triggers", "routines", "sequences", "types")
_CATEGORY_TITLES = {
    "views": "Views",
    "triggers": "Triggers",
    "routines": "Procedures & Functions",
    "sequences": "Sequences",
    "types": "Types",
}
_CATEGORY_PREFIX = {
    "views": "view", "triggers": "trg", "routines": "rtn",
    "sequences": "seq", "types": "typ",
}
#: ``object_type`` value for the oracle source tool (routines use their kind).
_SOURCE_TYPE_BY_CATEGORY = {
    "views": "VIEW", "triggers": "TRIGGER",
    "sequences": "SEQUENCE", "types": "TYPE",
}
_SOURCE_META_KEYS = ("definition", "source", "ddl", "query", "text", "create_statement")


def _category_entry(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Normalize one listing row into (full_name, meta) for a category."""
    name = ""
    for key in _NAME_KEYS:
        if isinstance(row.get(key), str) and row[key]:
            name = row[key]
            break
    schema = row.get("schema") or row.get("schema_name") or row.get("owner")
    schema = str(schema) if isinstance(schema, str) and schema else None
    full = f"{schema}.{name}" if schema else name
    source = ""
    for key in _SOURCE_META_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            source = value
            break
    kind = row.get("kind") or row.get("object_type") or row.get("type") or ""
    meta: Dict[str, Any] = {"schema": schema, "name": name}
    if kind:
        meta["kind"] = str(kind)
    if source:
        meta["source"] = source
    extras = {
        k: v for k, v in row.items()
        if k not in _NAME_KEYS and k not in _SOURCE_META_KEYS
        and k not in ("kind", "object_type", "schema", "schema_name", "owner")
        and isinstance(v, (str, int, float, bool)) and v not in ("", None)
    }
    if extras:
        meta["meta"] = extras
    return full, meta


async def _collect_category(
    cat: str,
    roles: Dict[str, List[Any]],
    bounded_call,
) -> Dict[str, Dict[str, Any]]:
    """List one category's objects via its role tool (best-effort)."""
    entries: Dict[str, Dict[str, Any]] = {}
    tools_for_role = roles.get(cat) or []
    tool = tools_for_role[0] if tools_for_role else None
    if tool is None:
        return entries
    raw = await bounded_call(tool, _build_tool_args(tool))
    if not raw or raw.startswith("ERROR:"):
        return entries
    for row in _parse_object_rows(raw):
        full, meta = _category_entry(row)
        if not full or full in entries:
            continue
        entries[full] = meta
    return entries


def _join_line_rows(
    rows: List[Dict[str, Any]],
    key_fields: Tuple[str, ...],
    *,
    type_key: Optional[str] = None,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Regroup Oracle ``ALL_SOURCE``-style line rows into one row per object."""
    groups: Dict[Tuple, Dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(f) or "") for f in key_fields)
        if not any(key):
            continue
        g = groups.get(key)
        if g is None:
            # Seed with the row's meta but NOT its text: the line is appended
            # below, otherwise the first source line would be duplicated.
            g = dict(row)
            g[text_key] = ""
            groups[key] = g
        line = str(row.get(text_key) or "")
        g[text_key] = g[text_key] + line + "\n"
        if type_key and row.get(type_key):
            g[type_key] = row.get(type_key)
    return list(groups.values())


def _ora_page_query(query: str, lo: int, hi: int) -> str:
    """ROWNUM-window wrapper fetching one catalog page.

    ``lo``/``hi`` are loop-counter ints only (never user input) — as inert
    as the catalog-derived owner literals guarded by
    ``_ORA_IDENTIFIER_RE``.
    """
    return (
        'SELECT * FROM (SELECT inner_q.*, ROWNUM AS "rnum" '
        f"FROM ({query}) inner_q WHERE ROWNUM <= {hi}) "
        f'WHERE "rnum" > {lo}'
    )


async def _paged_rows(
    sql_tool: Any, bounded_call: Callable[..., str], query: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """All rows of a catalog query, fetched in char-cap-safe pages.

    Stops on the first short page (or the page ceiling). Returns the LAST
    raw response — diagnostics for the caller's error messages — plus the
    aggregated rows.
    """
    rows: List[Dict[str, Any]] = []
    raw = ""
    for page in range(_ORA_MAX_PAGES):
        raw = await bounded_call(
            sql_tool,
            _sql_args(
                sql_tool,
                _ora_page_query(
                    query, page * _ORA_PAGE_ROWS, (page + 1) * _ORA_PAGE_ROWS
                ),
                max_rows=_ORA_PAGE_ROWS,
            ),
        )
        chunk = _rows_from_sql_result(raw)
        rows.extend(chunk)
        if len(chunk) < _ORA_PAGE_ROWS:
            break
    return raw, rows


async def _oracle_session_schema(
    sql_tool: Any, bounded_call: Callable[..., str]
) -> Optional[str]:
    """Which schema the MCP server resolves object names in (None = unknown)."""
    raw = await bounded_call(
        sql_tool, _sql_args(sql_tool, _ORA_SESSION_SCHEMA_QUERY)
    )
    for row in _rows_from_sql_result(raw):
        value = str(row.get("owner") or "").strip().upper()
        if value:
            return value
    return None


_ORA_ERROR_HEAD_RE = re.compile(r"\bORA-\d{5}\b")


def _looks_like_server_error(text: str) -> bool:
    """Server-side failures arrive as plain text (never our ERROR: prefix).

    oracle-mcp-server answers ``Database error: ORA-…`` / ``Unexpected
    error executing query: …`` / ``Error retrieving object source: …`` —
    none of that is documentation, so it must not be stored as one.
    """
    head = (text or "")[:400]
    return bool(
        "Database error" in head
        or "Unexpected error" in head
        or "Error retrieving" in head
        or _ORA_ERROR_HEAD_RE.search(head)
    )


async def _oracle_bulk_tables(
    sql_tool: Any, bounded_call: Callable[..., str], errors: List[str]
) -> Optional[Tuple[List[str], Dict[str, Dict[str, Any]]]]:
    """Cross-schema table walk via the read-only sql tool (ALL_* catalog).

    oracle-mcp-server caches ONE connected schema (TARGET_SCHEMA or the
    login user), so its per-table tools cannot see an enterprise
    multi-schema layout; the ALL_* catalog is the only cross-schema
    surface. Two paged calls per schema replace thousands of single-schema
    probes. Returns ``None`` when the surface cannot serve the catalog
    queries — the caller falls back to the adapter walk.
    """

    async def _q(sql: str) -> Tuple[str, List[Dict[str, Any]]]:
        return await _paged_rows(sql_tool, bounded_call, sql)

    _, owner_rows = await _q(_ORA_OWNERS_QUERY)
    owners = [
        str(row.get("owner") or "") for row in owner_rows
        if _ORA_IDENTIFIER_RE.match(str(row.get("owner") or ""))
        and _ora_user_owner(str(row.get("owner") or ""))
    ][:MAX_SCHEMAS]
    if not owners:
        return None

    tables: Dict[str, Dict[str, Any]] = {}
    for owner in owners:
        raw, rows = await _q(_ORA_COLUMNS_QUERY.format(owner=owner))
        if not rows:
            # The WHY travels with the error: empty result, server-side ORA-*,
            # or the pinned formatter's None-cell crash all look identical
            # to "zero rows" once the envelope is stripped away.
            errors.append(
                f"no all_tab_columns rows for schema {owner} "
                f"({_cap(_unwrap_untrusted(raw), 120)!r})"
            )
            continue
        columns: Dict[str, List[Dict[str, Any]]] = {}
        counts: Dict[str, int] = {}
        for row in rows:
            tname = str(row.get("table_name") or "")
            if not tname or tname.startswith("BIN$"):
                continue
            default = row.get("data_default")
            columns.setdefault(tname, []).append({
                "name": str(row.get("column_name") or ""),
                "type": str(row.get("data_type") or "?"),
                "nullable": str(row.get("nullable") or "").upper() == "Y",
                "default": None if default in (None, "") else str(default),
            })
            num = str(row.get("num_rows") or "")
            # NVL(num_rows, 0): 0 = no optimizer stats, not an empty table —
            # claim a row count only when stats actually exist.
            if num.isdigit() and int(num) > 0:
                counts[tname] = int(num)
        comments = {
            str(r.get("table_name") or ""): str(r.get("comments") or "")
            for r in (
                await _q(_ORA_COMMENTS_QUERY.format(owner=owner))
            )[1]
            if r.get("comments")
        }
        for tname in sorted(columns):
            # Controlled render — re-parsed by _parse_table_definition so the
            # bulk path and the adapter path share one structure parser.
            definition = [
                f"table {owner}.{tname}"
                + (f" — ~{counts[tname]} rows" if tname in counts else ""),
                "| column | type | null | default |",
                "| --- | --- | --- | --- |",
            ]
            definition.extend(
                "| {} | {} | {} | {} |".format(
                    c["name"], c["type"],
                    "YES" if c["nullable"] else "NO",
                    "-" if c["default"] is None else c["default"],
                )
                for c in columns[tname]
            )
            if comments.get(tname):
                definition.append(f"comment: {comments[tname]}")
            entry: Dict[str, Any] = {
                "schema": owner,
                "table": tname,
                "definition": _cap("\n".join(definition), MAX_DEFINITION_CHARS),
            }
            entry.update(_parse_table_definition(entry["definition"]))
            tables[f"{owner}.{tname}"] = entry
    if not tables:
        return None
    return owners, tables


# --------------------------------------------------------------------------- #
# Introspection walk: schemas → tables → structure → categories → SQL pack
# --------------------------------------------------------------------------- #
async def _introspect(
    roles: Dict[str, List[Any]], engine: Optional[str] = None
) -> Dict[str, Any]:
    """Run the deterministic introspection walk over classified tools.

    Returns the payload consumed by every later stage:
    ``{"schemas": [...], "tables": {full: {schema, table, definition,
    columns?, indexes?, comment?, row_count?, constraints?}},
    "fk_edges": [{from, from_cols, to, to_cols, constraint, kind}],
    "views"/"triggers"/"routines"/"sequences"/"types": {full: {schema,
    name, kind, source?, meta?}}, "tools_used": {role: name},
    "unavailable": [role…]}``. Oracle engines with a sql role take the
    cross-schema ``_oracle_bulk_tables`` path first. Raises ValueError when
    the surface cannot produce a schema at all — an honest failure instead
    of empty docs.
    """
    tools_used: Dict[str, str] = {}
    unavailable: List[str] = []
    schema_names: List[str] = []
    errors: List[str] = []
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

    sql_tool = roles["sql"][0] if roles.get("sql") else None

    # Oracle: cross-schema BULK walk over the ALL_* catalog — the pinned
    # server caches one schema, so per-table tools miss the enterprise
    # multi-schema layout. Tried before the table-tool check below: a
    # sql-only surface is a complete walk too. None → adapter/generic path.
    bulk: Optional[Tuple[List[str], Dict[str, Dict[str, Any]]]] = None
    if engine == "oracle" and sql_tool is not None:
        bulk = await _oracle_bulk_tables(sql_tool, _bounded_call, errors)
        if bulk is not None:
            schema_names, tables = bulk
            tools_used["tables"] = getattr(sql_tool, "name", "")
            logger.info(
                "oracle bulk walk: %d schema(s), %d table(s) via the "
                "read-only sql tool",
                len(schema_names), len(tables),
            )

    if bulk is None:
        schema_tool = roles["schemas"][0] if roles.get("schemas") else None
        if schema_tool is not None:
            tools_used["schemas"] = getattr(schema_tool, "name", "")
            raw = await _bounded_call(
                schema_tool, _build_tool_args(schema_tool)
            )
            schema_names = _parse_names(raw, "schemas", "databases")
            if engine == "postgresql":
                # dbhub lists pg_catalog/information_schema/pg_toast* —
                # system namespaces, never documentation targets.
                schema_names = [
                    s for s in schema_names if _pg_user_schema(s)
                ]
            schema_names = schema_names[:MAX_SCHEMAS]

        table_tool = roles["tables"][0] if roles.get("tables") else None
        if table_tool is None:
            raise ValueError(
                "No table-listing MCP tool found (expected a tool named like "
                "'list_tables'/'search_tables'); cannot introspect the database."
            )
        tools_used["tables"] = getattr(table_tool, "name", "")

        describer = roles["describe"][0] if roles.get("describe") else None
        ddl_tool = roles["ddl"][0] if roles.get("ddl") else None
        if describer is not None:
            tools_used["describe"] = getattr(describer, "name", "")
        if ddl_tool is not None:
            tools_used["ddl"] = getattr(ddl_tool, "name", "")

        scopes: List[Optional[str]] = schema_names or [None]
        tables = {}
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
                entry: Dict[str, Any] = {
                    "schema": schema,
                    "table": table_name,
                    "definition": _cap(definition or "", MAX_DEFINITION_CHARS),
                }
                for key, value in _parse_table_definition(definition).items():
                    entry[key] = value
                tables[full] = entry
    if not tables:
        detail = "; ".join(errors[:3]) or "unknown reason"
        raise ValueError(
            f"Database introspection produced no tables ({detail}); "
            "check the MCP server connection and tool arguments."
        )

    fk_edges: List[Dict[str, Any]] = []
    edge_seen: set = set()

    def _add_edge(
        frm: str, from_cols: List[str], to: str, to_cols: List[str],
        constraint: str, kind: str,
    ) -> None:
        key = (frm, to, constraint, tuple(from_cols or []), tuple(to_cols or []))
        if key in edge_seen or not frm or not to:
            return
        edge_seen.add(key)
        fk_edges.append({
            "from": frm, "from_cols": list(from_cols or []),
            "to": to, "to_cols": list(to_cols or []),
            "constraint": constraint or "", "kind": kind,
        })

    # Evidence probes (constraints / indexes / related tables), capped by the
    # walk budget: structure-rich tables first (they are the likely FK hubs).
    evidence_tables = sorted(
        tables,
        key=lambda f: (-len(tables[f].get("columns") or []), f),
    )[:_fk_evidence_tables()]
    if bulk is not None:
        # The oracle per-table evidence tools operate on the ONE cached
        # schema — wrong-schema answers after a cross-schema walk; the SQL
        # pack supplies constraints/indexes globally instead.
        evidence_tables = []

    constraints_tool = roles["constraints"][0] if roles.get("constraints") else None
    if constraints_tool is not None:
        tools_used["constraints"] = getattr(constraints_tool, "name", "")
        for full in evidence_tables:
            meta = tables[full]
            raw = await _bounded_call(
                constraints_tool,
                _build_tool_args(
                    constraints_tool, schema=meta.get("schema"), table=meta.get("table")
                ),
            )
            if not raw or raw.startswith("ERROR:"):
                continue
            rows = _rows_from_sql_result(raw) or _parse_object_rows(raw)
            parsed: List[Dict[str, Any]] = []
            for row in rows:
                cname = row.get("constraint_name") or row.get("name") or ""
                ctype = row.get("constraint_type") or row.get("type") or ""
                if not cname:
                    continue
                col = str(row.get("column_name") or row.get("column") or "")
                parsed.append({
                    "name": str(cname), "type": str(ctype),
                    "columns": [col] if col else [],
                })
                if str(ctype).upper() in ("FOREIGN KEY", "R"):
                    r_table = row.get("foreign_table") or row.get("references_table") or ""
                    if r_table:
                        r_schema = row.get("foreign_schema")
                        _add_edge(
                            full, [col] if col else [],
                            _qualify_name(
                                str(r_schema) if r_schema else meta.get("schema"),
                                str(r_table), tables,
                            ),
                            [str(row.get("foreign_column") or row.get("referenced_column") or "")],
                            str(cname), "fk",
                        )
            if parsed:
                merged = {
                    f'{c["name"]}|{c["type"]}': c
                    for c in (tables[full].get("constraints") or [])
                }
                for c in parsed:
                    merged.setdefault(f'{c["name"]}|{c["type"]}', c)
                tables[full]["constraints"] = list(merged.values())

    indexes_tool = roles["indexes"][0] if roles.get("indexes") else None
    if indexes_tool is not None:
        tools_used["indexes"] = getattr(indexes_tool, "name", "")
        for full in evidence_tables:
            meta = tables[full]
            raw = await _bounded_call(
                indexes_tool,
                _build_tool_args(
                    indexes_tool, schema=meta.get("schema"), table=meta.get("table")
                ),
            )
            if not raw or raw.startswith("ERROR:"):
                continue
            rows = _rows_from_sql_result(raw) or _parse_object_rows(raw)
            if rows:
                parsed_idx = _parse_index_dicts(rows)
                if parsed_idx:
                    existing = {i["name"]: i for i in (meta.get("indexes") or [])}
                    for i in parsed_idx:
                        existing.setdefault(i["name"], i)
                    meta["indexes"] = list(existing.values())

    relationships_tool = roles["relationships"][0] if roles.get("relationships") else None
    if relationships_tool is not None:
        tools_used["relationships"] = getattr(relationships_tool, "name", "")
        for full in evidence_tables:
            meta = tables[full]
            raw = await _bounded_call(
                relationships_tool,
                _build_tool_args(
                    relationships_tool, schema=meta.get("schema"), table=meta.get("table")
                ),
            )
            if not raw or raw.startswith("ERROR:"):
                continue
            for row in _rows_from_sql_result(raw) or _parse_object_rows(raw):
                target = (
                    row.get("related_table") or row.get("referenced_table")
                    or row.get("references_table") or row.get("table_name")
                    or row.get("name") or ""
                )
                if not isinstance(target, str) or not target or target == meta.get("table"):
                    continue
                cols = row.get("column_name") or row.get("column") or ""
                rcols = row.get("foreign_column") or row.get("referenced_column") or ""
                _add_edge(
                    full,
                    [str(cols)] if cols else [],
                    _qualify_name(meta.get("schema"), target, tables),
                    [str(rcols)] if rcols else [],
                    str(row.get("constraint_name") or row.get("foreign_key") or ""),
                    "related",
                )

    # Category listings (adapter roles; the SQL pack below may REPLACE
    # them: adapter surfaces are single-schema and/or extension-noisy).
    categories: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cat in _CATEGORIES:
        entries = await _collect_category(cat, roles, _bounded_call)
        if entries:
            tools_used.setdefault(
                cat, getattr((roles.get(cat) or [None])[0], "name", "")
            )
        elif roles.get(cat):
            unavailable.append(cat)
        categories[cat] = entries

    # Read-only SQL catalog packs (engine known + sql tool present): fill FK
    # edges, indexes and the category collections. Pack rows are AUTHORITATIVE
    # for their categories — the adapter listings are single-schema subsets
    # (oracle preset) or carry extension noise (pgvector functions in public),
    # so a successful-but-empty pack answer replaces them too (an
    # extension-only catalog genuinely has no user objects).
    pack = _PG_SQL_PACK if engine == "postgresql" else (
        _ORACLE_SQL_PACK if engine == "oracle" else None
    )
    pack_categories: Dict[str, Dict[str, Dict[str, Any]]] = {}
    pack_matviews: Dict[str, Dict[str, Any]] = {}
    if sql_tool is not None and pack:
        tools_used["sql"] = getattr(sql_tool, "name", "")
        for name, query in pack.items():
            # Constants are SELECT-only by construction; the guard is defense
            # in depth (it also documents the contract for future editors).
            if not _assert_readonly_sql(query):  # pragma: no cover - constants
                logger.warning("SQL pack query %r failed the read-only guard", name)
                continue
            if engine == "oracle":
                raw, rows = await _paged_rows(sql_tool, _bounded_call, query)
            else:
                raw = await _bounded_call(
                    sql_tool, _sql_args(sql_tool, query)
                )
                rows = _rows_from_sql_result(raw)
            if not rows:
                ok = bool(raw) and not raw.startswith("ERROR:")
                if ok:
                    # A server-side ORA-*/formatter crash arrives as plain
                    # text: an error is NOT an authoritative "empty".
                    ok = not _looks_like_server_error(_unwrap_untrusted(raw))
                if ok and name in _CATEGORIES:
                    pack_categories.setdefault(name, {})  # catalog says: empty
                elif ok:
                    unavailable.append(f"sql:{name}")
                continue
            if name == "fk_edges":
                for edge in _edges_from_fk_rows(rows, tables):
                    _add_edge(
                        edge["from"], edge["from_cols"], edge["to"],
                        edge["to_cols"], edge["constraint"], "fk",
                    )
            elif name == "indexes":
                grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
                for row in rows:
                    schema = str(row.get("table_schema") or "") or None
                    table = str(row.get("table_name") or "")
                    idx_name = str(row.get("index_name") or "")
                    if not table or not idx_name:
                        continue
                    key = (schema or "", table, idx_name)
                    g = grouped.setdefault(key, {
                        "name": idx_name, "columns": [], "unique": False,
                        "primary": False, "ddl": "",
                    })
                    col = row.get("column_name")
                    if isinstance(col, str) and col:
                        g["columns"].append(col)
                    g["unique"] = g["unique"] or str(
                        row.get("uniqueness") or ""
                    ).upper() == "UNIQUE"
                    g["ddl"] = g["ddl"] or str(
                        row.get("index_def") or row.get("indexdef") or ""
                    )
                for (schema, table, _n), idx in grouped.items():
                    meta = tables.get(_qualify_name(schema, table, tables))
                    if meta is None:
                        continue
                    existing = {i["name"]: i for i in (meta.get("indexes") or [])}
                    existing.setdefault(idx["name"], idx)
                    meta["indexes"] = list(existing.values())
            elif name == "triggers":
                merged_rows = (
                    _join_line_rows(
                        rows, ("schema_name", "trigger_name"), text_key="line_text",
                    )
                    if engine == "oracle" else rows
                )
                for row in merged_rows:
                    schema = str(row.get("schema_name") or "") or None
                    tname = str(row.get("trigger_name") or "")
                    if not tname:
                        continue
                    full = f"{schema}.{tname}" if schema else tname
                    trig_meta: Dict[str, Any] = {
                        "schema": schema, "name": tname, "kind": "TRIGGER",
                        "meta": {
                            k: str(v) for k, v in (
                                ("table", row.get("table_name")),
                                ("event", row.get("triggering_event")),
                                ("type", row.get("trigger_type")),
                                ("function", row.get("function_name")),
                                ("status", row.get("enabled") or row.get("status")),
                            ) if v not in (None, "")
                        },
                    }
                    definition = row.get("definition") or row.get("line_text")
                    if isinstance(definition, str) and definition.strip():
                        trig_meta["source"] = definition
                    pack_categories.setdefault("triggers", {})[full] = trig_meta
            elif name == "sequences":
                for row in rows:
                    schema = str(row.get("schema_name") or "") or None
                    sname = str(row.get("sequence_name") or "")
                    if not sname:
                        continue
                    full = f"{schema}.{sname}" if schema else sname
                    pack_categories.setdefault("sequences", {})[full] = {
                        "schema": schema, "name": sname, "kind": "SEQUENCE",
                        "meta": {
                            k: str(v) for k, v in (
                                ("min_value", row.get("min_value")),
                                ("max_value", row.get("max_value")),
                                ("increment", row.get("increment_by") or row.get("increment")),
                                ("cycle", row.get("cycle_flag")),
                                ("start_value", row.get("start_value")),
                            ) if v not in (None, "")
                        },
                    }
            elif name == "views":
                # Oracle names-only listing; definitions come from the capped
                # per-object source fetch below.
                for row in rows:
                    schema = str(row.get("schema_name") or "") or None
                    vname = str(row.get("name") or row.get("view_name") or "")
                    if not vname:
                        continue
                    full = f"{schema}.{vname}" if schema else vname
                    pack_categories.setdefault("views", {})[full] = {
                        "schema": schema, "name": vname, "kind": "VIEW",
                    }
            elif name == "matviews":
                for row in rows:
                    schema = str(row.get("schema_name") or "") or None
                    vname = str(row.get("view_name") or row.get("mview_name") or "")
                    if not vname:
                        continue
                    full = f"{schema}.{vname}" if schema else vname
                    view_meta: Dict[str, Any] = {
                        "schema": schema, "name": vname, "kind": "MATERIALIZED VIEW",
                    }
                    definition = row.get("view_definition") or row.get("definition")
                    if isinstance(definition, str) and definition.strip():
                        view_meta["source"] = definition
                    pack_matviews[full] = view_meta
            elif name == "types":
                for row in rows:
                    schema = str(row.get("schema_name") or "") or None
                    tname = str(row.get("type_name") or "")
                    if not tname:
                        continue
                    full = f"{schema}.{tname}" if schema else tname
                    type_meta: Dict[str, Any] = {
                        "schema": schema, "name": tname, "kind": "TYPE",
                    }
                    attrs = row.get("attributes") or row.get("typecode")
                    if attrs:
                        type_meta["meta"] = {"attributes": str(attrs)}
                    pack_categories.setdefault("types", {})[full] = type_meta
            elif name == "routines":
                merged_rows = (
                    _join_line_rows(
                        rows, ("schema_name", "object_name", "object_type"),
                        type_key="object_type", text_key="line_text",
                    )
                    if engine == "oracle" else rows
                )
                for row in merged_rows:
                    schema = str(row.get("schema_name") or "") or None
                    rname = str(row.get("routine_name") or row.get("object_name") or "")
                    if not rname:
                        continue
                    full = f"{schema}.{rname}" if schema else rname
                    kind = str(row.get("kind") or row.get("object_type") or "FUNCTION")
                    routine_meta: Dict[str, Any] = {
                        "schema": schema, "name": rname, "kind": kind,
                    }
                    source = row.get("source") or row.get("line_text")
                    if isinstance(source, str) and source.strip():
                        routine_meta["source"] = _cap(source, MAX_DEFINITION_CHARS)
                    pack_categories.setdefault("routines", {})[full] = routine_meta

        # Maintenance owners that slipped past the SQL-side NOT IN (APEX
        # releases, resurrected legacy accounts) are dropped before the
        # authoritative replacement — they are not documentation.
        if engine == "oracle":
            for cat in pack_categories:
                pack_categories[cat] = {
                    f: m for f, m in pack_categories[cat].items()
                    if _ora_user_owner(str(m.get("schema") or ""))
                }
            pack_matviews = {
                f: m for f, m in pack_matviews.items()
                if _ora_user_owner(str(m.get("schema") or ""))
            }

        # Authoritative replacement; matviews merge AFTERWARDS so the fresh
        # views dict cannot wipe them.
        for cat, entries in pack_categories.items():
            categories[cat] = entries
            tools_used[cat] = getattr(sql_tool, "name", "")
        for full, meta in pack_matviews.items():
            categories["views"].setdefault(full, meta)
        unavailable = [u for u in unavailable if u not in pack_categories]

    # Per-object source fetch (oracle get_object_source & friends), capped —
    # runs AFTER the pack so pack-added name-only objects get sources too.
    source_tool = roles["source"][0] if roles.get("source") else None
    if source_tool is not None:
        tools_used["source"] = getattr(source_tool, "name", "")
        # oracle-mcp-server resolves object names against ITS effective
        # schema (DBMS_METADATA answers ORA-31603 for everything else; the
        # tool has no owner argument). After a cross-schema bulk walk only
        # that schema's objects can resolve — burning the source budget on
        # the rest produced "sources" that were really error texts. When
        # the schema cannot even be determined, skip oracle sources rather
        # than gamble the budget (adapter path: objects carry no schema and
        # the server is single-schema anyway — fetch as before).
        effective: Optional[str] = None
        skip_sources = False
        if bulk is not None and engine == "oracle" and sql_tool is not None:
            effective = await _oracle_session_schema(sql_tool, _bounded_call)
            skip_sources = effective is None
        if not skip_sources:
            budget = _source_objects()
            declared = _tool_arg_names(source_tool)
            for cat in _CATEGORIES:
                if budget <= 0:
                    break
                for full, meta in categories[cat].items():
                    if budget <= 0:
                        break
                    if meta.get("source"):
                        continue
                    if (
                        effective is not None
                        and str(meta.get("schema") or "").upper() != effective
                    ):
                        continue  # would only buy an ORA-31603 from the server
                    args = _build_tool_args(source_tool, table=meta.get("name") or full)
                    if "object_type" in declared:
                        args["object_type"] = (
                            meta.get("kind")
                            if cat == "routines"
                            else _SOURCE_TYPE_BY_CATEGORY.get(cat, "")
                        )
                    raw = await _bounded_call(source_tool, args)
                    budget -= 1
                    if (
                        raw
                        and not raw.startswith("ERROR:")
                        and not _looks_like_server_error(_unwrap_untrusted(raw))
                    ):
                        meta["source"] = _cap(raw, MAX_DEFINITION_CHARS)

    return {
        "schemas": schema_names,
        "tables": tables,
        "fk_edges": fk_edges,
        **{cat: categories[cat] for cat in _CATEGORIES},
        "tools_used": tools_used,
        "unavailable": unavailable,
    }


# --------------------------------------------------------------------------- #
# Deterministic skeleton (prompt evidence + no-LLM fallback)
# --------------------------------------------------------------------------- #
#: How many list rows stay visible on root pages; the rest go under a
#: <details> disclosure (fold, «не резать, а прятать»).
_TABLES_ROOT_VISIBLE = 60
_RELATIONS_VISIBLE = 80


def _fk_degrees(edges: List[Dict[str, Any]]) -> Counter:
    """FK adjacency degree per table (both directions)."""
    degrees: Counter = Counter()
    for edge in edges or []:
        degrees[edge.get("from") or ""] += 1
        degrees[edge.get("to") or ""] += 1
    return degrees


def _ranked_tables(info: Dict[str, Any]) -> List[str]:
    """Tables ranked FK-degree → column count → name (subpage priority)."""
    tables = info.get("tables") or {}
    degrees = _fk_degrees(info.get("fk_edges") or [])
    return sorted(
        tables,
        key=lambda full: (
            -degrees.get(full, 0),
            -len((tables[full] or {}).get("columns") or []),
            full,
        ),
    )


def _edge_line(edge: Dict[str, Any]) -> str:
    """One rendered relationship line."""
    def _cols(cols: Optional[List[str]]) -> str:
        cols = [c for c in (cols or []) if c]
        return f"({', '.join(cols)})" if cols else ""

    line = "- `{}`{} → `{}`{}".format(
        edge.get("from") or "?",
        _cols(edge.get("from_cols")),
        edge.get("to") or "?",
        _cols(edge.get("to_cols")),
    )
    if edge.get("constraint"):
        line += f" ({edge['constraint']})"
    if edge.get("kind") in ("related", "inferred"):
        line += f" — {edge['kind']}"
    return line


def _mermaid_safe(token: Any) -> str:
    """Entity/label-safe mermaid identifier (erDiagram name charset)."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(token or "")).strip("_")
    return safe or "x"


def _er_mermaid(edges: List[Dict[str, Any]]) -> str:
    """``erDiagram`` body derived from the FK graph (parent-one → child-many)."""
    if not edges:
        return ""
    lines = ["erDiagram"]
    for edge in edges:
        parent = _mermaid_safe(edge.get("to"))
        child = _mermaid_safe(edge.get("from"))
        if not parent or not child:
            continue
        label = _mermaid_safe(edge.get("constraint") or "fk")
        lines.append(f"    {parent} ||--o{{ {child} : {label}")
    return "\n".join(lines)


def _relationships_section(edges: List[Dict[str, Any]], limit: int) -> List[str]:
    """Rendered Relationships block with fold for long edge lists."""
    if not edges:
        return [
            "## Relationships", "",
            "_Introspection reported no explicit foreign keys for this database._",
            "",
        ]
    lines = [_edge_line(e) for e in edges]
    visible, hidden = rank_split(lines, key=lambda r: 0, keep=limit)
    out = ["## Relationships", ""]
    out.extend(visible)
    if hidden:
        out.append("")
        out.extend(fold("Remaining relations", list(hidden), count=len(hidden)))
    out.append("")
    return out


def _er_section(edges: List[Dict[str, Any]]) -> List[str]:
    """Rendered ER Diagram block (mermaid fence) or the empty-evidence note."""
    body = _er_mermaid(edges)
    if not body:
        return [
            "## ER Diagram", "",
            "_No relationships were reported or confidently inferred; "
            "an ER diagram would be speculation._",
            "",
        ]
    return ["## ER Diagram", "", "```mermaid", body, "```", ""]


def _category_root_skeletons(info: Dict[str, Any]) -> List[str]:
    """Compact category listings for the prompt skeleton (names only)."""
    out: List[str] = []
    for cat in _CATEGORIES:
        entries = info.get(cat) or {}
        if not entries:
            continue
        out.append(f"## {_CATEGORY_TITLES[cat]} ({len(entries)})")
        for full in sorted(entries):
            out.append(f"- `{full}`")
        out.append("")
    return out


def _render_skeleton(entity: Any, info: Dict[str, Any]) -> Tuple[str, str]:
    """Render the deterministic (overview facts, tables root) skeleton pair.

    The overview carries the FACT block (masked DSN, schemas, counts); the
    tables root carries the ranked table list, the FK relationships and the
    derived ER diagram — pure introspection evidence, no LLM text. Used as
    the overview prompt's ground truth and the no-LLM fallback content.
    """
    name = getattr(entity, "name", None) or "Database"
    dsn_masked = getattr(entity, "dsn_masked", None) or ""
    schemas = info.get("schemas") or []
    tables: Dict[str, Any] = info.get("tables") or {}
    edges = info.get("fk_edges") or []

    overview: List[str] = [f"# Database: {name}", ""]
    if dsn_masked:
        overview.append(f"**Connection (masked):** `{dsn_masked}`")
        overview.append("")
    overview.append(f"**Schemas:** {', '.join(schemas) if schemas else '(default)'}")
    overview.append(f"**Tables introspected:** {len(tables)}")
    overview.append(f"**Foreign keys:** {len(edges)}")
    for cat in _CATEGORIES:
        entries = info.get(cat) or {}
        if entries:
            overview.append(f"**{_CATEGORY_TITLES[cat]}:** {len(entries)}")
    overview.append("")

    rows = [f"- `{full}`" for full in _ranked_tables(info)]
    visible, hidden = rank_split(rows, key=lambda r: 0, keep=_TABLES_ROOT_VISIBLE)
    tables_md: List[str] = ["## Tables", "", f"{len(tables)} table(s)", ""]
    tables_md.extend(visible)
    if hidden:
        tables_md.append("")
        tables_md.extend(fold("Remaining tables", list(hidden), count=len(hidden)))
    tables_md.append("")
    tables_md.extend(_relationships_section(edges, _RELATIONS_VISIBLE))
    tables_md.extend(_er_section(edges))
    return "\n".join(overview), "\n".join(tables_md)


def _schema_dump(info: Dict[str, Any]) -> str:
    """Raw evidence blob for the LLM/judge (JSON, capped)."""
    try:
        dump = json.dumps(info, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        dump = str(info)
    return _cap(dump, MAX_SCHEMA_DUMP_CHARS)


# --------------------------------------------------------------------------- #
# Page tree rendering (root pages + per-entity subpages)
# --------------------------------------------------------------------------- #
def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (name or "").strip()).strip("_").lower()
    return (slug or "obj")[:80]


class _Slugger:
    """De-duplicating slug generator (stable page ids per run)."""

    def __init__(self) -> None:
        self._seen: set = set()

    def __call__(self, name: str) -> str:
        base = _slugify(name)
        slug, n = base, 1
        while slug in self._seen:
            n += 1
            slug = f"{base}_{n}"
        self._seen.add(slug)
        return slug


def _render_structure_table(columns: List[Dict[str, Any]]) -> List[str]:
    out = ["| column | type | null | default |", "| --- | --- | --- | --- |"]
    for c in columns:
        default = c.get("default")
        out.append("| {} | {} | {} | {} |".format(
            c.get("name") or "?",
            c.get("type") or "?",
            "YES" if c.get("nullable") else "NO",
            "-" if default is None else str(default),
        ))
    return out


def _render_indexes_block(indexes: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for i in indexes:
        cols = ", ".join(c for c in (i.get("columns") or []) if c)
        flags = " ".join(
            f for f in ("UNIQUE" if i.get("unique") else "",
                        "PRIMARY" if i.get("primary") else "") if f
        )
        line = f"- `{i.get('name') or '?'}`" + (f" ({cols})" if cols else "")
        if flags:
            line += f" {flags}"
        if i.get("ddl"):
            line += f" — {i['ddl']}"
        out.append(line)
    return out


def _render_table_subpage(
    full: str,
    meta: Dict[str, Any],
    desc: Dict[str, Any],
    edges: List[Dict[str, Any]],
    triggers: Dict[str, Dict[str, Any]],
) -> str:
    """One table child page: description + structure + evidence blocks."""
    out: List[str] = [f"## `{full}`", ""]
    purpose = (desc or {}).get("purpose") or meta.get("comment") or ""
    if purpose:
        out.extend([str(purpose), ""])
    else:
        out.extend([
            "_(no description available: the enrichment stage returned no "
            "text for this table; see Structure/DDL for the raw evidence)_", "",
        ])
    notes = (desc or {}).get("notes")
    if notes:
        out.extend(["## Notes", "", str(notes), ""])

    columns = meta.get("columns") or []
    if columns:
        out.extend(["## Structure", "", *_render_structure_table(columns), ""])
    indexes = meta.get("indexes") or []
    if indexes:
        out.extend(["## Indexes", "", *_render_indexes_block(indexes), ""])
    constraints = meta.get("constraints") or []
    if constraints:
        out.extend([
            "## Constraints", "",
            "| name | type | columns |", "| --- | --- | --- |",
        ])
        for c in constraints:
            cols = ", ".join(x for x in (c.get("columns") or []) if x)
            out.append(f"| {c.get('name') or '?'} | {c.get('type') or '?'} | {cols} |")
        out.append("")

    touching = [e for e in edges if e.get("from") == full or e.get("to") == full]
    if touching:
        out.extend(["## Relations", ""])
        out.extend(_edge_line(e) for e in touching)
        out.append("")

    bare = (meta.get("table") or "").lower()
    table_triggers = [
        t for t, tmeta in sorted(triggers.items())
        if isinstance(tmeta, dict)
        and str((tmeta.get("meta") or {}).get("table") or "").lower()
        in (bare, full.lower())
    ]
    if table_triggers:
        out.extend(["## Triggers", ""])
        out.extend(f"- `{t}`" for t in table_triggers)
        out.append("")

    definition = (meta.get("definition") or "").strip()
    if definition and not definition.startswith("ERROR:"):
        out.extend(["## DDL", "", "```sql", definition, "```", ""])
    elif definition.startswith("ERROR:"):
        out.extend([f"_{definition}_", ""])
    return "\n".join(out).strip()


def _render_tables_root(
    info: Dict[str, Any],
    descriptions: Dict[str, Dict[str, Any]],
    table_page_ids: Dict[str, str],
) -> str:
    """Final Tables root page: descriptions + relationships + ER diagram."""
    tables: Dict[str, Any] = info.get("tables") or {}
    edges = info.get("fk_edges") or []
    rows: List[str] = []
    for full in _ranked_tables(info):
        purpose = (descriptions.get(full) or {}).get("purpose")
        row = (
            f"- [`{full}`]({table_page_ids[full]})"
            if full in table_page_ids else f"- `{full}`"
        )
        if purpose:
            row += f" — {purpose}"
        rows.append(row)
    visible, hidden = rank_split(rows, key=lambda r: 0, keep=_TABLES_ROOT_VISIBLE)
    out: List[str] = ["## Tables", "", f"{len(tables)} table(s)", ""]
    out.extend(visible)
    if hidden:
        out.append("")
        out.extend(fold("Remaining tables", list(hidden), count=len(hidden)))
    out.append("")
    out.extend(_relationships_section(edges, _RELATIONS_VISIBLE))
    out.extend(_er_section(edges))
    return "\n".join(out).strip()


def _render_category_root(
    cat: str,
    entries: Dict[str, Dict[str, Any]],
    descriptions: Dict[str, str],
    visible_limit: int = 40,
) -> str:
    title = _CATEGORY_TITLES[cat]
    rows: List[str] = []
    for full in sorted(entries):
        purpose = descriptions.get(full)
        rows.append(f"- `{full}`" + (f" — {purpose}" if purpose else ""))
    visible, hidden = rank_split(rows, key=lambda r: 0, keep=visible_limit)
    out: List[str] = [f"## {title}", "", f"{len(entries)} object(s)", ""]
    out.extend(visible)
    if hidden:
        out.append("")
        out.extend(fold(f"Remaining {title.lower()}", list(hidden), count=len(hidden)))
    out.append("")
    return "\n".join(out).strip()


def _render_category_subpage(
    cat: str, full: str, meta: Dict[str, Any], purpose: str
) -> str:
    out: List[str] = [f"## `{full}`", ""]
    if purpose:
        out.extend([purpose, ""])
    else:
        kind = meta.get("kind") or cat
        out.extend([f"_(no description available; {kind} evidence below)_", ""])
    flat_meta = {
        **(meta.get("meta") or {}),
        **({"kind": meta["kind"]} if meta.get("kind") else {}),
    }
    if flat_meta:
        out.extend(["## Metadata", "", "| key | value |", "| --- | --- |"])
        for key in sorted(flat_meta):
            out.append(f"| {key} | {str(flat_meta[key])[:300]} |")
        out.append("")
    source = (meta.get("source") or "").strip()
    if source and not source.startswith("ERROR:"):
        out.extend(["## Source", "", "```sql", source, "```", ""])
    return "\n".join(out).strip()


def _render_page_tree(
    entity: Any,
    info: Dict[str, Any],
    enrich: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Build the full pages dict (viewer contract) + the assembly order.

    ``enrich`` carries the LLM outputs: ``overview`` (text or None),
    ``tables`` ({full: {purpose, notes, related}}) and ``categories``
    ({cat: {full: purpose}}). Inferred edges are merged into
    ``info["fk_edges"]`` by the caller BEFORE this runs.
    """
    llm_overview = enrich.get("overview")
    descriptions: Dict[str, Dict[str, Any]] = enrich.get("tables") or {}
    cat_descs: Dict[str, Dict[str, str]] = enrich.get("categories") or {}
    tables: Dict[str, Any] = info.get("tables") or {}
    edges = info.get("fk_edges") or []
    triggers = info.get("triggers") or {}
    slugger = _Slugger()

    pages: Dict[str, Any] = {}
    order: List[str] = ["page_overview"]

    facts, _tables_skeleton = _render_skeleton(entity, info)
    overview_parts = [facts]
    if llm_overview:
        overview_parts.append(str(llm_overview))
    else:
        overview_parts.append(
            "_(LLM enrichment unavailable — deterministic overview only)_"
        )
    pages["page_overview"] = {
        "id": "page_overview",
        "title": "Overview",
        "content": "\n\n".join(overview_parts).strip(),
        "filePaths": [],
        "importance": "high",
        "relatedPages": ["page_tables"],
    }

    # Table subpages: ranked, capped; surplus stays as folded root rows.
    cap = _subpage_cap()
    ranked = _ranked_tables(info)
    subpaged = ranked[:cap]
    table_page_ids = {full: f"page_tbl_{slugger(full)}" for full in subpaged}
    pages["page_tables"] = {
        "id": "page_tables",
        "title": "Tables",
        "content": _render_tables_root(info, descriptions, table_page_ids),
        "filePaths": [],
        "importance": "high",
        "relatedPages": ["page_overview"],
    }
    order.append("page_tables")

    neighbors: Dict[str, set] = {full: set() for full in subpaged}
    for edge in edges:
        frm, to = edge.get("from"), edge.get("to")
        if frm in table_page_ids and to in table_page_ids:
            neighbors[frm].add(table_page_ids[to])
            neighbors[to].add(table_page_ids[frm])
    for full in subpaged:
        meta = tables.get(full) or {}
        pages[table_page_ids[full]] = {
            "id": table_page_ids[full],
            # Schema-qualified: with hundreds of same-named tables across
            # schemas, a bare "USERS" is ambiguous in the nav and search.
            "title": full,
            "content": _render_table_subpage(
                full, meta, descriptions.get(full) or {}, edges, triggers
            ),
            "parent": "page_tables",
            "filePaths": [],
            "importance": "medium",
            "relatedPages": sorted(neighbors.get(full) or ()),
        }
        order.append(table_page_ids[full])

    # Category roots + children (only with evidence), sharing the same cap.
    remaining = max(0, cap - len(subpaged))
    for cat in _CATEGORIES:
        entries = info.get(cat) or {}
        if not entries:
            continue
        root_id = f"page_{cat}"
        pages[root_id] = {
            "id": root_id,
            "title": _CATEGORY_TITLES[cat],
            "content": _render_category_root(cat, entries, cat_descs.get(cat) or {}),
            "filePaths": [],
            "importance": "medium",
            "relatedPages": ["page_tables"],
        }
        order.append(root_id)
        if remaining <= 0:
            continue
        children = sorted(entries)[:remaining]
        remaining -= len(children)
        prefix = _CATEGORY_PREFIX[cat]
        child_ids = {full: f"page_{prefix}_{slugger(full)}" for full in children}
        for full in children:
            meta = entries[full] or {}
            pages[child_ids[full]] = {
                "id": child_ids[full],
                "title": full,  # schema-qualified, same as table subpages
                "content": _render_category_subpage(
                    cat, full, meta, (cat_descs.get(cat) or {}).get(full, "")
                ),
                "parent": root_id,
                "filePaths": [],
                "importance": "low",
                "relatedPages": [],
            }
            order.append(child_ids[full])
        pages[root_id]["relatedPages"] = sorted(child_ids.values())
    return pages, order


def _assemble_docs(pages: Dict[str, Any], order: List[str]) -> str:
    """Canonical ``generated_docs``: overview → tables root → children → categories."""
    parts = []
    for page_id in order:
        content = ((pages.get(page_id) or {}).get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Enrichment prompts (externalized in refs/prompts/<lang>/database_*.md)
# --------------------------------------------------------------------------- #
_DATABASE_DOC_FALLBACK = (
    "You are a database reverse-engineering expert. Write the OVERVIEW PAGE "
    "of the database `{database_name}` for engineers, based EXCLUSIVELY on "
    "the schema evidence collected from its introspection tools.\n\n"
    "<masked_connection>\n{dsn_masked}\n</masked_connection>\n\n"
    "<deterministic_skeleton>\n{skeleton}\n</deterministic_skeleton>\n\n"
    "<schema_evidence>\n{schema_dump}\n</schema_evidence>\n\n"
    "<product_context>\n{product_context}\n</product_context>\n\n"
    "Write Markdown with exactly three sections: an Overview (purpose, "
    "overall shape, workload character), Schema layout (namespaces and what "
    "each contains), and Design notes (key styles, indexing policy, "
    "denormalization, audit columns, naming conventions). Do NOT invent "
    "tables, columns or constraints absent from the evidence; mark "
    "inferences explicitly. Write in {language_name} (technical terms in "
    "English). Your FINAL message must contain ONLY the finished document."
)


def build_database_doc_prompt(
    *,
    database_name: str,
    dsn_masked: str,
    skeleton: str,
    schema_dump: str,
    language: str = "ru",
    product_context: str = "",
    reviewer_notes: Optional[str] = None,
) -> str:
    """Build the overview enrichment prompt; ``language`` selects the OUTPUT
    language (per-request, mirrors the spec flow). ``reviewer_notes`` carries
    the judge issues stored on the previous page version (per-page regen)."""
    template = load_prompt_file("database_doc.md", _DATABASE_DOC_FALLBACK)
    for var, value in (
        ("database_name", database_name),
        ("dsn_masked", dsn_masked or "(not provided)"),
        ("skeleton", skeleton or "(unavailable)"),
        ("schema_dump", schema_dump or "(unavailable)"),
        ("product_context", product_context or "(no product context available)"),
        ("language_name", LANGUAGE_NAMES.get(language, language)),
    ):
        template = template.replace("{" + var + "}", str(value))
    if reviewer_notes:
        template += (
            "\n\n<reviewer_notes>\nПредыдущая версия страницы получила "
            "замечания проверяющего — обязательно исправь их:\n"
            + reviewer_notes + "\n</reviewer_notes>"
        )
    return template


_DATABASE_TABLES_FALLBACK = (
    "You are a database documentation expert. For every table in the batch "
    "below write a short description (1-2 sentences), based ONLY on the "
    "provided evidence (columns, keys, relations, comment, product context).\n\n"
    "<table_batch>\n{table_batch}\n</table_batch>\n\n"
    "<product_context>\n{product_context}\n</product_context>\n\n"
    'Return ONLY a JSON array — no prose, no fences: '
    '[{"name": str, "purpose": str, "notes": str, "related": [str]}]. '
    "Take table names EXACTLY from the batch; do not invent tables. "
    "Write purposes in {language_name}."
)


def build_database_tables_prompt(
    *,
    table_batch: str,
    product_context: str = "",
    language: str = "ru",
) -> str:
    """Build one strict-JSON table-description batch prompt."""
    template = load_prompt_file("database_tables.md", _DATABASE_TABLES_FALLBACK)
    for var, value in (
        ("table_batch", table_batch or "(empty)"),
        ("product_context", product_context or "(no product context available)"),
        ("language_name", LANGUAGE_NAMES.get(language, language)),
    ):
        template = template.replace("{" + var + "}", str(value))
    return template


_DATABASE_CATEGORIES_FALLBACK = (
    'You are a database documentation expert. For every object of the '
    'category "{category_title}" below write a short description (1-2 '
    'sentences), based ONLY on the provided evidence (definition/source, '
    'metadata, product context).\n\n'
    "<objects>\n{objects}\n</objects>\n\n"
    "<product_context>\n{product_context}\n</product_context>\n\n"
    'Return ONLY a JSON array — no prose, no fences: '
    '[{"name": str, "purpose": str}]. Take object names EXACTLY from the '
    "list; do not invent objects. Write purposes in {language_name}."
)


def build_database_categories_prompt(
    *,
    category_title: str,
    objects: str,
    product_context: str = "",
    language: str = "ru",
) -> str:
    """Build the strict-JSON category-objects description prompt."""
    template = load_prompt_file(
        "database_categories.md", _DATABASE_CATEGORIES_FALLBACK
    )
    for var, value in (
        ("category_title", category_title),
        ("objects", objects or "(empty)"),
        ("product_context", product_context or "(no product context available)"),
        ("language_name", LANGUAGE_NAMES.get(language, language)),
    ):
        template = template.replace("{" + var + "}", str(value))
    return template


_DATABASE_RELATIONS_FALLBACK = (
    "You are a database reverse-engineering expert. Introspection found NO "
    "explicit foreign keys, but column names can suggest relations. For the "
    "tables below infer the MOST OBVIOUS many-to-one relations.\n\n"
    "<table_batch>\n{table_batch}\n</table_batch>\n\n"
    'Return ONLY a JSON array — no prose, no fences: '
    '[{"from": str, "from_cols": [str], "to": str, "to_cols": [str]}]. '
    "Every relation is an INFERENCE and will be marked as such. Take names "
    "EXACTLY from the evidence; an empty array [] is valid. "
    "Write in {language_name}."
)


def build_database_relations_prompt(
    *,
    table_batch: str,
    language: str = "ru",
) -> str:
    """Build the inferred-relations prompt (only used with zero FK edges)."""
    template = load_prompt_file(
        "database_relations.md", _DATABASE_RELATIONS_FALLBACK
    )
    for var, value in (
        ("table_batch", table_batch or "(empty)"),
        ("language_name", LANGUAGE_NAMES.get(language, language)),
    ):
        template = template.replace("{" + var + "}", str(value))
    return template


# --------------------------------------------------------------------------- #
# Cross-context (semantic memory ↔ DB docs ↔ codebase docs)
# --------------------------------------------------------------------------- #
async def _product_knowledge_context(product_id: str) -> str:
    """Recall product knowledge (codebase docs/specs/Confluence) for prompts.

    Attached as SUPPLEMENTARY context (never schema evidence): the DB
    descriptions gain domain meaning, grounding stays introspection-only.
    Non-fatal: any failure degrades to an empty block.
    """
    if not product_id:
        return ""
    try:
        from api.config.timeout import resolve_memory_query_timeout
        from api.memory import query_memory

        text = await asyncio.wait_for(
            query_memory(
                "database schema tables columns foreign keys views triggers "
                "procedures product architecture domain",
                product_id,
                top_k=12,
            ),
            timeout=max(5.0, resolve_memory_query_timeout()),
        )
    except Exception as e:  # pragma: no cover - context is never fatal
        logger.debug("product knowledge context unavailable: %s", e)
        return ""
    text = (text or "").strip()
    return _cap(text, _MAX_PRODUCT_CONTEXT_CHARS) if text else ""


def _db_context_payload(info: Dict[str, Any]) -> Dict[str, Any]:
    """Compact digest stored in the Tables page provenance for codebase docs."""
    tables = info.get("tables") or {}
    edges = info.get("fk_edges") or []
    degrees = _fk_degrees(edges)
    ranked = _ranked_tables(info)[:40]
    return {
        "schemas": list(info.get("schemas") or []),
        "tables": [[full, degrees.get(full, 0)] for full in ranked],
        "counts": {
            "tables": len(tables),
            "fk_edges": len(edges),
            **{
                cat: len(info.get(cat) or {})
                for cat in _CATEGORIES if info.get(cat)
            },
        },
    }


def db_context_enabled() -> bool:
    """Cross-context toggle: inject the DB digest into codebase docgen briefs.

    Resolves through the timeout registry (admin store > env
    ``DOCGEN_DB_CONTEXT_ENABLED`` > default on); the legacy env
    truthy-string semantics are preserved by ``resolve_timeout_bool``.
    """
    from api.config.timeout import resolve_docgen_db_context_enabled

    return resolve_docgen_db_context_enabled()


def product_database_context(product_id: str, max_chars: int = 4000) -> str:
    """Compact Russian markdown digest of the product's documented databases.

    Sync + best-effort (called from the codebase flow's brief builder): reads
    the stored ``page_tables`` provenance ``db_context`` digest of every
    database artifact of the product, falling back to a regex over
    ``generated_docs`` for legacy rows. Explicitly marked as supplementary —
    NOT repository paths — so writers (and the citation guard) never treat
    these identifiers as file citations.
    """
    if not product_id:
        return ""
    try:
        from api.db import SessionLocal
        from api.models import DatabaseORM

        session = SessionLocal()
        try:
            rows = (
                session.query(DatabaseORM)
                .filter(DatabaseORM.product_id == product_id)
                .order_by(DatabaseORM.id)
                .all()
            )
        finally:
            session.close()
    except Exception as e:  # pragma: no cover - context is never fatal
        logger.debug("product_database_context unavailable: %s", e)
        return ""
    if not rows:
        return ""

    blocks: List[str] = []
    for row in rows:
        ctx = None
        pages = getattr(row, "pages", None)
        if isinstance(pages, dict):
            prov = ((pages.get("page_tables") or {}).get("provenance") or {})
            ctx = prov.get("db_context") if isinstance(prov, dict) else None
        if isinstance(ctx, dict):
            schemas = ", ".join(str(s) for s in (ctx.get("schemas") or [])) or "—"
            counts = (ctx.get("counts") or {}).get("tables", "?")
            top = ", ".join(
                f"`{name}`" for name, _deg in (ctx.get("tables") or [])[:15]
            )
            piece = (
                f"**{row.name}** (schemas: {schemas}; таблиц: {counts})"
                + (f" — ключевые таблицы: {top}" if top else "")
            )
            blocks.append(piece)
            continue
        docs = getattr(row, "generated_docs", None) or ""
        names = re.findall(r"^- `([^`\n]+)`", docs, re.MULTILINE)[:15]
        if names:
            blocks.append(
                f"**{row.name}** — таблицы: " + ", ".join(f"`{n}`" for n in names)
            )
    if not blocks:
        return ""
    text = (
        "### Контекст баз данных продукта (дополнительно — НЕ файлы "
        "репозитория; не цитировать как пути)\n"
        + "\n".join(f"- {b}" for b in blocks)
    )
    return _cap(text, max_chars)


# --------------------------------------------------------------------------- #
# Batched LLM enrichment
# --------------------------------------------------------------------------- #
def _table_stub(full: str, meta: Dict[str, Any], edge_lines: List[str]) -> str:
    """Evidence stub for one table inside a description batch prompt."""
    lines = [f"### `{full}`"]
    if meta.get("comment"):
        lines.append(f"comment: {meta['comment']}")
    if isinstance(meta.get("row_count"), int):
        lines.append(f"row_count: ~{meta['row_count']}")
    columns = meta.get("columns") or []
    if columns:
        lines.append("columns: " + ", ".join(
            f"{c.get('name')} ({c.get('type')})" for c in columns[:30]
        ))
    else:
        head = re.sub(r"\s+", " ", (meta.get("definition") or ""))[:600]
        if head:
            lines.append(f"definition: {head}")
    if edge_lines:
        lines.append("foreign keys: " + "; ".join(edge_lines))
    return "\n".join(lines)


def _table_edge_lines(full: str, edges: List[Dict[str, Any]]) -> List[str]:
    out = []
    for edge in edges:
        if edge.get("from") == full:
            out.append(
                f"{full}.{','.join(c for c in edge.get('from_cols') or [])} -> "
                f"{edge.get('to')}.{','.join(c for c in edge.get('to_cols') or [])}"
            )
        elif edge.get("to") == full:
            out.append(
                f"{edge.get('from')}.{','.join(c for c in edge.get('from_cols') or [])} -> "
                f"{full}.{','.join(c for c in edge.get('to_cols') or [])}"
            )
    return out[:20]


def _match_table_name(
    name: Any, tables: Dict[str, Any], lower_map: Dict[str, str]
) -> Optional[str]:
    """Validate an LLM-returned table name against the introspected set."""
    if not isinstance(name, str) or not name.strip():
        return None
    candidate = name.strip()
    if candidate in tables:
        return candidate
    low = candidate.lower()
    if low in lower_map:
        return lower_map[low]
    for full in tables:
        if full.lower() == low or full.lower().endswith("." + low):
            return full
    return None


async def _enrich_table_descriptions(
    info: Dict[str, Any],
    *,
    product_context: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    language: str,
    progress: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """Table descriptions via strict-JSON batches (admin-tunable size)."""
    tables: Dict[str, Any] = info.get("tables") or {}
    edges = info.get("fk_edges") or []
    lower_map = {full.lower(): full for full in tables}
    ranked = _ranked_tables(info)[:_max_descriptions()]
    if not ranked:
        return {}
    batch_size = _enrich_batch_size()
    descriptions: Dict[str, Dict[str, Any]] = {}
    edge_lines = {full: _table_edge_lines(full, edges) for full in ranked}
    for start in range(0, len(ranked), batch_size):
        batch = ranked[start:start + batch_size]
        stub = "\n\n".join(
            _table_stub(full, tables[full], edge_lines.get(full) or [])
            for full in batch
        )
        emit_progress(progress, phase="sections", current_section="descriptions")
        prompt = build_database_tables_prompt(
            table_batch=stub,
            product_context=product_context,
            language=language,
        )
        text = await _llm_or_none(
            prompt, model, base_url=base_url, api_key=api_key
        )
        rows = _parse_json_array(text)
        if not rows:
            logger.warning(
                "database docgen: description batch %d..%d returned no valid "
                "JSON; tables keep their deterministic render",
                start, start + len(batch),
            )
            continue
        for row in rows:
            full = _match_table_name(row.get("name"), tables, lower_map)
            if full is None:
                continue
            entry: Dict[str, Any] = {}
            if isinstance(row.get("purpose"), str) and row["purpose"].strip():
                entry["purpose"] = row["purpose"].strip()
            if isinstance(row.get("notes"), str) and row["notes"].strip():
                entry["notes"] = row["notes"].strip()
            related_raw = row.get("related")
            if isinstance(related_raw, list):
                related = [
                    r for r in (
                        _match_table_name(x, tables, lower_map)
                        for x in related_raw if isinstance(x, str)
                    ) if r and r != full
                ]
                if related:
                    entry["related"] = related[:10]
            if entry:
                descriptions[full] = entry
    return descriptions


def _category_stub(full: str, meta: Dict[str, Any]) -> str:
    lines = [f"### `{full}`"]
    if meta.get("kind"):
        lines.append(f"kind: {meta['kind']}")
    for key in sorted(meta.get("meta") or {}):
        value = str(meta["meta"][key])
        if value:
            lines.append(f"{key}: {value[:200]}")
    source = (meta.get("source") or "").strip()
    if source:
        head = re.sub(r"\s+", " ", source)[:800]
        lines.append(f"source: {head}")
    return "\n".join(lines)


async def _enrich_categories(
    info: Dict[str, Any],
    *,
    product_context: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    language: str,
    already_described: int,
) -> Dict[str, Dict[str, str]]:
    """One strict-JSON call per non-empty category (within the description cap)."""
    budget = max(0, _max_descriptions() - already_described)
    out: Dict[str, Dict[str, str]] = {}
    if budget <= 0:
        return out
    for cat in _CATEGORIES:
        entries = info.get(cat) or {}
        if not entries:
            continue
        selected = sorted(entries)[:budget]
        budget -= len(selected)
        stub = "\n\n".join(
            _category_stub(full, entries[full]) for full in selected
        )
        prompt = build_database_categories_prompt(
            category_title=_CATEGORY_TITLES[cat],
            objects=stub,
            product_context=product_context,
            language=language,
        )
        text = await _llm_or_none(
            prompt, model, base_url=base_url, api_key=api_key
        )
        rows = _parse_json_array(text)
        if not rows:
            logger.warning(
                "database docgen: category %s returned no valid JSON; "
                "objects keep their deterministic render", cat,
            )
            continue
        lower_map = {full.lower(): full for full in selected}
        cat_descs: Dict[str, str] = {}
        for row in rows:
            name = row.get("name")
            if not isinstance(name, str):
                continue
            full = lower_map.get(name.strip().lower())
            if full is None:
                continue
            if isinstance(row.get("purpose"), str) and row["purpose"].strip():
                cat_descs[full] = row["purpose"].strip()
        if cat_descs:
            out[cat] = cat_descs
        if budget <= 0:
            break
    return out


async def _infer_relations(
    info: Dict[str, Any],
    *,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    language: str,
) -> List[Dict[str, Any]]:
    """Infer obvious relations by column names — ONLY with zero FK edges.

    Every returned edge is marked ``kind="inferred"`` and rendered with an
    explicit inference marker; endpoints are validated against the tables.
    """
    tables: Dict[str, Any] = info.get("tables") or {}
    if info.get("fk_edges") or len(tables) < 2:
        return []
    ranked = _ranked_tables(info)[:40]
    stub = "\n\n".join(_table_stub(full, tables[full], []) for full in ranked)
    prompt = build_database_relations_prompt(table_batch=stub, language=language)
    text = await _llm_or_none(prompt, model, base_url=base_url, api_key=api_key)
    rows = _parse_json_array(text)
    lower_map = {full.lower(): full for full in tables}
    edges: List[Dict[str, Any]] = []
    for row in rows:
        frm = _match_table_name(row.get("from"), tables, lower_map)
        to = _match_table_name(row.get("to"), tables, lower_map)
        if not frm or not to or frm == to:
            continue

        def _cols(value: Any) -> List[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(v) for v in value if isinstance(v, str) and v]
            return []

        edges.append({
            "from": frm, "from_cols": _cols(row.get("from_cols")),
            "to": to, "to_cols": _cols(row.get("to_cols")),
            "constraint": "", "kind": "inferred",
        })
    return edges


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def _schema_fingerprint(info: Dict[str, Any]) -> str:
    """Stable fingerprint of the introspected payload (sha256[:16])."""
    try:
        blob = json.dumps(info, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        blob = str(info)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
async def generate_database_docs(
    entity: Any,
    product: Any,
    model: Optional[str] = None,
    language: str = "ru",
    progress: Optional[Any] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    force_pages: Optional[List[str]] = None,
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

    # Per-page regeneration: only the forced pages are swapped in at persist;
    # the overview's stored judge issues ride into its prompt as notes.
    old_pages = entity.pages if isinstance(getattr(entity, "pages", None), dict) else {}
    reviewer_notes = ""
    if force_pages and "page_overview" in force_pages:
        issues = (
            ((old_pages.get("page_overview") or {}).get("provenance") or {})
            .get("judge", {})
            .get("issues")
            or []
        )
        reviewer_notes = "\n".join(f"- {i}" for i in issues if str(i).strip())

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
                _introspect(roles, engine=_detect_engine(entity, tools)),
                timeout=INTROSPECTION_TIMEOUT_SECONDS,
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

    # Cancellation checkpoint (post-introspection / pre-enrichment).
    _check_cancel(should_cancel)

    # 3) Deterministic skeleton (overview facts + tables root evidence).
    overview_md, tables_md = _render_skeleton(entity, info)
    skeleton = "\n\n".join(
        part for part in (
            overview_md, tables_md, *_category_root_skeletons(info)
        ) if part
    )
    dump = _schema_dump(info)

    # 4) Batched LLM enrichment (standard docgen path; empty → skeleton).
    r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    model = model or r_model
    product_context = await _product_knowledge_context(product_id)
    emit_progress(progress, phase="sections", current_section="overview")
    llm_overview = await _llm_or_none(
        build_database_doc_prompt(
            database_name=name,
            dsn_masked=dsn_masked,
            skeleton=skeleton,
            schema_dump=dump,
            language=language,
            product_context=product_context,
            reviewer_notes=reviewer_notes or None,
        ),
        model,
        base_url=r_base_url,
        api_key=r_api_key,
    )
    enrich_source = "standard-llm" if llm_overview else "skeleton"
    descriptions: Dict[str, Dict[str, Any]] = {}
    cat_descs: Dict[str, Dict[str, str]] = {}
    if llm_overview:
        # The overview worked — the batches are worth their tokens. A dead
        # LLM after a good overview simply leaves the deterministic render.
        try:
            descriptions = await _enrich_table_descriptions(
                info,
                product_context=product_context,
                model=model,
                base_url=r_base_url,
                api_key=r_api_key,
                language=language,
                progress=progress,
            )
        except Exception as e:  # pragma: no cover - enrichment is never fatal
            logger.warning("database docgen: table description batches failed: %s", e)
        try:
            cat_descs = await _enrich_categories(
                info,
                product_context=product_context,
                model=model,
                base_url=r_base_url,
                api_key=r_api_key,
                language=language,
                already_described=len(descriptions),
            )
        except Exception as e:  # pragma: no cover - enrichment is never fatal
            logger.warning("database docgen: category enrichment failed: %s", e)
        try:
            inferred = await _infer_relations(
                info,
                model=model,
                base_url=r_base_url,
                api_key=r_api_key,
                language=language,
            )
        except Exception as e:  # pragma: no cover - inference is never fatal
            logger.warning("database docgen: relation inference failed: %s", e)
            inferred = []
        if inferred:
            info.setdefault("fk_edges", []).extend(inferred)
            logger.info(
                "database docgen: %d relation(s) inferred by column-name "
                "evidence (marked as inferences)", len(inferred),
            )

    # Cancellation checkpoint (post-enrichment).
    _check_cancel(should_cancel)

    # 5) Page tree (viewer contract) from evidence + enrichment.
    pages, order = _render_page_tree(
        entity, info,
        {"overview": llm_overview, "tables": descriptions, "categories": cat_descs},
    )

    # 6) Guard: mermaid repair → secret masking → corroborate → judge.
    emit_progress(progress, phase="verifying")
    repair_llm = None
    try:
        repair_llm = _make_repair_llm(model, base_url=r_base_url, api_key=r_api_key)
        for page_id in order:
            content = (pages[page_id] or {}).get("content") or ""
            if "```mermaid" in content:
                try:
                    content, _mstats = await run_repair_loop(content, repair_llm)
                    pages[page_id]["content"] = content
                except Exception as e:  # pragma: no cover - verifier is non-fatal
                    logger.warning(
                        "Mermaid repair loop failed for database page %s: %s",
                        page_id, e,
                    )
    except Exception as e:  # pragma: no cover - verifier must never break gen
        logger.warning("Mermaid repair stage failed for database doc: %s", e)
    finally:
        await _close_owned_llm(repair_llm)

    masked_per_page: Dict[str, int] = {}
    for page_id in order:
        content = (pages[page_id] or {}).get("content") or ""
        content, findings = mask_secrets(content)
        masked_per_page[page_id] = len(findings)
        pages[page_id]["content"] = content
    total_masked = sum(masked_per_page.values())
    if total_masked:
        logger.warning(
            "database docgen guard: masked %d secret-like value(s) across %d page(s)",
            total_masked, len(order),
        )

    # 3.1: corroborate filter — model-generated sentences naming identifiers
    # that are NOT in the introspected schema (invented tables/columns/
    # enums/functions) are dropped BEFORE the judge, so the judged text and
    # the persisted text agree. Only the LLM overview carries model prose;
    # every other page is deterministic evidence. Fail-open per contract.
    corroborate_removed: List[str] = []
    if llm_overview:
        try:
            grounding = grounding_from_introspection(info)
            if grounding:
                overview_content = pages["page_overview"]["content"]
                filtered, corrob = filter_ungrounded_prose(
                    overview_content, grounding
                )
                if corrob.emptied:
                    logger.warning(
                        "database corroborate filter would empty the overview; "
                        "original kept"
                    )
                elif corrob.touched:
                    pages["page_overview"]["content"] = filtered
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
    if judge_enabled() and llm_overview:
        try:
            judge_verdict = await judge_section(
                "database", pages["page_overview"]["content"],
                skeleton + "\n\n" + dump, model=model,
            )
            if judge_verdict.verdict != "consistent" and judge_verdict.issues:
                logger.warning(
                    "database docgen judge: %s — %s",
                    judge_verdict.verdict,
                    "; ".join(judge_verdict.issues[:3]),
                )
        except Exception as e:  # pragma: no cover - judge must never break gen
            logger.warning("database docgen judge failed: %s", e)

    # 7) Provenance (per page) + the DB-context digest for the codebase flow.
    fingerprint = _schema_fingerprint(info)
    caps = {
        "enrich_batch": _enrich_batch_size(),
        "max_subpages": _subpage_cap(),
        "max_descriptions": _max_descriptions(),
        "descriptions": len(descriptions),
        "fk_evidence_tables": _fk_evidence_tables(),
        "source_objects": _source_objects(),
    }
    prompt_file_by_root: Dict[str, str] = {
        "page_overview": "database_doc.md",
        "page_tables": "database_tables.md",
        **{f"page_{cat}": "database_categories.md" for cat in _CATEGORIES},
    }
    for page_id in order:
        page = pages[page_id]
        is_child = bool(page.get("parent"))
        prompt_file = (
            "introspection" if is_child
            else prompt_file_by_root.get(page_id, "introspection")
        )
        if page_id == "page_overview":
            section_id = "database"
            regen = "legacy-fallback" if enrich_source == "standard-llm" else "skeleton"
        elif page_id == "page_tables" and descriptions:
            section_id = "database_tables"
            regen = "legacy-fallback"
        elif is_child:
            section_id = "database_introspection"
            regen = "skeleton"
        else:
            section_id = f"database_{page_id[len('page_'):]}"
            regen = "skeleton"
        generator = enrich_source if page_id == "page_overview" else (
            "standard-llm" if page_id == "page_tables" and descriptions
            else "introspection"
        )
        prov = build_section_provenance(
            section_id,
            model=model,
            prompt_file=prompt_file,
            source_files=[],
            fingerprint=fingerprint,
            citations={"resolved": [], "unresolved": []},
            judge=judge_verdict if page_id == "page_overview" else None,
            regen=regen,
            secrets_masked=masked_per_page.get(page_id, 0),
            ungrounded=(
                corroborate_removed or None
            ) if page_id == "page_overview" else None,
        )
        prov["generator"] = generator
        prov["tools_used"] = dict(info.get("tools_used") or {})
        prov["schema_fingerprint"] = fingerprint
        prov["schema_fingerprint_source"] = "mcp_introspection"
        prov["introspection_cache"] = cache_state
        prov["caps"] = dict(caps)
        if info.get("unavailable"):
            prov["unavailable"] = list(info["unavailable"])
        page["provenance"] = prov
    pages["page_tables"]["provenance"]["db_context"] = _db_context_payload(info)

    # 8) Persist (viewer contract) + background indexing.
    _check_cancel(should_cancel)  # pre-persist: nothing written after Stop
    if force_pages and old_pages:
        # Per-page regen: keep the stored pages, swap in only the forced ones
        # (with their fresh provenance); old-only pages append at the end.
        merged = dict(old_pages)
        merged.update({p: pages[p] for p in force_pages if p in pages})
        order = list(order) + [p for p in merged if p not in order]
        pages = merged
    _carry_page_verify_flags(pages, old_pages)
    emit_progress(progress, phase="indexing")
    docs = _assemble_docs(pages, order)
    _persist_artifact(entity, docs, pages)
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
    "db_context_enabled",
    "generate_database_docs",
    "preset_adapter_roles",
    "product_database_context",
]
