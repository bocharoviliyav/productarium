"""Admin MCP server registry router (Wave C, contract 1).

Admin-guarded CRUD + health-check/tool-discovery for external MCP servers
(``McpServerORM``). Prefix ``/api/admin/mcp``:

- ``GET    /api/admin/mcp/servers``             — list (secrets masked)
- ``POST   /api/admin/mcp/servers``             — register a server
- ``PUT    /api/admin/mcp/servers/{id}``        — partial update (transport immutable)
- ``DELETE /api/admin/mcp/servers/{id}``        — delete (bindings cascade)
- ``POST   /api/admin/mcp/servers/{id}/test``   — bounded health-check (updates status)
- ``GET    /api/admin/mcp/servers/{id}/tools``  — cached discovery result (no connect)

Security:
- ``headers``/``env`` arrive as PLAINTEXT dicts, are Fernet-encrypted before
  storage (``api/mcp/secrets.py``) and only ever leave as masked key-only
  views (``headers_masked``/``env_masked``) — values never round-trip.
- ``http`` transport requires an ``http(s)`` URL; ``stdio`` requires a
  shell-safe single binary path (no metacharacters, no ``..``, no
  whitespace/``sh -c``) with arguments carried separately in ``args`` —
  PLUS the shared policy from ``api/mcp/policy.py``: shells, interpreters
  and script runners (``sh``, ``python3``, ``node``, ``npx`` …) and
  behavior-shaping env keys (``PATH``, ``LD_*``, ``PYTHONPATH`` …) are
  rejected, because through them the validated ``args`` collapse into
  arbitrary code execution.
- All string inputs are length-capped (Pydantic ``Field`` + manual dict caps).
- Errors from the health-check are short + sanitized; full tracebacks stay in
  the server log.
"""

from __future__ import annotations

import logging
import secrets as pysecrets
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.auth.deps import require_admin
from api.db import get_db
from api.mcp.manager import get_mcp_manager
from api.mcp.policy import stdio_command_error, stdio_env_error
from api.mcp.secrets import (
    MASK,
    decrypt_secret_dict,
    encrypt_secret_dict,
    mask_secret_dict,
)
from api.models import McpServerORM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/mcp", tags=["mcp-admin"])

# --- validation constants -----------------------------------------------------
#: Shell metacharacters forbidden anywhere in a stdio command.
_SHELL_METACHARS = "|;&`$><"
#: Max entries + per-entry sizes for the secrets dicts and args lists.
_MAX_DICT_ENTRIES = 64
_MAX_SECRET_KEY_LEN = 128
_MAX_SECRET_VALUE_LEN = 2048
_MAX_ARGS = 64
_MAX_ARG_LEN = 512
#: Min seconds between two health-checks of the SAME server (anti SSRF-oracle
#: hammering: closed port -> instant error, open port -> timeout; the window
#: makes the oracle expensive to probe with).
_TEST_MIN_INTERVAL = 2.0
#: ``server_id -> monotonic timestamp`` of the last accepted /test call.
_last_test_at: Dict[str, float] = {}


# --- pydantic models ----------------------------------------------------------
class McpServerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    transport: str = Field(min_length=1, max_length=16)
    url: Optional[str] = Field(default=None, max_length=512)
    command: Optional[str] = Field(default=None, max_length=256)
    args: Optional[List[str]] = Field(default=None, max_length=_MAX_ARGS)
    headers: Optional[Dict[str, str]] = None
    env: Optional[Dict[str, str]] = None
    enabled: bool = True


class McpServerUpdate(BaseModel):
    """Partial update. ``transport`` is immutable; a differing value is a 400.

    Secret dicts use explicit-presence semantics: a field ABSENT from the
    request keeps the stored value; a field PRESENT replaces it wholesale
    (empty dict = clear) — with one masked-echo exception: entries whose
    value is the mask ``***`` are DROPPED, and an all-mask payload is a no-op
    (a UI echoing the masked view back must not wipe the stored secrets).

    Changing url/command/args (or replacing headers/env) resets ``status``
    to ``unknown`` — the persisted health verdict belonged to the old config.
    """

    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    transport: Optional[str] = Field(default=None, min_length=1, max_length=16)
    url: Optional[str] = Field(default=None, max_length=512)
    command: Optional[str] = Field(default=None, max_length=256)
    args: Optional[List[str]] = Field(default=None, max_length=_MAX_ARGS)
    headers: Optional[Dict[str, str]] = None
    env: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None


class McpServerOut(BaseModel):
    id: str
    name: str
    transport: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    headers_masked: Optional[Dict[str, str]] = None
    env_masked: Optional[Dict[str, str]] = None
    enabled: bool
    status: str
    status_checked_at: Optional[datetime] = None
    status_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class McpTestResult(BaseModel):
    ok: bool
    detail: Optional[str] = None
    tools: List[Dict[str, Optional[str]]] = []


# --- validation helpers ---------------------------------------------------------
def _validate_url(url: str) -> str:
    """Require a well-formed http(s) URL (trimmed), without embedded userinfo.

    Credentials in the URL (``http://user:pass@host/``) would be persisted
    unencrypted and echoed back in API/UI responses — auth belongs in the
    Fernet-encrypted ``headers``. Private/internal hosts are intentionally
    allowed: MCP servers usually live on the internal network, and this is an
    admin-only, rate-limited endpoint.
    """
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="url must be a valid http(s) URL")
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=400,
            detail="url must not embed credentials (user:pass@host); use headers",
        )
    return url


def _validate_command(command: str) -> str:
    """Require a shell-safe single binary path for the stdio transport.

    The command is passed to the subprocess layer as-is, so it must be a bare
    executable path: no shell metacharacters, no ``..`` path traversal, no
    whitespace (which would smuggle arguments — e.g. ``sh -c ...``), no
    control characters. Arguments belong in ``args``.
    """
    command = (command or "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="command is required for stdio transport")
    policy_error = stdio_command_error(command)
    if policy_error:
        raise HTTPException(status_code=400, detail=policy_error)
    if any(ch in command for ch in _SHELL_METACHARS):
        raise HTTPException(
            status_code=400,
            detail="command must not contain shell metacharacters (|;&`$><)",
        )
    if ".." in command:
        raise HTTPException(status_code=400, detail="command must not contain '..'")
    if any(ch.isspace() for ch in command):
        raise HTTPException(
            status_code=400,
            detail="command must be a single executable path without spaces "
            "(pass arguments via 'args')",
        )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in command):
        raise HTTPException(status_code=400, detail="command must not contain control characters")
    return command


def _validate_args(args: List[str]) -> List[str]:
    """Args are exec'd directly (no shell) — still cap size and control chars."""
    if len(args) > _MAX_ARGS:
        raise HTTPException(status_code=400, detail=f"args must hold at most {_MAX_ARGS} entries")
    out: List[str] = []
    for a in args:
        if not isinstance(a, str):
            raise HTTPException(status_code=400, detail="args entries must be strings")
        if len(a) > _MAX_ARG_LEN:
            raise HTTPException(
                status_code=400, detail=f"args entries must be at most {_MAX_ARG_LEN} chars"
            )
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in a):
            raise HTTPException(
                status_code=400, detail="args entries must not contain control characters"
            )
        out.append(a)
    return out


def _validate_secret_dict(value: Dict[str, Any], field_name: str) -> Dict[str, str]:
    """Length/shape caps for headers/env dicts (values coerced to str)."""
    if len(value) > _MAX_DICT_ENTRIES:
        raise HTTPException(
            status_code=400, detail=f"{field_name} must hold at most {_MAX_DICT_ENTRIES} entries"
        )
    out: Dict[str, str] = {}
    for k, v in value.items():
        key = str(k)
        if not key or len(key) > _MAX_SECRET_KEY_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} keys must be 1..{_MAX_SECRET_KEY_LEN} chars",
            )
        val = "" if v is None else str(v)
        if len(val) > _MAX_SECRET_VALUE_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} values must be at most {_MAX_SECRET_VALUE_LEN} chars",
            )
        if any(ord(ch) < 32 and ch != "\t" for ch in key + val):
            raise HTTPException(
                status_code=400, detail=f"{field_name} must not contain control characters"
            )
        out[key] = val
    return out


def _strip_masked_entries(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Masked-echo-aware secret-dict handling for PUT.

    Entries whose value equals the mask (``***``) are dropped — the admin UI
    echoes masked values back and must not overwrite real secrets with them.
    Returns ``None`` when nothing real remains (``None`` input or an ALL-mask
    payload): keep the stored dict untouched. An empty dict clears the stored
    secrets; a mixed payload replaces wholesale with only the real entries.
    """
    if data is None:
        return None
    real = {str(k): v for k, v in data.items() if str(v) != MASK}
    if data and not real:
        return None  # masked-echo PUT — keep the stored secrets
    return real


def _validate_transport_config(
    transport: str,
    *,
    url: Optional[str],
    command: Optional[str],
    args: Optional[List[str]],
) -> Dict[str, Any]:
    """Cross-field validation for one (transport, url, command, args) combo."""
    if transport not in ("http", "stdio"):
        raise HTTPException(status_code=400, detail="transport must be 'http' or 'stdio'")
    if transport == "http":
        if command is not None:
            raise HTTPException(status_code=400, detail="command is only valid for stdio transport")
        if args is not None:
            raise HTTPException(status_code=400, detail="args are only valid for stdio transport")
        return {"url": _validate_url(url or "")}
    # stdio
    if url is not None:
        raise HTTPException(status_code=400, detail="url is only valid for http transport")
    return {
        "command": _validate_command(command or ""),
        "args": _validate_args(args or []),
    }


def _server_out(s: McpServerORM) -> McpServerOut:
    """API view of a server row — secrets become masked key-only dicts."""
    headers_masked = mask_secret_dict(decrypt_secret_dict(s.headers)) or None
    env_masked = mask_secret_dict(decrypt_secret_dict(s.env)) or None
    return McpServerOut(
        id=s.id,
        name=s.name,
        transport=s.transport,
        url=s.url,
        command=s.command,
        args=list(s.args) if isinstance(s.args, list) else None,
        headers_masked=headers_masked,
        env_masked=env_masked,
        enabled=bool(s.enabled),
        status=s.status or "unknown",
        status_checked_at=s.status_checked_at,
        status_error=s.status_error,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


def _load_server(db: Session, server_id: str) -> McpServerORM:
    server = db.query(McpServerORM).filter(McpServerORM.id == server_id).first()
    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return server


def _commit_or_conflict(db: Session, *, conflict_detail: str) -> None:
    """Commit, mapping integrity violations to 409 and other errors to 500."""
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail=conflict_detail)
    except Exception as e:
        db.rollback()
        logger.error("mcp_admin: DB commit failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Database error")


# --- endpoints -------------------------------------------------------------------
@router.get("/servers", response_model=List[McpServerOut])
async def list_mcp_servers(
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    rows = db.query(McpServerORM).order_by(McpServerORM.created_at).all()
    return [_server_out(s) for s in rows]


@router.post("/servers", response_model=McpServerOut)
async def create_mcp_server(
    body: McpServerCreate,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    if db.query(McpServerORM).filter(McpServerORM.name == name).first() is not None:
        raise HTTPException(status_code=409, detail=f"MCP server named {name!r} already exists")

    cfg = _validate_transport_config(
        body.transport, url=body.url, command=body.command, args=body.args
    )
    headers = (
        _validate_secret_dict(body.headers, "headers") if body.headers is not None else None
    )
    env = _validate_secret_dict(body.env, "env") if body.env is not None else None
    if env and body.transport == "stdio":
        policy_error = stdio_env_error(env)
        if policy_error:
            raise HTTPException(status_code=400, detail=policy_error)

    server = McpServerORM(
        id=f"mcp_{pysecrets.token_hex(16)}",
        name=name,
        transport=body.transport,
        url=cfg.get("url"),
        command=cfg.get("command"),
        args=cfg.get("args"),
        headers=encrypt_secret_dict(headers) if headers is not None else None,
        env=encrypt_secret_dict(env) if env is not None else None,
        enabled=body.enabled,
        status="unknown",
    )
    db.add(server)
    _commit_or_conflict(db, conflict_detail=f"MCP server named {name!r} already exists")
    db.refresh(server)
    return _server_out(server)


@router.put("/servers/{server_id}", response_model=McpServerOut)
async def update_mcp_server(
    server_id: str,
    body: McpServerUpdate,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    server = _load_server(db, server_id)
    fields = body.model_fields_set

    if "transport" in fields and body.transport is not None and body.transport != server.transport:
        raise HTTPException(status_code=400, detail="transport cannot be changed")

    # Merge requested config fields over the stored ones, then cross-validate
    # the merged (transport, url, command, args) combination.
    url = body.url if "url" in fields else server.url
    command = body.command if "command" in fields else server.command
    args = body.args if "args" in fields else server.args
    cfg = _validate_transport_config(server.transport, url=url, command=command, args=args)
    config_changed = (
        cfg.get("url") != server.url
        or cfg.get("command") != server.command
        or (cfg.get("args") or None) != (server.args or None)
    )

    if "name" in fields and body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name must not be empty")
        clash = (
            db.query(McpServerORM)
            .filter(McpServerORM.name == name, McpServerORM.id != server_id)
            .first()
        )
        if clash is not None:
            raise HTTPException(
                status_code=409, detail=f"MCP server named {name!r} already exists"
            )
        server.name = name

    server.url = cfg.get("url")
    server.command = cfg.get("command")
    server.args = cfg.get("args")

    if "headers" in fields:
        cleaned = _strip_masked_entries(body.headers)
        if cleaned is not None:  # all-mask echo keeps the stored secrets
            server.headers = encrypt_secret_dict(_validate_secret_dict(cleaned, "headers"))
            config_changed = True
    if "env" in fields:
        cleaned = _strip_masked_entries(body.env)
        if cleaned is not None:
            server.env = encrypt_secret_dict(_validate_secret_dict(cleaned, "env"))
            config_changed = True
    if server.transport == "stdio":
        # Final-state check: catches both the payload above and legacy rows.
        merged_env = decrypt_secret_dict(server.env)
        if merged_env:
            policy_error = stdio_env_error(merged_env)
            if policy_error:
                raise HTTPException(status_code=400, detail=policy_error)
    if "enabled" in fields and body.enabled is not None:
        server.enabled = body.enabled
    if config_changed:
        # The persisted health verdict belonged to the OLD config.
        server.status = "unknown"
        server.status_error = None
        server.status_checked_at = None

    _commit_or_conflict(db, conflict_detail="MCP server name conflict")
    db.refresh(server)
    # Config may have changed -> drop cached clients/tool lists for this server.
    get_mcp_manager().invalidate(server_id)
    return _server_out(server)


@router.delete("/servers/{server_id}")
async def delete_mcp_server(
    server_id: str,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    server = _load_server(db, server_id)
    db.delete(server)  # bindings cascade (ORM cascade + FK ON DELETE CASCADE)
    _commit_or_conflict(db, conflict_detail="MCP server conflict")
    get_mcp_manager().invalidate(server_id)
    return {"message": "MCP server deleted"}


@router.post("/servers/{server_id}/test", response_model=McpTestResult)
async def test_mcp_server(
    server_id: str,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    """Bounded health-check: one connect + tools/list, status persisted.

    Rate-limited per server (``_TEST_MIN_INTERVAL``): /test doubles as a
    connectivity oracle (closed port errors fast, open port times out), so
    hammering it is rejected with 429.
    """
    server = _load_server(db, server_id)
    now = time.monotonic()
    if now - _last_test_at.get(server_id, 0.0) < _TEST_MIN_INTERVAL:
        raise HTTPException(
            status_code=429,
            detail="health-check rate limit — retry in a couple of seconds",
        )
    _last_test_at[server_id] = now
    if len(_last_test_at) > 4096:  # simple pruning of stale entries
        cutoff = now - _TEST_MIN_INTERVAL
        for stale in [sid for sid, ts in _last_test_at.items() if ts < cutoff]:
            _last_test_at.pop(stale, None)
    ok, detail, tools = await get_mcp_manager().health_check(server, use_cache=False)
    server.status = "ok" if ok else "error"
    server.status_checked_at = datetime.utcnow()
    server.status_error = detail
    _commit_or_conflict(db, conflict_detail="MCP server conflict")
    db.refresh(server)
    return McpTestResult(ok=ok, detail=detail, tools=tools)


@router.get("/servers/{server_id}/tools")
async def cached_mcp_server_tools(
    server_id: str,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
) -> List[Dict[str, Optional[str]]]:
    """The cached discovery result — never connects to the server."""
    _load_server(db, server_id)
    return get_mcp_manager().cached_tools_meta(server_id)
