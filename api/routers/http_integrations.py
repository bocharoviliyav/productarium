"""Admin HTTP integrations router (issue #3).

Admin-guarded CRUD + bounded test call for templated read-only GET
integrations (``HttpIntegrationORM``) that are exposed to agents as named
tools (built in :mod:`api.integrations.http_tools`). Prefix
``/api/admin/integrations/http``:

- ``GET    /api/admin/integrations/http``        — list (headers masked)
- ``POST   /api/admin/integrations/http``        — register an integration
- ``PUT    /api/admin/integrations/http/{id}``   — partial update
- ``DELETE /api/admin/integrations/http/{id}``   — delete
- ``POST   /api/admin/integrations/http/{id}/test`` — bounded GET with the
  declared default variable values.

Security (mirrors ``api/routers/mcp_admin.py``):
- ``headers`` arrive as a PLAINTEXT dict, are Fernet-encrypted before storage
  and only ever leave as a masked key-only view (``headers_masked``).
- ``url_template`` must be http(s) without embedded credentials; every
  ``{placeholder}`` must be a declared variable (or the implicit
  ``{product_name}``) so the agent can never smuggle arbitrary URLs.
- Variable names are ``[A-Za-z0-9_]``-only (they become tool parameters);
  values are URL-quoted at call time.
- All inputs are length-capped; test errors are short + sanitized.
"""

from __future__ import annotations

import logging
import re
import secrets as pysecrets
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.auth.deps import require_admin
from api.config.timeout import resolve_integration_http_timeout
from api.db import get_db
from api.mcp.secrets import (
    MASK,
    decrypt_secret_dict,
    encrypt_secret_dict,
    mask_secret_dict,
)
from api.models import HttpIntegrationORM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/integrations/http", tags=["http-integrations"])

#: Max entries / sizes (mirrors the MCP registry caps).
_MAX_DICT_ENTRIES = 64
_MAX_SECRET_KEY_LEN = 128
_MAX_SECRET_VALUE_LEN = 2048
_MAX_VARIABLES = 16
_MAX_NAME_LEN = 128
_MAX_DESC_LEN = 1024
_MAX_URL_LEN = 512

_VARIABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
#: The always-available placeholder (substituted at tool-call time).
_IMPLICIT_PLACEHOLDER = "product_name"
#: Chars kept from a test response body in the API response.
_TEST_BODY_PREVIEW_CHARS = 2000


# --- pydantic models ----------------------------------------------------------
class HttpVariable(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: Optional[str] = Field(default=None, max_length=512)
    default: Optional[str] = Field(default=None, max_length=512)


class HttpIntegrationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=_MAX_NAME_LEN)
    description: Optional[str] = Field(default=None, max_length=_MAX_DESC_LEN)
    url_template: str = Field(min_length=1, max_length=_MAX_URL_LEN)
    headers: Optional[Dict[str, str]] = None
    variables: Optional[List[HttpVariable]] = Field(default=None, max_length=_MAX_VARIABLES)
    enabled: bool = True


class HttpIntegrationUpdate(BaseModel):
    """Partial update. Secret-dict semantics mirror the MCP registry: a field
    ABSENT keeps the stored value; entries whose value is the mask ``***``
    are dropped (a UI echoing the masked view back must not wipe secrets)."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=_MAX_NAME_LEN)
    description: Optional[str] = Field(default=None, max_length=_MAX_DESC_LEN)
    url_template: Optional[str] = Field(default=None, min_length=1, max_length=_MAX_URL_LEN)
    headers: Optional[Dict[str, str]] = None
    variables: Optional[List[HttpVariable]] = Field(default=None, max_length=_MAX_VARIABLES)
    enabled: Optional[bool] = None


class HttpIntegrationOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    url_template: str
    headers_masked: Optional[Dict[str, str]] = None
    variables: Optional[List[Dict[str, Optional[str]]]] = None
    enabled: bool
    created_at: datetime
    updated_at: datetime


class HttpIntegrationTestResult(BaseModel):
    ok: bool
    status_code: Optional[int] = None
    detail: Optional[str] = None
    body_preview: Optional[str] = None


# --- validation helpers ---------------------------------------------------------
def _validate_url_template(template: str) -> str:
    """Require http(s), no embedded credentials, known placeholders only."""
    template = (template or "").strip()
    parsed = urlparse(template)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=400, detail="url_template must be a valid http(s) URL"
        )
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=400,
            detail="url_template must not embed credentials (user:pass@host); use headers",
        )
    return template


def _validate_placeholders(template: str, variables: List[HttpVariable]) -> None:
    """Every {placeholder} must be a declared variable or {product_name}."""
    declared = {v.name for v in variables} | {_IMPLICIT_PLACEHOLDER}
    unknown = sorted(set(_PLACEHOLDER_RE.findall(template)) - declared)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"url_template placeholder(s) {unknown} are not declared "
                f"variables (available implicit placeholder: {{{_IMPLICIT_PLACEHOLDER}}})"
            ),
        )


def _validate_variables(variables: List[HttpVariable]) -> List[Dict[str, Optional[str]]]:
    if len(variables) > _MAX_VARIABLES:
        raise HTTPException(
            status_code=400, detail=f"variables must hold at most {_MAX_VARIABLES} entries"
        )
    seen: set = set()
    out: List[Dict[str, Optional[str]]] = []
    for v in variables:
        name = (v.name or "").strip()
        if not _VARIABLE_NAME_RE.match(name):
            raise HTTPException(
                status_code=400,
                detail=f"variable name {name!r} must match [A-Za-z_][A-Za-z0-9_]*",
            )
        if name == _IMPLICIT_PLACEHOLDER:
            raise HTTPException(
                status_code=400,
                detail=f"{name!r} is reserved (substituted automatically)",
            )
        if name in seen:
            raise HTTPException(status_code=400, detail=f"duplicate variable name {name!r}")
        seen.add(name)
        out.append(
            {
                "name": name,
                "description": (v.description or "").strip() or None,
                "default": v.default,
            }
        )
    return out


def _validate_secret_dict(value: Dict[str, Any], field_name: str) -> Dict[str, str]:
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
    """Masked-echo-aware secret handling for PUT (same rules as MCP registry)."""
    if data is None:
        return None
    real = {str(k): v for k, v in data.items() if str(v) != MASK}
    if data and not real:
        return None  # masked-echo PUT — keep the stored secrets
    return real


def _integration_out(row: HttpIntegrationORM) -> HttpIntegrationOut:
    headers_masked = mask_secret_dict(decrypt_secret_dict(row.headers)) or None
    return HttpIntegrationOut(
        id=row.id,
        name=row.name,
        description=row.description,
        url_template=row.url_template,
        headers_masked=headers_masked,
        variables=row.variables if isinstance(row.variables, list) else None,
        enabled=bool(row.enabled),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _load_integration(db: Session, integration_id: str) -> HttpIntegrationORM:
    row = db.query(HttpIntegrationORM).filter(HttpIntegrationORM.id == integration_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="HTTP integration not found")
    return row


def _commit_or_conflict(db: Session, *, conflict_detail: str) -> None:
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail=conflict_detail)
    except Exception as e:
        db.rollback()
        logger.error("http_integrations: DB commit failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Database error")


def _check_name_free(db: Session, name: str, *, exclude_id: Optional[str] = None) -> None:
    q = db.query(HttpIntegrationORM).filter(HttpIntegrationORM.name == name)
    if exclude_id is not None:
        q = q.filter(HttpIntegrationORM.id != exclude_id)
    if q.first() is not None:
        raise HTTPException(
            status_code=409, detail=f"HTTP integration named {name!r} already exists"
        )


# --- endpoints -------------------------------------------------------------------
@router.get("", response_model=List[HttpIntegrationOut])
async def list_http_integrations(
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    rows = db.query(HttpIntegrationORM).order_by(HttpIntegrationORM.created_at).all()
    return [_integration_out(r) for r in rows]


@router.post("", response_model=HttpIntegrationOut)
async def create_http_integration(
    body: HttpIntegrationCreate,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    _check_name_free(db, name)

    variables = _validate_variables(body.variables or [])
    template = _validate_url_template(body.url_template)
    _validate_placeholders(template, body.variables or [])
    headers = (
        _validate_secret_dict(body.headers, "headers") if body.headers is not None else None
    )

    row = HttpIntegrationORM(
        id=f"httpint_{pysecrets.token_hex(16)}",
        name=name,
        description=(body.description or "").strip() or None,
        url_template=template,
        headers=encrypt_secret_dict(headers) if headers is not None else None,
        variables=variables or None,
        enabled=body.enabled,
    )
    db.add(row)
    _commit_or_conflict(db, conflict_detail=f"HTTP integration named {name!r} already exists")
    db.refresh(row)
    return _integration_out(row)


@router.put("/{integration_id}", response_model=HttpIntegrationOut)
async def update_http_integration(
    integration_id: str,
    body: HttpIntegrationUpdate,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    row = _load_integration(db, integration_id)
    fields = body.model_fields_set

    # Merge requested variables over the stored ones. Request payloads arrive
    # as HttpVariable models; stored rows hold plain dicts.
    raw_vars: List[Any] = list(
        body.variables if "variables" in fields and body.variables is not None
        else (row.variables or [])
    )
    if "variables" in fields and body.variables is None:
        raw_vars = []  # explicit null clears the variable contract
    merged_variables: List[HttpVariable] = []
    for var in raw_vars:
        if isinstance(var, HttpVariable):
            merged_variables.append(var)
        elif isinstance(var, dict):
            merged_variables.append(
                HttpVariable(
                    **{
                        k: v
                        for k, v in var.items()
                        if k in ("name", "description", "default")
                    }
                )
            )
    variables = _validate_variables(merged_variables)

    template = (
        _validate_url_template(body.url_template)
        if "url_template" in fields and body.url_template is not None
        else row.url_template
    )
    _validate_placeholders(template, merged_variables)

    if "name" in fields and body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name must not be empty")
        _check_name_free(db, name, exclude_id=integration_id)
        row.name = name
    if "description" in fields:
        row.description = (body.description or "").strip() or None
    row.url_template = template
    row.variables = variables or None

    if "headers" in fields:
        cleaned = _strip_masked_entries(body.headers)
        if cleaned is not None:  # all-mask echo keeps the stored secrets
            validated = _validate_secret_dict(cleaned, "headers")
            row.headers = encrypt_secret_dict(validated) if validated else None
        elif body.headers == {}:  # explicit empty dict clears the secrets
            row.headers = None

    if "enabled" in fields and body.enabled is not None:
        row.enabled = body.enabled

    _commit_or_conflict(
        db, conflict_detail=f"HTTP integration named {row.name!r} already exists"
    )
    db.refresh(row)
    return _integration_out(row)


@router.delete("/{integration_id}")
async def delete_http_integration(
    integration_id: str,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    row = _load_integration(db, integration_id)
    db.delete(row)
    _commit_or_conflict(db, conflict_detail="HTTP integration not found")
    return {"success": True}


@router.post("/{integration_id}/test", response_model=HttpIntegrationTestResult)
async def test_http_integration(
    integration_id: str,
    db: Session = Depends(get_db),
    _admin: Any = Depends(require_admin),
):
    """Perform one bounded GET using the declared default variable values."""
    row = _load_integration(db, integration_id)
    headers = decrypt_secret_dict(row.headers)
    url = row.url_template
    for var in row.variables or []:
        if isinstance(var, dict) and var.get("name"):
            url = url.replace("{" + str(var["name"]) + "}", str(var.get("default") or ""))

    try:
        async with httpx.AsyncClient(
            timeout=resolve_integration_http_timeout(), follow_redirects=True
        ) as client:
            resp = await client.get(url, headers=headers or None)
        return HttpIntegrationTestResult(
            ok=resp.status_code < 400,
            status_code=resp.status_code,
            body_preview=resp.text[:_TEST_BODY_PREVIEW_CHARS],
        )
    except Exception as e:
        name = type(e).__name__
        msg = " ".join(str(e).split())[:160]
        logger.warning("http integration %r test failed: %s", row.name, e)
        return HttpIntegrationTestResult(
            ok=False, detail=f"{name}: {msg}" if msg else name
        )


__all__ = ["router"]
