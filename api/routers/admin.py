"""Admin panel router (contract J, plan section D).

Admin-guarded CRUD + connectivity-test endpoints for configurable groups:
``models``, ``git``, ``confluence``, ``integrations``, ``users``,
``apitokens``. Every endpoint requires an admin session (``require_admin``).
Secrets are encrypted on save via
``api.config.settings.set_setting(..., encrypt=True)`` and redacted on read
(callers see ``hasKey`` booleans, never raw secret values).

API tokens are created here (``POST /api/admin/apitokens``) and verified by
``api.auth.deps.require_api_token`` (sha256 hash). The plaintext token is
returned exactly once at creation time.

Integration/git/confluence connectivity tests import ``api.integrations``
lazily: the integrations package is built in parallel and may not be present
yet, so a missing connector degrades to a clear failure result instead of
crashing the import.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.auth.deps import require_admin
from api.auth.local import (
    RESET_TOKEN_TTL_SECONDS,
    generate_reset_token,
    hash_password,
    hash_token,
)
from api.db import get_db
from api.models import ApiTokenORM, UserORM
from api.schemas import (
    ApiTokenCreate,
    ApiTokenOut,
    UserCreateAdmin,
    UserCreateResult,
    UserOut,
)
from api.config.settings import (
    _sanitize_api_key,
    get_all_rlm_modes,
    get_confluence_creds,
    get_git_creds,
    get_integration_config,
    get_model_for_task,
    list_settings,
    set_setting,
)
from api.prompts import PROMPTS_DIR, PROMPT_FILES, reload_prompt_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Setting-key suffixes that mark a value as a secret (encrypted on save,
# redacted on read). Keep in sync with the secret fields used by the grouped
# getters in api.config.settings (e.g. models.<task>.api_key, git.<host>.token,
# confluence.token).
_SECRET_SUFFIXES = (".api_key", ".token", ".password", ".secret")

# Groups backed by the SettingORM key/value store (contract J).
# ``rlm`` stores per-task LLM/RLM routing modes (rlm.<task>.mode) and is NOT a
# secret group — its values are plain strings (auto/rlm/llm).
# ``ssl`` stores TLS config (ssl.ca_bundle path + ssl.verify toggle) for reaching
# a corporate AI gateway whose cert is signed by an internal CA; NOT secret.
# ``cognee`` stores knowledge graph rate limiting & concurrency settings; NOT secret.
# ``timeouts`` stores per-key timeout overrides (timeouts.<key>) resolved through
# api.config.timeout (admin store > env var > default); NOT secret.
_SETTING_GROUPS = ("models", "git", "confluence", "integrations", "rlm", "ssl", "cognee", "timeouts")

# Model "tasks" exposed in the admin Models section (contract J / plan D).
_MODEL_TASKS = ("docgen", "expert", "summary", "cognee", "embedder")

# Tasks that support an admin-configurable LLM/RLM routing mode.
_RLM_TASKS = ("docgen", "expert", "summary")
# Valid RLM mode values (stored under ``rlm.<task>.mode``).
_RLM_MODE_VALUES = ("auto", "rlm", "llm")

# Git hosts configurable in the admin Git section.
_GIT_HOSTS = ("github", "gitlab")


def _is_secret_key(key: str) -> bool:
    """True for setting keys that hold a secret (encrypted + redacted)."""
    return any(key.endswith(suf) for suf in _SECRET_SUFFIXES)


def _redact_setting(key: str, value: Optional[str], encrypted: bool) -> Dict[str, Any]:
    """Build a redacted view of a stored setting (secrets -> null + hasKey).

    ``list_settings`` returns ciphertext for encrypted rows (never decrypted),
    so for secret keys we drop the value entirely and only report whether a
    value is set, so secrets are never leaked to the admin UI.
    """
    if _is_secret_key(key):
        return {"value": None, "encrypted": encrypted, "hasKey": bool(value)}
    return {"value": value, "encrypted": encrypted, "hasKey": bool(value)}


# --- Response helpers -------------------------------------------------------
def _user_out(u: UserORM) -> UserOut:
    return UserOut(
        id=u.id,
        username=u.username,
        email=u.email,
        role=u.role,
        provider=u.provider,
        created_at=u.created_at,
        must_change_password=getattr(u, "must_change_password", False),
    )


def _token_out(
    t: ApiTokenORM, *, include_token: bool = False, plaintext: Optional[str] = None
) -> ApiTokenOut:
    return ApiTokenOut(
        id=t.id,
        name=t.name,
        created_at=t.created_at,
        last_used_at=t.last_used_at,
        # The raw token is only populated once, at creation time.
        token=plaintext if include_token else None,
    )


def _redact_model_task(cfg: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    has_key = bool(cfg.get("api_key"))
    return {
        "model": cfg.get("model"),
        "base_url": cfg.get("base_url"),
        "api_key": None,
        "hasApiKey": has_key,
        "max_prompt_tokens": cfg.get("max_prompt_tokens"),
        "dimensions": cfg.get("dimensions"),
    }


def _redact_git_creds(creds: Dict[str, Optional[str]]) -> Dict[str, Any]:
    return {"url": creds.get("url"), "token": None, "hasToken": bool(creds.get("token"))}


def _redact_confluence_creds(creds: Dict[str, Optional[str]]) -> Dict[str, Any]:
    return {
        "base_url": creds.get("base_url"),
        "space": creds.get("space"),
        "token": None,
        "hasToken": bool(creds.get("token")),
    }


# --- System prompts (refs/prompts/*.md) -------------------------------------
# These are separate from the _SETTING_GROUPS key/value store: prompt bodies
# live as .md files on disk and are hot-reloadable via reload_prompt_file.
# Declared before the /{group} catch-all so they take priority.
class PromptUpdateRequest(BaseModel):
    content: str


class CogneeReindexRequest(BaseModel):
    product_id: Optional[str] = None


@router.post("/cognee/reindex")
async def trigger_cognee_reindex(
    body: Optional[CogneeReindexRequest] = None,
    _admin: UserORM = Depends(require_admin),
):
    """Manually trigger/force a re-index of the Cognee knowledge graph."""
    pid = body.product_id if body else None
    try:
        from api.cognee import reindex_product_knowledge_graph
        res = await reindex_product_knowledge_graph(pid)
        return res
    except Exception as e:
        logger.error("Admin cognee reindex failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")


def _safe_prompt_filename(filename: str) -> Optional[str]:
    """Validate a prompt filename: must end with .md and stay within PROMPTS_DIR."""
    if not filename or not filename.endswith(".md"):
        return None
    # Reject path traversal: normalize and ensure the resolved path is within PROMPTS_DIR.
    base = os.path.realpath(PROMPTS_DIR)
    target = os.path.realpath(os.path.join(PROMPTS_DIR, filename))
    if not target.startswith(base + os.sep) and target != base:
        return None
    # Only allow known prompt files (registered in PROMPT_FILES).
    if filename not in PROMPT_FILES:
        return None
    return filename


@router.get("/prompts")
def list_prompts(
    _admin: UserORM = Depends(require_admin),
) -> List[Dict[str, Any]]:
    """List all prompt files in refs/prompts/ with their size and mtime."""
    out: List[Dict[str, Any]] = []
    try:
        if not os.path.isdir(PROMPTS_DIR):
            return out
        for fname in sorted(os.listdir(PROMPTS_DIR)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(PROMPTS_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                st = os.stat(fpath)
                out.append({
                    "filename": fname,
                    "size": st.st_size,
                    "modified": int(st.st_mtime),
                })
            except Exception as e:  # pragma: no cover
                logger.warning("Could not stat prompt file %s: %s", fname, e)
    except Exception as e:  # pragma: no cover
        logger.warning("list_prompts failed: %s", e)
    return out


@router.get("/prompts/{filename}")
def get_prompt(
    filename: str,
    _admin: UserORM = Depends(require_admin),
) -> Dict[str, Any]:
    """Return the content of a single prompt file."""
    safe = _safe_prompt_filename(filename)
    if safe is None:
        raise HTTPException(status_code=400, detail="Invalid or unknown prompt filename.")
    fpath = os.path.join(PROMPTS_DIR, safe)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="Prompt file not found.")
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read prompt file: {e}")
    return {"filename": safe, "content": content}


@router.put("/prompts/{filename}")
def update_prompt(
    filename: str,
    body: PromptUpdateRequest,
    _admin: UserORM = Depends(require_admin),
) -> Dict[str, Any]:
    """Write new content to a prompt file and hot-reload it in memory."""
    safe = _safe_prompt_filename(filename)
    if safe is None:
        raise HTTPException(status_code=400, detail="Invalid or unknown prompt filename.")
    fpath = os.path.join(PROMPTS_DIR, safe)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="Prompt file not found.")
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(body.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not write prompt file: {e}")
    # Invalidate the in-memory cache so the new text takes effect immediately.
    try:
        reload_prompt_file(safe)
    except Exception as e:  # pragma: no cover - non-fatal
        logger.warning("reload_prompt_file(%s) failed: %s", safe, e)
    return {"success": True, "filename": safe}


# --- GET /api/admin/{group} -------------------------------------------------
@router.get("/{group}")
def get_group(
    group: str,
    admin: UserORM = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Return the current settings/users/tokens for a group (secrets redacted).

    For ``models``/``git``/``confluence``/``integrations``: raw settings under
    the group prefix plus a ``resolved`` view (effective config with secrets
    redacted as ``hasKey``/``hasToken`` booleans).
    For ``users``: list of users (``UserOut``).
    For ``apitokens``: the current user's API tokens (admins see all).
    """
    if group in _SETTING_GROUPS:
        rows = list_settings(prefix=f"{group}.")
        settings = {
            r["key"]: _redact_setting(r["key"], r["value"], r["encrypted"]) for r in rows
        }
        resp: Dict[str, Any] = {"group": group, "settings": settings}
        if group == "models":
            resp["resolved"] = {
                task: _redact_model_task(get_model_for_task(task)) for task in _MODEL_TASKS
            }
        elif group == "git":
            resp["resolved"] = {
                host: _redact_git_creds(get_git_creds(host)) for host in _GIT_HOSTS
            }
        elif group == "confluence":
            resp["resolved"] = _redact_confluence_creds(get_confluence_creds())
        elif group == "integrations":
            parsed: Dict[str, Any] = {}
            for r in rows:
                if r["key"].startswith("integrations."):
                    name = r["key"][len("integrations."):]
                    parsed[name] = get_integration_config(name)
            resp["resolved"] = parsed
        elif group == "rlm":
            # ``resolved`` is the effective per-task mode AFTER env fallback +
            # fast-rlm availability check (so the UI shows the real routing,
            # e.g. "llm" when fast-rlm is not installed even if stored="auto").
            resp["resolved"] = get_all_rlm_modes()
        elif group == "cognee":
            from api.cognee import _cognee_rate_limiter
            max_conc, delay_sec = _cognee_rate_limiter.get_rate_settings()
            resp["resolved"] = {
                "max_concurrency": str(max_conc),
                "delay_seconds": str(delay_sec),
                "rate_limit_rps": str(round(1.0 / delay_sec, 2)) if delay_sec > 0 else "0",
            }
        elif group == "timeouts":
            # ``resolved`` is the effective per-key timeout AFTER admin store >
            # env var > default + floor, so the UI shows the real value each
            # resolver will return on the next call (and the floor/default for
            # helper text). Stored overrides are plain strings (not secret),
            # so they are already in ``settings`` above.
            from api.config.timeout import get_timeout_resolved_view
            resp["resolved"] = get_timeout_resolved_view()
        return resp

    if group == "users":
        users = db.query(UserORM).all()
        return {"group": "users", "users": [_user_out(u) for u in users]}

    if group == "apitokens":
        q = db.query(ApiTokenORM)
        # Admin sees all tokens; non-admins only their own.
        if admin.role != "admin":
            q = q.filter(ApiTokenORM.user_id == admin.id)
        toks = q.all()
        return {"group": "apitokens", "tokens": [_token_out(t) for t in toks]}

    raise HTTPException(status_code=404, detail=f"Unknown admin group: {group}")


# --- PUT /api/admin/{group} -------------------------------------------------
@router.put("/{group}")
def put_group(
    group: str,
    body: Dict[str, Any],
    admin: UserORM = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Save settings for a group (secrets encrypted). For ``users``: promote/demote.

    Body for setting groups: a JSON object of ``<group>.<...> -> value``. Keys
    outside the requested group are ignored to avoid accidental cross-write.
    Secret keys (suffix ``.api_key``/``.token``/``.password``/``.secret``) are
    encrypted via ``set_setting(..., encrypt=True)``.
    Body for ``users``: ``{user_id, role}`` with ``role`` in (user, admin).
    """
    if group in _SETTING_GROUPS:
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400, detail="Expected a JSON object of setting key -> value"
            )
        saved: List[str] = []
        for key, value in body.items():
            if not isinstance(key, str) or not key.startswith(f"{group}."):
                continue
            # Validate RLM mode values so an invalid mode can't be persisted.
            if group == "rlm" and key.endswith(".mode"):
                v = (value or "").strip().lower() if isinstance(value, str) else value
                if v not in _RLM_MODE_VALUES:
                    logger.warning("Ignoring invalid RLM mode for %r: %r", key, value)
                    continue
            encrypt = _is_secret_key(key)
            str_value = (
                value if (value is None or isinstance(value, str)) else json.dumps(value)
            )
            # Validate optional per-model prompt-token budget so a non-numeric
            # value can't be persisted (and later crash RLM). An empty value
            # clears the override; otherwise it must be a non-negative int.
            if group == "models" and key.endswith(".max_prompt_tokens"):
                normed = str_value.strip() if isinstance(str_value, str) else str(str_value)
                if normed == "":
                    # Clear: store an explicit empty string so the UI reflects it.
                    str_value = ""
                else:
                    try:
                        parsed = int(normed)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Ignoring invalid max_prompt_tokens for %r: %r", key, value
                        )
                        continue
                    if parsed < 0:
                        logger.warning(
                            "Ignoring negative max_prompt_tokens for %r: %r", key, value
                        )
                        continue
                    str_value = str(parsed)
            # Validate timeout overrides: an empty value clears the override;
            # otherwise it must be a positive number (>= the key floor, which
            # the resolver also enforces, so we only reject non-numeric /
            # negative / zero here). Reuse the max_prompt_tokens pattern.
            if group == "timeouts":
                normed = str_value.strip() if isinstance(str_value, str) else str(str_value)
                if normed == "":
                    # Clear: store an explicit empty string so the UI reflects it
                    # and the resolver falls through to env var / default.
                    str_value = ""
                else:
                    try:
                        parsed = float(normed)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Ignoring invalid timeout value for %r: %r", key, value
                        )
                        continue
                    if parsed <= 0:
                        logger.warning(
                            "Ignoring non-positive timeout value for %r: %r", key, value
                        )
                        continue
                    # Keep a clean representation: int when whole, else float.
                    str_value = str(int(parsed)) if float(parsed).is_integer() else str(parsed)
            set_setting(key, str_value, encrypt=encrypt)
            saved.append(key)

        # Trigger instant synchronization across all process subsystems and cognee
        try:
            from api.config.abstraction import sync_runtime_settings
            sync_runtime_settings()
        except Exception as e:
            logger.warning("sync_runtime_settings after admin put_group failed: %s", e)

        return {"group": group, "success": True, "saved": saved}

    if group == "users":
        user_id = body.get("user_id") if isinstance(body, dict) else None
        role = body.get("role") if isinstance(body, dict) else None
        if not user_id or role not in ("user", "admin"):
            raise HTTPException(
                status_code=400,
                detail="Body must include {user_id, role} with role in (user, admin)",
            )
        u = db.get(UserORM, user_id)
        if u is None:
            raise HTTPException(status_code=404, detail="User not found")
        u.role = role
        db.commit()
        db.refresh(u)
        return {"group": "users", "user": _user_out(u)}

    if group == "apitokens":
        raise HTTPException(
            status_code=400,
            detail="Use POST /api/admin/apitokens to create and "
            "DELETE /api/admin/apitokens/{id} to remove",
        )

    raise HTTPException(status_code=404, detail=f"Unknown admin group: {group}")


# --- POST /api/admin/users ---------------------------------------------------
# Create a local user with a temporary password + a one-time reset token.
# Both the temp password and the raw reset token are returned ONCE so the admin
# can hand them to the user (the reset token lets the user set their own
# password via POST /api/auth/reset-password).
@router.post("/users", response_model=UserCreateResult)
def create_user(
    body: UserCreateAdmin,
    admin: UserORM = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a local user. Generates a temp password (if omitted) + a reset token."""
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required.")
    if body.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'.")
    if db.query(UserORM).filter(UserORM.username == username).first() is not None:
        raise HTTPException(status_code=409, detail="Username already taken.")
    temp_password = body.password or secrets.token_urlsafe(12)
    reset_token = generate_reset_token()
    user = UserORM(
        id=f"user_{uuid.uuid4().hex[:24]}",
        username=username,
        email=body.email or None,
        password_hash=hash_password(temp_password),
        role=body.role,
        provider="local",
        reset_token_hash=hash_token(reset_token),
        reset_token_expires=datetime.utcnow() + timedelta(seconds=RESET_TOKEN_TTL_SECONDS),
        must_change_password=body.must_change_password,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Admin %r created user %r (must_change_password=%s).", admin.username, user.username, user.must_change_password)
    return UserCreateResult(
        user=_user_out(user),
        temp_password=temp_password,
        reset_token=reset_token,
    )


@router.post("/users/{user_id}/reset-token")
def issue_reset_token(
    user_id: str,
    admin: UserORM = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Issue a fresh one-time reset token for an existing local user.

    Optionally also resets the user's password to a new temp password (returned
    here once). The user can then sign in with the temp password and change it,
    or use the reset token via POST /api/auth/reset-password.
    """
    u = db.get(UserORM, user_id)
    if u is None:
        raise HTTPException(status_code=404, detail="User not found")
    if u.provider != "local":
        raise HTTPException(status_code=400, detail="Only local users can get a reset token.")
    reset_token = generate_reset_token()
    temp_password = secrets.token_urlsafe(12)
    u.password_hash = hash_password(temp_password)
    u.reset_token_hash = hash_token(reset_token)
    u.reset_token_expires = datetime.utcnow() + timedelta(seconds=RESET_TOKEN_TTL_SECONDS)
    u.must_change_password = True
    db.commit()
    logger.info("Admin %r issued a reset token for user %r.", admin.username, u.username)
    return {
        "user": _user_out(u),
        "temp_password": temp_password,
        "reset_token": reset_token,
    }


# --- POST /api/admin/{group}/test -------------------------------------------
@router.post("/{group}/test")
def test_group(
    group: str,
    body: Optional[Dict[str, Any]] = None,
    admin: UserORM = Depends(require_admin),
):
    """Connectivity test for a group (models/git/confluence/integrations).

    Returns ``{success, message, models?}``. Integration/git/confluence tests
    import ``api.integrations`` lazily and degrade to a failure result if the
    connector is not registered (integrations are built in parallel).
    """
    body = body or {}
    if group == "models":
        return _test_models(body)
    if group == "git":
        return _test_git(body)
    if group == "confluence":
        return _test_confluence(body)
    if group == "integrations":
        return _test_integration(body)
    raise HTTPException(status_code=404, detail=f"No test available for group: {group}")


def _test_models(body: Dict[str, Any]) -> Dict[str, Any]:
    """Ping a configured LLM endpoint (default: the ``expert`` task config)."""
    base_url = body.get("base_url")
    api_key = body.get("api_key")
    model = body.get("model")
    task = body.get("task") or "expert"
    if not base_url or not model:
        cfg = get_model_for_task(task)
        base_url = base_url or cfg.get("base_url")
        api_key = api_key if api_key else cfg.get("api_key")
        model = model or cfg.get("model")
    if not base_url:
        return {"success": False, "message": "No base_url configured for models test."}
    return _ping_model_endpoint(base_url, api_key, model, task=task)


def _ping_model_endpoint(
    base_url: str, api_key: Optional[str], model: Optional[str] = None, task: Optional[str] = None
) -> Dict[str, Any]:
    """Ping a configured LLM or Embedder endpoint."""
    try:
        import requests
    except Exception as e:  # pragma: no cover - dep missing
        return {"success": False, "message": f"requests not available: {e}"}
    from api.config.ssl import requests_verify
    # Normalize a pasted key (strip whitespace / a stray "Bearer " prefix the
    # SDK would double up) so the test reflects real client behaviour.
    api_key = _sanitize_api_key(api_key)
    headers = {}
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        # Every supported server (Ollama, LM Studio, llama.cpp, vLLM, ...)
        # exposes the OpenAI-compatible /v1/models endpoint. Ollama's :11434
        # also serves /v1/models.
        url = base_url
        if not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        resp = requests.get(f"{url}/models", headers=headers, timeout=10, verify=requests_verify())
        if resp.status_code != 200:
            return {
                "success": False,
                "message": f"OpenAI-compatible API returned status {resp.status_code}",
            }
        names = [m.get("id", "") for m in resp.json().get("data", [])]
        probe_model = model or (names[0] if names else "")
        chat_note = ""
        if probe_model:
            try:
                if task == "embedder":
                    probe_resp = requests.post(
                        f"{url}/embeddings",
                        headers=headers,
                        json={
                            "model": probe_model,
                            "input": "ping",
                        },
                        timeout=15,
                        verify=requests_verify(),
                    )
                    if probe_resp.status_code in (401, 403):
                        return {
                            "success": False,
                            "message": (
                                f"Auth rejected by /v1/embeddings "
                                f"(status {probe_resp.status_code}). The key is "
                                f"invalid or lacks embedding permissions."
                            ),
                        }
                    if probe_resp.status_code >= 400:
                        chat_note = f" (embeddings probe returned status {probe_resp.status_code})"
                else:
                    chat_resp = requests.post(
                        f"{url}/chat/completions",
                        headers=headers,
                        json={
                            "model": probe_model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                        },
                        timeout=15,
                        verify=requests_verify(),
                    )
                    if chat_resp.status_code in (401, 403):
                        return {
                            "success": False,
                            "message": (
                                f"Auth rejected by /v1/chat/completions "
                                f"(status {chat_resp.status_code}). The key is "
                                f"invalid or lacks chat permissions, though "
                                f"GET /v1/models succeeded (this gateway does not "
                                f"enforce auth on /v1/models)."
                            ),
                        }
                    if chat_resp.status_code >= 400:
                        chat_note = f" (chat probe returned status {chat_resp.status_code})"
            except Exception as ce:
                chat_note = f" (probe failed: {ce})"
        return {
            "success": True,
            "message": f"OpenAI-compatible API reachable; {len(names)} model(s).{chat_note}",
            "models": names,
        }
    except Exception as e:
        # Covers ConnectionError / Timeout / JSON errors uniformly.
        return {"success": False, "message": f"Error testing connection: {e}"}


def _lazy_connector(name: str):
    """Import api.integrations.registry lazily and fetch a connector by name."""
    try:
        from api.integrations import registry as _reg  # lazy: built in parallel
    except Exception as e:
        logger.debug("integrations registry not importable: %s", e)
        return None, f"Integrations not available: {e}"
    getter = getattr(_reg, "get_connector", None)
    if not callable(getter):
        return None, "Integrations registry has no get_connector(name)."
    try:
        return getter(name), None
    except Exception as e:
        return None, f"No connector registered for '{name}': {e}"


def _connector_test_result(name: str, connector, cfg: Any) -> Dict[str, Any]:
    """Call a connector's test() and normalize the result to {success, message}."""
    test_fn = getattr(connector, "test", None)
    if not callable(test_fn):
        return {"success": False, "message": f"Connector '{name}' has no test()."}
    try:
        # Connectors are instantiated with their config by get_connector(),
        # so test() takes no arguments here.
        result = test_fn()
    except Exception as e:
        return {"success": False, "message": f"'{name}' test failed: {e}"}
    if isinstance(result, dict):
        return result
    if result is True:
        return {"success": True, "message": f"'{name}' reachable."}
    return {"success": False, "message": f"'{name}' test returned no success."}


def _test_git(body: Dict[str, Any]) -> Dict[str, Any]:
    host = (body.get("host") or "github").lower()
    if host not in _GIT_HOSTS:
        return {"success": False, "message": f"Unknown git host: {host}"}
    creds = get_git_creds(host)
    connector, err = _lazy_connector(host)
    if connector is None:
        return {"success": False, "message": err or f"No git connector for '{host}'."}
    return _connector_test_result(host, connector, creds)


def _test_confluence(body: Dict[str, Any]) -> Dict[str, Any]:
    creds = get_confluence_creds()
    connector, err = _lazy_connector("confluence")
    if connector is None:
        return {"success": False, "message": err or "No Confluence connector registered."}
    return _connector_test_result("confluence", connector, creds)


def _test_integration(body: Dict[str, Any]) -> Dict[str, Any]:
    name = body.get("name")
    if not name:
        return {
            "success": False,
            "message": "Body must include {name} of the integration to test.",
        }
    cfg = get_integration_config(name)
    connector, err = _lazy_connector(name)
    if connector is None:
        return {"success": False, "message": err or f"No connector for '{name}'."}
    return _connector_test_result(name, connector, cfg)


# --- API tokens -------------------------------------------------------------
def _hash_token(raw: str) -> str:
    """sha256 hex of the raw token (matches api.auth.deps.require_api_token)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@router.post("/apitokens", response_model=ApiTokenOut, status_code=status.HTTP_201_CREATED)
def create_api_token(
    body: ApiTokenCreate,
    admin: UserORM = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create an API token for the current admin.

    The plaintext token is returned exactly once; only its sha256 hash is
    persisted. Verification happens in ``api.auth.deps.require_api_token``.
    """
    raw = secrets.token_urlsafe(32)
    tok = ApiTokenORM(
        id=f"tok_{uuid.uuid4().hex[:24]}",
        user_id=admin.id,
        token_hash=_hash_token(raw),
        name=body.name,
        created_at=datetime.utcnow(),
    )
    db.add(tok)
    try:
        db.commit()
        db.refresh(tok)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create token: {e}")
    return _token_out(tok, include_token=True, plaintext=raw)


@router.delete("/apitokens/{token_id}")
def delete_api_token(
    token_id: str,
    admin: UserORM = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete an API token. Admins can delete any token; non-admins only their own."""
    tok = db.get(ApiTokenORM, token_id)
    if tok is None:
        raise HTTPException(status_code=404, detail="API token not found")
    if admin.role != "admin" and tok.user_id != admin.id:
        raise HTTPException(
            status_code=403, detail="Cannot delete another user's token"
        )
    db.delete(tok)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete token: {e}")
    return {"success": True, "id": token_id}
