"""Encrypted key/value settings store over ``SettingORM`` (admin config, item 8).

Persistence layer for admin-configured models, git credentials, Confluence,
and integrations. Secrets are encrypted with ``cryptography.fernet`` using
``SETTINGS_SECRET_KEY``. If the key is unset, a stable-per-process dev key is
generated and a WARNING is logged (NOT for production).

All functions are import-safe when the DB is down: they catch exceptions, log a
warning/debug message, and return defaults/None so callers (and app import)
never crash. Grouped convenience getters read-through with env fallback:

- ``get_model_for_task(task)``     -> {provider, model, base_url, api_key}
- ``get_git_creds(host)``          -> {url, token}
- ``get_confluence_creds()``       -> {base_url, token, space}
- ``get_integration_config(name)`` -> dict (JSON value)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet  # type: ignore
    _CRYPTO_AVAILABLE = True
except Exception as _e:  # pragma: no cover - dep missing
    Fernet = None  # type: ignore
    _CRYPTO_AVAILABLE = False
    logger.warning("cryptography not available; settings_store secrets disabled: %s", _e)


# Ephemeral dev key (stable within a single process). Used ONLY as a last
# resort when neither SETTINGS_SECRET_KEY nor the persisted key file is
# available (e.g. read-only filesystem). Real deployments should set
# SETTINGS_SECRET_KEY or let the persisted key file be created once.
_DEV_KEY: Optional[str] = None


def _dev_fernet_key() -> str:
    """Generate (and cache) a stable-per-process Fernet key for dev."""
    global _DEV_KEY
    if _DEV_KEY is None:
        if not _CRYPTO_AVAILABLE:
            return ""
        _DEV_KEY = Fernet.generate_key().decode("utf-8")
    return _DEV_KEY


def _persisted_key_path() -> str:
    """Filesystem location of the persisted Fernet key (used when env unset).

    Honours DEEPWIKI_CONFIG_DIR when set (same override as the config loader);
    otherwise defaults to ``~/.adalflow/.settings_secret_key`` so the key
    survives container restarts (``~/.adalflow`` is the mounted data volume).
    """
    base = os.environ.get("DEEPWIKI_CONFIG_DIR")
    if base:
        return os.path.join(base, ".settings_secret_key")
    return os.path.join(os.path.expanduser("~"), ".adalflow", ".settings_secret_key")


def _load_or_create_persisted_key() -> Optional[str]:
    """Load the persisted Fernet key; create one on first run.

    Returns the key string, or None if the filesystem is unwritable / crypto
    unavailable. Never raises.
    """
    if not _CRYPTO_AVAILABLE:
        return None
    try:
        path = _persisted_key_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            key = ""
            try:
                with open(path, "r", encoding="utf-8") as f:
                    key = f.read().strip()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Could not read persisted secret key %s: %s", path, e)
            if key:
                return key
        # First run: generate a fresh key and persist it (0600 perms).
        key = Fernet.generate_key().decode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key)
        logger.info("Generated and persisted SETTINGS_SECRET_KEY to %s.", path)
        return key
    except Exception as e:
        logger.warning("Could not load/create persisted SETTINGS_SECRET_KEY: %s", e)
        return None


def bootstrap_secret_key() -> None:
    """Ensure SETTINGS_SECRET_KEY is available (env or persisted file).

    Call early at startup so JWT signing (api.auth.tokens) and Fernet
    encryption share a stable key that survives restarts. Idempotent and
    never raises: on any failure the ephemeral dev key is used as a fallback.
    """
    try:
        if os.environ.get("SETTINGS_SECRET_KEY"):
            return
        key = _load_or_create_persisted_key()
        if key:
            os.environ["SETTINGS_SECRET_KEY"] = key
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("bootstrap_secret_key failed: %s", e)


def _fernet() -> Optional["Fernet"]:
    """Build a Fernet from SETTINGS_SECRET_KEY (env or persisted file).

    Precedence: ``SETTINGS_SECRET_KEY`` env var > persisted key file
    (``~/.adalflow/.settings_secret_key``) > ephemeral per-process dev key
    (last resort, logged as a warning). When the persisted key is used it is
    also exported to ``SETTINGS_SECRET_KEY`` so other readers (e.g. JWT
    signing in api.auth.tokens) share the same stable key.
    """
    if not _CRYPTO_AVAILABLE:
        return None
    key = os.environ.get("SETTINGS_SECRET_KEY")
    if not key:
        key = _load_or_create_persisted_key()
        if key:
            # Export so JWT signing + future calls share it without re-reading.
            os.environ["SETTINGS_SECRET_KEY"] = key
        else:
            # Last resort: ephemeral per-process key (with warning).
            key = _dev_fernet_key()
            logger.warning(
                "SETTINGS_SECRET_KEY not set and persisted key unavailable; "
                "using an ephemeral dev Fernet key (NOT for production). Set "
                "SETTINGS_SECRET_KEY to persist encrypted settings across restarts."
            )
    try:
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except Exception as e:
        logger.warning("Invalid SETTINGS_SECRET_KEY; encrypted settings disabled: %s", e)
        return None


# --- Core CRUD --------------------------------------------------------------
def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting by key. Decrypts if the row is marked encrypted."""
    try:
        from api.db import SessionLocal
        from api.models import SettingORM
        with SessionLocal() as db:
            row = db.get(SettingORM, key)
            if row is None:
                return default
            if row.encrypted:
                f = _fernet()
                if f is None or not row.value:
                    return default
                try:
                    return f.decrypt(row.value.encode("utf-8")).decode("utf-8")
                except Exception as e:
                    logger.warning("Failed to decrypt setting %r: %s", key, e)
                    return default
            return row.value
    except Exception as e:
        logger.debug("get_setting(%r) failed (DB down?): %s", key, e)
        return default


def set_setting(key: str, value: Optional[str], encrypt: bool = False) -> None:
    """Upsert a setting. If encrypt=True, stores a Fernet-encrypted ciphertext."""
    stored: Optional[str] = value
    encrypted = False
    if encrypt and value is not None:
        f = _fernet()
        if f is not None:
            try:
                stored = f.encrypt(value.encode("utf-8")).decode("utf-8")
                encrypted = True
            except Exception as e:
                logger.warning("Failed to encrypt setting %r; storing plaintext: %s", key, e)
                stored = value
                encrypted = False
        else:
            logger.warning("Encryption requested for %r but Fernet unavailable; storing plaintext.", key)
            stored = value
            encrypted = False
    try:
        from api.db import SessionLocal
        from api.models import SettingORM
        with SessionLocal() as db:
            row = db.get(SettingORM, key)
            if row is None:
                row = SettingORM(key=key, value=stored, encrypted=encrypted)
                db.add(row)
            else:
                row.value = stored
                row.encrypted = encrypted
            db.commit()
    except Exception as e:
        logger.warning("set_setting(%r) failed (DB down?): %s", key, e)


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a (possibly encrypted) secret. Alias for get_setting (which decrypts)."""
    return get_setting(key, default=default)


def delete_setting(key: str) -> bool:
    """Delete a setting by key. Returns True if a row was deleted."""
    try:
        from api.db import SessionLocal
        from api.models import SettingORM
        with SessionLocal() as db:
            row = db.get(SettingORM, key)
            if row is None:
                return False
            db.delete(row)
            db.commit()
            return True
    except Exception as e:
        logger.warning("delete_setting(%r) failed (DB down?): %s", key, e)
        return False


def list_settings(prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    """List settings (optionally filtered by key prefix).

    Encrypted values are NOT decrypted in the listing (to avoid leaking
    secrets); the ``encrypted`` flag is returned so callers can decide.
    """
    try:
        from api.db import SessionLocal
        from api.models import SettingORM
        with SessionLocal() as db:
            q = db.query(SettingORM)
            if prefix:
                q = q.filter(SettingORM.key.like(f"{prefix}%"))
            rows = q.all()
            return [
                {"key": r.key, "value": r.value, "encrypted": r.encrypted}
                for r in rows
            ]
    except Exception as e:
        logger.debug("list_settings(%r) failed (DB down?): %s", prefix, e)
        return []


# --- Grouped convenience getters (read-through with env fallback) -----------
def _parse_int_setting(value: Optional[str]) -> Optional[int]:
    """Parse a stored setting as a non-negative int; None when unset/invalid.

    Used for ``models.<task>.max_prompt_tokens``: an empty/missing value keeps
    the caller's default, a non-numeric value is ignored (never raises) so a
    junk value cannot crash RLM startup.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _sanitize_api_key(value: Optional[str]) -> Optional[str]:
    """Normalize an API key read from the settings store.

    Admins paste keys into the admin panel (or a ``.env``), and common mistakes
    (quotes, leading Bearer prefix, trailing whitespace) break auth with
    OpenAI-compatible gateways.

    Preserves any key format (UUID, hex, sk-*, JWT, custom) as long as it is a
    non-empty string after normalization.
    """
    if not value:
        return value
    k = value.strip()
    # Strip surrounding quotes if accidentally quoted in .env or UI
    if len(k) >= 2 and ((k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'"))):
        k = k[1:-1].strip()
    # Strip a leading "Bearer " (case-insensitive) the SDK would double up.
    if len(k) >= 7 and k[:7].lower() == "bearer ":
        k = k[7:].strip()
    return k


def get_model_for_task(task: str) -> Dict[str, Optional[str]]:
    """Resolve a model config for a task (docgen/expert/summary/cognee/embedder).

    Reads keys ``models.<task>.{model,base_url,api_key}`` from the settings
    store, falling back to environment variables when unset. Also reads the
    optional ``models.<task>.max_prompt_tokens`` (int) used by fast-rlm:
    ``None`` when unset (callers keep the fast-rlm default); non-numeric stored
    values are ignored (treated as unset) so a bad value never crashes callers.

    Every supported local server (LM Studio, llama.cpp, vLLM, ...)
    exposes an OpenAI-compatible ``/v1`` API, so a single defaults path covers
    all cases.
    """
    p = "models.%s." % task
    # OpenAI-compatible defaults: LOCAL_OPENAI_BASE_URL points at the server
    # (LM Studio :1234, llama.cpp, vLLM, ...). The same defaults work for
    # every local server.
    default_base = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1")
    default_model = os.environ.get("LOCAL_OPENAI_MODEL") or os.environ.get("RLM_MODEL_NAME") or os.environ.get("LLM_MODEL") or "qwen/qwen3.6-27b"
    default_key = os.environ.get("LOCAL_OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "not-needed"
    return {
        "model": get_setting(p + "model") or default_model,
        "base_url": get_setting(p + "base_url") or default_base,
        "api_key": _sanitize_api_key(get_secret(p + "api_key")) or default_key,
        "max_prompt_tokens": _parse_int_setting(get_setting(p + "max_prompt_tokens")),
        "dimensions": _parse_int_setting(get_setting(p + "dimensions")),
    }


# --- RLM mode (per-task LLM/RLM routing) ------------------------------------
# Valid modes:
#   "auto" - use RLM when the context is large (>= RLM_MIN_CHARS), else LLM
#   "rlm"  - always use RLM (falls back to LLM on RLM failure)
#   "llm"  - never use RLM; always use the standard LLM directly
_RLM_MODES = ("auto", "rlm", "llm")
_RLM_TASKS = ("docgen", "expert", "summary")


def get_rlm_mode(task: str) -> str:
    """Resolve the LLM/RLM routing mode for a task (docgen/expert/summary).

    Returns one of ``auto`` / ``rlm`` / ``llm``. Reads ``rlm.<task>.mode`` from
    the settings store, falling back to ``RLM_DEFAULT_MODE`` (default ``auto``).
    If fast-rlm is not installed (``_FAST_RLM_AVAILABLE`` is False in
    ``api.rlm.runner``), ALWAYS returns ``llm`` so callers never try RLM when
    it cannot work — this is the "guaranteed operation" baseline.
    """
    # Check fast-rlm availability WITHOUT importing rlm.runner at module load
    # (it imports fast_rlm lazily; we read its flag defensively).
    try:
        from api.rlm.runner import _FAST_RLM_AVAILABLE  # lazy; avoids circular import
        if not _FAST_RLM_AVAILABLE:
            return "llm"
    except Exception:  # pragma: no cover - import-safe
        # If we can't even import the flag, assume RLM is unavailable.
        return "llm"
    raw = get_setting(f"rlm.{task}.mode")
    if raw and raw.strip().lower() in _RLM_MODES:
        return raw.strip().lower()
    env_default = os.environ.get("RLM_DEFAULT_MODE", "auto").strip().lower()
    return env_default if env_default in _RLM_MODES else "auto"


def get_all_rlm_modes() -> Dict[str, str]:
    """Return the resolved RLM mode for every task (for the admin UI)."""
    return {task: get_rlm_mode(task) for task in _RLM_TASKS}


def get_git_creds(host: str) -> Dict[str, Optional[str]]:
    """Resolve git credentials for a host (github|gitlab).

    Reads ``git.<host>.{url,token}`` from the store, falling back to the
    existing ``GITHUB_ENTERPRISE_URL`` / ``GITLAB_SELF_HOSTED_URL`` env vars
    (token has no env fallback by design).
    """
    p = "git.%s." % host
    env_url_map = {
        "github": "GITHUB_ENTERPRISE_URL",
        "gitlab": "GITLAB_SELF_HOSTED_URL",
    }
    return {
        "url": get_setting(p + "url") or os.environ.get(env_url_map.get(host, ""), ""),
        "token": get_secret(p + "token"),
    }


# Public hosts used when an account's URL is left blank ("this account is for
# the public cloud").
_PUBLIC_GIT_HOSTS = {"github": "github.com", "gitlab": "gitlab.com"}


def get_git_accounts(host: str) -> List[Dict[str, Optional[str]]]:
    """Resolve all git accounts for a host (github|gitlab).

    Reads per-account keys ``git.<host>.accounts.<index>.url`` (plain) and
    ``git.<host>.accounts.<index>.token`` (encrypted). When no explicit
    accounts exist, falls back to the legacy single-account
    :func:`get_git_creds` (``git.<host>.{url,token}`` + env var URL fallback)
    so existing configuration keeps working unchanged.
    """
    prefix = "git.%s.accounts." % host
    indices: set[int] = set()
    for row in list_settings(prefix=prefix):
        rest = row["key"][len(prefix):]
        if "." not in rest:
            continue
        idx_str = rest.split(".", 1)[0]
        if idx_str.isdigit():
            indices.add(int(idx_str))

    if not indices:
        legacy = get_git_creds(host)
        if legacy.get("url") or legacy.get("token"):
            return [legacy]
        return []

    accounts: List[Dict[str, Optional[str]]] = []
    for i in sorted(indices):
        url = get_setting(prefix + "%d.url" % i) or ""
        token = get_secret(prefix + "%d.token" % i)
        if url or token:
            accounts.append({"url": url, "token": token})
    return accounts


def _normalize_git_host(url: str) -> str:
    """Normalize a git account/repo URL to its ``scheme://host:port`` host."""
    text = (url or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    parsed = urlparse(text)
    netloc = (parsed.netloc or "").lower().rstrip(".")
    if not netloc:
        return ""
    if parsed.scheme:
        return "%s://%s" % (parsed.scheme.lower(), netloc)
    return netloc


def _git_account_host(account_url: Optional[str], provider: str) -> str:
    if account_url and account_url.strip():
        return _normalize_git_host(account_url)
    return _normalize_git_host(_PUBLIC_GIT_HOSTS.get(provider, ""))


def resolve_git_token(
    repo_url: str, repo_type: Optional[str] = None
) -> Optional[str]:
    """Return the token for the account whose host matches ``repo_url``.

    The provider is taken from ``repo_type`` or inferred from the repo URL host
    (``gitlab`` when the host contains ``gitlab``, else ``github``). Matching is
    by normalized ``scheme://host[:port]``, case-insensitive. Returns ``None``
    for public access when no account matches.
    """
    repo_url = (repo_url or "").strip()
    if not repo_url:
        return None
    repo_host = _normalize_git_host(repo_url)
    if not repo_host:
        return None
    provider = (repo_type or "").strip().lower()
    if provider not in _PUBLIC_GIT_HOSTS:
        provider = "gitlab" if "gitlab" in repo_host else "github"
    for account in get_git_accounts(provider):
        if _git_account_host(account.get("url"), provider) == repo_host:
            return account.get("token") or None
    return None


def get_confluence_creds() -> Dict[str, Optional[str]]:
    """Resolve Confluence configuration: {mode, base_url, token, username, space, mcp_server, mcp_tool}."""
    mode = get_setting("confluence.mode") or os.environ.get("CONFLUENCE_MODE", "direct")
    return {
        "mode": mode.lower().strip(),
        "base_url": get_setting("confluence.base_url") or os.environ.get("CONFLUENCE_BASE_URL"),
        "token": get_secret("confluence.token") or os.environ.get("CONFLUENCE_TOKEN"),
        "username": get_setting("confluence.username") or os.environ.get("CONFLUENCE_USERNAME"),
        "space": get_setting("confluence.space") or os.environ.get("CONFLUENCE_SPACE"),
        "mcp_server": get_setting("confluence.mcp_server") or os.environ.get("CONFLUENCE_MCP_SERVER", "confluence"),
        "mcp_tool": get_setting("confluence.mcp_tool") or os.environ.get("CONFLUENCE_MCP_TOOL"),
    }


def get_integration_config(name: str) -> Dict[str, Any]:
    """Read an integration config stored as JSON under ``integrations.<name>``."""
    raw = get_setting("integrations.%s" % name)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"value": data}
    except Exception as e:
        logger.warning("Integration config %r is not valid JSON: %s", name, e)
        return {}
