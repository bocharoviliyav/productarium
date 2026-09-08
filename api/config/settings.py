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
import threading
import time
from typing import Optional, List, Dict, Any, Tuple
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

    Precedence: ``DEEPWIKI_CONFIG_DIR`` (same override as the config loader) >
    ``PRODUCTARIUM_STATE_DIR`` (the mounted state volume in docker-compose, so
    the key survives container rebuilds) > ``~/.adalflow/.settings_secret_key``.
    """
    for env_var in ("DEEPWIKI_CONFIG_DIR", "PRODUCTARIUM_STATE_DIR"):
        base = os.environ.get(env_var)
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
# P1-13: short in-process TTL cache so hot read paths (per-request timeout
# resolution via api.config.timeout, rate limits, embedder settings) stop doing
# a synchronous DB round-trip on every call. Writes invalidate immediately;
# stale reads are bounded by the TTL (default 5s, env-tunable).
_SETTINGS_CACHE: Dict[str, Tuple[float, Any]] = {}
_SETTINGS_CACHE_LOCK = threading.Lock()
_SETTINGS_CACHE_MISS = object()  # sentinel: key present in cache as "not set"
_SETTINGS_CACHE_TTL_SECONDS = 5.0


def _settings_cache_ttl() -> float:
    raw = os.environ.get("SETTINGS_CACHE_TTL_SECONDS")
    if raw:
        try:
            val = float(str(raw).strip())
            if val >= 0:
                return val
        except ValueError:
            pass
    return _SETTINGS_CACHE_TTL_SECONDS


def clear_settings_cache() -> None:
    """Drop all cached setting values (tests, admin "apply now" paths)."""
    with _SETTINGS_CACHE_LOCK:
        _SETTINGS_CACHE.clear()


def _cache_get(key: str) -> Tuple[bool, Any]:
    """Return (hit, value) from the TTL cache. value may be _SETTINGS_CACHE_MISS."""
    now = time.monotonic()
    with _SETTINGS_CACHE_LOCK:
        entry = _SETTINGS_CACHE.get(key)
        if entry is None:
            return False, None
        ts, value = entry
        if now - ts > _settings_cache_ttl():
            _SETTINGS_CACHE.pop(key, None)
            return False, None
        return True, value


def _cache_put(key: str, value: Any) -> None:
    with _SETTINGS_CACHE_LOCK:
        _SETTINGS_CACHE[key] = (time.monotonic(), value)


def _cache_drop(key: str) -> None:
    with _SETTINGS_CACHE_LOCK:
        _SETTINGS_CACHE.pop(key, None)


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting by key. Decrypts if the row is marked encrypted.

    Serves from a short TTL cache (P1-13): a hit (including a cached "key is
    not set") skips the DB round-trip; ``set_setting`` / ``delete_setting``
    invalidate the key immediately.
    """
    hit, cached = _cache_get(key)
    if hit:
        return default if cached is _SETTINGS_CACHE_MISS else cached
    try:
        from api.db import SessionLocal
        from api.models import SettingORM
        with SessionLocal() as db:
            row = db.get(SettingORM, key)
            if row is None:
                _cache_put(key, _SETTINGS_CACHE_MISS)
                return default
            if row.encrypted:
                f = _fernet()
                if f is None or not row.value:
                    _cache_put(key, _SETTINGS_CACHE_MISS)
                    return default
                try:
                    value = f.decrypt(row.value.encode("utf-8")).decode("utf-8")
                    _cache_put(key, value)
                    return value
                except Exception as e:
                    logger.warning("Failed to decrypt setting %r: %s", key, e)
                    _cache_put(key, _SETTINGS_CACHE_MISS)
                    return default
            _cache_put(key, row.value)
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
        _cache_drop(key)  # P1-13: writes invalidate immediately
    except Exception as e:
        logger.warning("set_setting(%r) failed (DB down?): %s", key, e)
        _cache_drop(key)


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a (possibly encrypted) secret. Alias for get_setting (which decrypts)."""
    return get_setting(key, default=default)


# --- Standalone secret crypto (P0-2: git tokens on codebases) ----------------
def encrypt_secret(plaintext: str) -> Optional[str]:
    """Encrypt a standalone secret (e.g. a per-codebase git token).

    Returns the Fernet ciphertext string, or None when crypto is unavailable —
    callers decide whether to fall back to plaintext storage (never crash).
    """
    f = _fernet()
    if f is None:
        return None
    try:
        return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.warning("encrypt_secret failed: %s", e)
        return None


def decrypt_secret(ciphertext: str) -> Optional[str]:
    """Decrypt a secret produced by ``encrypt_secret``.

    Returns the plaintext, or None when decryption fails (wrong key, corrupt
    data, crypto unavailable) — never raises.
    """
    f = _fernet()
    if f is None:
        return None
    try:
        return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.warning("decrypt_secret failed: %s", e)
        return None


def is_encrypted_secret(value: Optional[str]) -> bool:
    """Heuristic: Fernet tokens always start with the versioned prefix 'gAAAA'.

    Used by the lazy plaintext->ciphertext migration for codebase git tokens:
    anything else is treated as legacy plaintext.
    """
    return bool(value) and value.startswith("gAAAA")


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
    finally:
        _cache_drop(key)  # P1-13: deletes invalidate immediately


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
    junk value cannot crash callers.
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
    """Resolve a model config for a task (docgen/expert/summary/embedder).

    Reads keys ``models.<task>.{model,base_url,api_key}`` from the settings
    store, falling back to environment variables when unset. Also reads the
    optional ``models.<task>.max_prompt_tokens`` (int) used to cap prompt
    budgets: ``None`` when unset (callers keep their default); non-numeric
    stored values are ignored (treated as unset) so a bad value never crashes
    callers.

    Every supported local server (LM Studio, llama.cpp, vLLM, ...)
    exposes an OpenAI-compatible ``/v1`` API, so a single defaults path covers
    all cases.
    """
    p = "models.%s." % task
    # OpenAI-compatible defaults: LOCAL_OPENAI_BASE_URL points at the server
    # (LM Studio :1234, llama.cpp, vLLM, ...). The same defaults work for
    # every local server.
    default_base = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1")
    default_model = os.environ.get("LOCAL_OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "qwen/qwen3.6-27b"
    default_key = os.environ.get("LOCAL_OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "not-needed"
    return {
        "model": get_setting(p + "model") or default_model,
        "base_url": get_setting(p + "base_url") or default_base,
        "api_key": _sanitize_api_key(get_secret(p + "api_key")) or default_key,
        "max_prompt_tokens": _parse_int_setting(get_setting(p + "max_prompt_tokens")),
        "dimensions": _parse_int_setting(get_setting(p + "dimensions")),
    }


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
    """Resolve Confluence configuration: {base_url, token, username, space}.

    (The former ``mode``/``mcp_server``/``mcp_tool`` keys were removed together
    with the legacy hand-written MCP client — external MCP servers are now
    managed by the Wave C MCP platform, see ``api/mcp/``.)
    """
    return {
        "base_url": get_setting("confluence.base_url") or os.environ.get("CONFLUENCE_BASE_URL"),
        "token": get_secret("confluence.token") or os.environ.get("CONFLUENCE_TOKEN"),
        "username": get_setting("confluence.username") or os.environ.get("CONFLUENCE_USERNAME"),
        "space": get_setting("confluence.space") or os.environ.get("CONFLUENCE_SPACE"),
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
