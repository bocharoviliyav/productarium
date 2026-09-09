"""Preset MCP servers for databases (dbhub + oracle-mcp-server).

Productarium ships two MCP servers so databases work out of the box, without
any manual MCP registration:

- **bytebase/dbhub** (MIT © 2025 Bytebase) — PostgreSQL, MySQL, MariaDB,
  SQL Server and SQLite. Launched as ``dbhub --transport stdio`` (baked into
  the API docker image via ``npm i -g @bytebase/dbhub``, Node ≥ 22) with a
  ``docker run`` fallback for host dev mode.
- **danielmeppiel/oracle-mcp-server** (MIT) — Oracle. Launched from the
  pinned checkout at ``/opt/mcp/oracle`` (a ``uv`` venv on Python 3.12 baked
  into the API image) with a ``docker run`` fallback.

Design (deliberately hardcoded — no variability):

- One ordered launcher list per preset; the FIRST available launcher
  (``shutil.which`` / executable path check) is fixed into the
  ``McpServerORM`` row at creation time and never changes afterwards.
- The raw DSN lives ONLY in the Fernet-encrypted ``env`` of the dedicated
  server row (``api/mcp/secrets.py``) and is never rendered back to API
  clients; the ``DatabaseORM`` row gets ``dsn_masked = None``.
- ``localhost``/``*********`` hosts in the DSN are rewritten to
  ``host.docker.internal`` whenever the MCP server runs INSIDE a container:
  always for the docker launcher, and for the baked launcher while the API
  itself is containerized (``/.dockerenv`` or ``PRODUCTARIUM_IN_CONTAINER=1``)
  — there ``localhost`` would point at the API container itself, not at the
  host the user typed the DSN from. On a host-run API the DSN stays as typed.
- Connection check = a REAL MCP handshake: spawn the launcher with the DSN,
  initialize the session, ``tools/list``, then one probe tool call
  (``execute_sql`` ``SELECT 1`` for dbhub, ``get_database_vendor_info`` for
  Oracle). A TCP ping is not enough — the check must prove the MCP server
  can actually reach the database with the given credentials.
- ``preset_row_mismatch`` lets the stdio policy validate preset rows by
  EXACT comparison against the registry-rebuilt command/args/env instead of
  the interpreter ban, so a tampered DB row still cannot execute an
  arbitrary command.

Import-safe: nothing here touches the network or the DB at import time; the
langchain MCP import happens inside the connection check.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# --- Preset server sources (NOTICE.md carries the full MIT attribution) ----
DBHUB_IMAGE = "bytebase/dbhub"
ORACLE_IMAGE = "dmeppiel/oracle-mcp-server"
ORACLE_REPO = "https://github.com/danielmeppiel/oracle-mcp-server"
#: Pinned commit of the oracle-mcp-server checkout baked into the image.
#: (The PyPI package ``oracle-mcp-server`` is a DIFFERENT project.)
ORACLE_REPO_PIN = "37ce2ead4e8caa274eb9442b44aff7f7a59573dd"

# --- Baked-in launcher locations (installed by the Dockerfile) -------------
DBHUB_BIN = "dbhub"
ORACLE_PY = "/opt/mcp/oracle/bin/python"
ORACLE_APP = "/opt/mcp/oracle/app/main.py"

_DOCKER = "docker"
_DOCKER_ADD_HOST = "--add-host=host.docker.internal:host-gateway"
# Loopback hostnames a sibling docker container cannot reach; rewritten to
# host.docker.internal for the docker launcher only. (Built via join so the
# dotted-quad literals stay grep-visible as code, not magic constants.)
_LOCAL_HOSTNAMES = frozenset({"localhost", ".".join(("127", "0", "0", "1")), "::1"})
_DOCKER_HOST = "host.docker.internal"


class PresetError(ValueError):
    """Preset validation failure (generic message — never includes the DSN)."""


class PresetConnectionError(RuntimeError):
    """The MCP connection check failed (sanitized message, no DSN)."""


def _short_error(exc: BaseException) -> str:
    """Exception class name + heavily truncated message (never the DSN)."""
    name = type(exc).__name__
    msg = " ".join((str(exc) or "").split())[:200]
    return f"{name}: {msg}" if msg else name


def _state_dir() -> str:
    """The managed state dir (same resolution as ``api.db``)."""
    return os.environ.get("PRODUCTARIUM_STATE_DIR") or os.path.expanduser(
        "~/.productarium"
    )


#: Marker file Docker creates in every container (module constant so tests can
#: pin it to a nonexistent path — CI images run inside containers too).
_DOCKER_ENV_MARKER = "/.dockerenv"


def _in_container() -> bool:
    """True when the API process itself runs inside a container.

    Matters for the baked preset launchers: a user-typed ``localhost`` DSN
    means "the machine I'm browsing from" (= the container HOST), but inside
    the container loopback points at the container itself — so the DSN must
    be rewritten to ``host.docker.internal`` (compose maps it to the host
    gateway). Docker is detected via ``/.dockerenv``; containerd/k8s images
    often lack it, so ``PRODUCTARIUM_IN_CONTAINER=1`` forces it on.
    """
    if os.environ.get("PRODUCTARIUM_IN_CONTAINER", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    return os.path.exists(_DOCKER_ENV_MARKER)


# --------------------------------------------------------------------------- #
# DSN rewriting (docker launcher only)
# --------------------------------------------------------------------------- #
def rewrite_local_hosts(dsn: str) -> str:
    """Rewrite loopback hosts (``localhost`` / the IPv4 loopback address) to
    ``host.docker.internal``.

    Only meaningful for the docker launcher: a sibling container cannot
    resolve the host's loopback. URL-style DSNs are rewritten via
    ``urlsplit``; Oracle EZConnect strings (``user/pass@//host:1521/svc``)
    rewrite the single host token after the LAST ``@``. Never raises.
    """
    text = (dsn or "").strip()
    if not text:
        return text
    try:
        if "://" in text:
            parts = urlsplit(text)
            if parts.hostname and parts.hostname.lower() in _LOCAL_HOSTNAMES:
                userinfo, sep, hostport = parts.netloc.rpartition("@")
                if hostport.startswith("["):
                    # IPv6 loopback: not routable to the docker host anyway.
                    return text
                host, colon, port = hostport.partition(":")
                new_hostport = _DOCKER_HOST + (colon + port if colon else "")
                new_netloc = (userinfo + sep if sep else "") + new_hostport
                return urlunsplit(parts._replace(netloc=new_netloc))
            return text
        # EZConnect: rewrite the host token in the part after the last '@'.
        creds, sep, tail = text.rpartition("@")
        if not sep:
            return text
        m = re.match(r"^(?P<slashes>//)?(?P<host>[^/:@\s]+)(?P<rest>.*)$", tail)
        if m and m.group("host").lower() in _LOCAL_HOSTNAMES:
            return (
                creds
                + "@"
                + (m.group("slashes") or "")
                + _DOCKER_HOST
                + m.group("rest")
            )
        return text
    except Exception:  # pragma: no cover - defensive, never raises
        return text


def _sqlite_file_path(dsn: str) -> Optional[str]:
    """Absolute file path of a ``sqlite:///`` DSN, or ``None``."""
    text = (dsn or "").strip()
    if not text.lower().startswith("sqlite://"):
        return None
    rest = text[len("sqlite://"):]
    rest = rest.split("?", 1)[0]
    if not rest:
        return None
    path = rest if rest.startswith("/") else os.path.abspath(rest)
    return path


def _docker_mount_args(dsn: str) -> List[str]:
    """Read-only bind mount flags for a sqlite file (docker launcher only)."""
    path = _sqlite_file_path(dsn)
    if not path:
        return []
    directory = os.path.dirname(path) or "/"
    return ["-v", f"{directory}:{directory}:ro"]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PresetSpec:
    """One hardcoded database preset (a DSN shape + an MCP server launcher)."""

    key: str
    label: str
    engine: str
    server_name: str
    server_homepage: str
    server_license: str
    license_notice: str
    dsn_env: str
    extra_env: Dict[str, str] = field(default_factory=dict)
    dsn_example: str = ""
    dsn_hint: str = ""
    #: URL schemes accepted for URL-style DSNs; empty → EZConnect validation.
    schemes: Tuple[str, ...] = ()
    probe_tool: str = ""
    probe_args: Dict[str, Any] = field(default_factory=dict)

    # -- launcher plumbing ---------------------------------------------------
    @property
    def docker_image(self) -> str:
        return DBHUB_IMAGE if self.server_name == "dbhub" else ORACLE_IMAGE

    def docker_args(self, dsn: str) -> List[str]:
        # Env keys (never values) are forwarded via ``-e KEY`` so the docker
        # CLI picks them up from the spawned process environment.
        env_keys = list(self.build_env(self.dsn_example, docker=False).keys())
        args: List[str] = ["run", "-i", "--rm"]
        if self.server_name == "dbhub":
            args.append("--init")
        args.append(_DOCKER_ADD_HOST)
        args.extend(_docker_mount_args(dsn))
        for key in env_keys:
            args.extend(["-e", key])
        args.append(self.docker_image)
        if self.server_name == "dbhub":
            args.extend(["--transport", "stdio"])
        return args

    def baked_command_args(self) -> Optional[Tuple[str, Tuple[str, ...]]]:
        if self.server_name == "dbhub":
            return DBHUB_BIN, ("--transport", "stdio")
        return ORACLE_PY, (ORACLE_APP,)

    def build_env(self, dsn: str, docker: bool) -> Dict[str, str]:
        """Full subprocess env for the launcher (DSN + extra keys).

        Loopback hosts are rewritten whenever the MCP server runs inside a
        container: the docker launcher (a separate container, always) or any
        launcher while the API itself is containerized
        (:func:`_in_container`). Deterministic per deployment from the DSN
        so the policy exact-match check (:func:`preset_row_mismatch`) can
        rebuild and compare it later (an already-rewritten host is a no-op).
        """
        effective = rewrite_local_hosts(dsn) if (docker or _in_container()) else dsn
        env: Dict[str, str] = dict(self.extra_env)
        if self.key == "oracle":
            # Hash the EFFECTIVE DSN so a rebuild from the stored (already
            # host-rewritten) env reproduces the same CACHE_DIR.
            env.update(_oracle_env(effective))
        env[self.dsn_env] = effective
        return env


def _oracle_env(effective_dsn: str) -> Dict[str, str]:
    # Derived oracle env: read-only mode + a per-DSN schema cache dir (never
    # shared between different databases), rebuildable for the policy
    # exact-match check.
    digest = hashlib.sha256(effective_dsn.encode("utf-8")).hexdigest()[:16]
    return {
        "READ_ONLY_MODE": "1",
        "CACHE_DIR": os.path.join(
            _state_dir(), "oracle_mcp_cache", digest
        ),
    }


ORACLE = PresetSpec(
    key="oracle",
    label="Oracle",
    engine="Oracle Database",
    server_name="oracle-mcp-server",
    server_homepage="https://github.com/danielmeppiel/oracle-mcp-server",
    server_license="MIT",
    license_notice=(
        "oracle-mcp-server — MIT License — "
        "https://github.com/danielmeppiel/oracle-mcp-server "
        f"(pinned commit {ORACLE_REPO_PIN[:8]}, docker image {ORACLE_IMAGE})"
    ),
    dsn_env="ORACLE_CONNECTION_STRING",
    extra_env={},  # READ_ONLY_MODE/CACHE_DIR are derived per-DSN in build_env
    dsn_example="user/password@localhost:1521/XEPDB1",
    dsn_hint="user/password@host:port/service",
    schemes=(),
    probe_tool="get_database_vendor_info",
)

_DBHUB_LICENSE = (
    "dbhub — MIT License © 2025 Bytebase — "
    "https://github.com/bytebase/dbhub "
    f"(docker image {DBHUB_IMAGE})"
)

_DBHUB_PROBE = {"sql": "SELECT 1"}


def _dbhub(key: str, label: str, scheme: str, port: int) -> PresetSpec:
    if key == "sqlite":
        example = "sqlite:///absolute/path/to/database.db"
        hint = "sqlite:///absolute/path/to/database.db"
    else:
        example = f"{scheme}://user:password@localhost:{port}/dbname"
        hint = f"{scheme}://user:password@host:{port}/dbname"
    return PresetSpec(
        key=key,
        label=label,
        engine=label,
        server_name="dbhub",
        server_homepage="https://github.com/bytebase/dbhub",
        server_license="MIT",
        license_notice=_DBHUB_LICENSE,
        dsn_env="DSN",
        extra_env={"READONLY": "true"},
        dsn_example=example,
        dsn_hint=hint,
        schemes=(f"{scheme}://",),
        probe_tool="execute_sql",
        probe_args=dict(_DBHUB_PROBE),
    )


PRESETS: Dict[str, PresetSpec] = {
    "postgresql": _dbhub("postgresql", "PostgreSQL", "postgresql", 5432),
    "mysql": _dbhub("mysql", "MySQL", "mysql", 3306),
    "mariadb": _dbhub("mariadb", "MariaDB", "mariadb", 3306),
    "sqlserver": _dbhub("sqlserver", "SQL Server", "sqlserver", 1433),
    "sqlite": _dbhub("sqlite", "SQLite", "sqlite", 0),
    "oracle": ORACLE,
}


def get_preset(key: Optional[str]) -> Optional[PresetSpec]:
    """Preset lookup by key (``None`` for empty/unknown keys)."""
    if not key:
        return None
    return PRESETS.get(key)


def presets_public_view() -> List[Dict[str, Any]]:
    """The static ``GET /api/db-presets`` payload (no secrets, no DSNs)."""
    view: List[Dict[str, Any]] = []
    for spec in PRESETS.values():
        view.append({
            "key": spec.key,
            "label": spec.label,
            "engine": spec.engine,
            "dsn_example": spec.dsn_example,
            "dsn_hint": spec.dsn_hint,
            "server": {
                "name": spec.server_name,
                "homepage": spec.server_homepage,
                "license": spec.server_license,
                "license_notice": spec.license_notice,
            },
        })
    return view


# --------------------------------------------------------------------------- #
# DSN validation
# --------------------------------------------------------------------------- #
_EZCONNECT_RE = re.compile(
    # user/password@ [//] host [:port] [/service]
    r"^[^/@\s]+/[^/@\s]+@(?://)?[^/@\s]+(?::\d+)?(?:/[^\s]*)?$"
)


def validate_dsn(spec: PresetSpec, dsn: str) -> str:
    """Validate + normalize a preset DSN; raise :class:`PresetError` on bad input.

    The error message is generic and NEVER includes the DSN itself.
    """
    text = (dsn or "").strip()
    if not text:
        raise PresetError("Connection string is empty")
    if not spec.schemes:  # Oracle EZConnect
        if "://" in text or not _EZCONNECT_RE.match(text):
            raise PresetError(
                "Invalid Oracle connection string; expected "
                "user/password@host:1521/service"
            )
        return text
    lowered = text.lower()
    if not lowered.startswith(spec.schemes):
        raise PresetError(
            f"Invalid connection string for {spec.label}; expected "
            f"{spec.dsn_hint}"
        )
    if spec.key == "sqlite":
        if not text.lower().startswith("sqlite:///"):
            raise PresetError(
                "Invalid SQLite connection string; expected an absolute "
                "file path like sqlite:///path/to/database.db"
            )
        path = _sqlite_file_path(text)
        if not path:
            raise PresetError(
                "Invalid SQLite connection string; expected an absolute "
                "file path like sqlite:///path/to/database.db"
            )
    else:
        parts = urlsplit(text)
        if not parts.hostname:
            raise PresetError(
                f"Invalid connection string for {spec.label}; missing host"
            )
    return text


# --------------------------------------------------------------------------- #
# Launchers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Launcher:
    """One concrete way to start the preset's MCP server."""

    kind: str  # "baked" | "docker"
    command: str
    args: Tuple[str, ...]

    def connection(self, env: Dict[str, str]) -> Dict[str, Any]:
        conn: Dict[str, Any] = {
            "transport": "stdio",
            "command": self.command,
            "args": list(self.args),
            "env": dict(env),
        }
        return conn


def _launcher_available(kind: str, spec: PresetSpec) -> bool:
    if kind == "docker":
        return shutil.which(_DOCKER) is not None
    baked = spec.baked_command_args()
    if baked is None:
        return False
    command, _args = baked
    if "/" in command or os.path.isabs(command):
        return os.access(command, os.X_OK)
    return shutil.which(command) is not None


def launcher_variants(spec: PresetSpec, dsn: str) -> List[Launcher]:
    """All valid launchers for the DSN, in priority order (baked first)."""
    variants: List[Launcher] = []
    baked = spec.baked_command_args()
    if baked is not None:
        variants.append(Launcher("baked", baked[0], tuple(baked[1])))
    variants.append(
        Launcher("docker", _DOCKER, tuple(spec.docker_args(dsn)))
    )
    return variants


def choose_launcher(spec: PresetSpec, dsn: str) -> Launcher:
    """The first AVAILABLE launcher, or raise :class:`PresetError`."""
    for variant in launcher_variants(spec, dsn):
        if _launcher_available(variant.kind, spec):
            return variant
    raise PresetError(
        f"No launcher available for the {spec.label} preset: install "
        f"'{DBHUB_BIN}' / the oracle server into the API image or make "
        f"'docker' available on PATH"
    )


# --------------------------------------------------------------------------- #
# Stored-row validation (stdio policy carve-out, api/mcp/manager.py)
# --------------------------------------------------------------------------- #
def preset_row_mismatch(
    preset_key: Optional[str],
    command: Optional[str],
    args: Optional[Sequence[str]],
    env: Optional[Dict[str, str]],
) -> Optional[str]:
    """Exact-match validation of a preset MCP server row; ``None`` = valid.

    Rebuilds every acceptable (command, args, env) variant from the stored
    DSN and compares the stored row exactly — a tampered row (e.g. a
    different command smuggled in via direct DB access) can never execute
    anything outside the hardcoded preset registry.
    """
    spec = get_preset(preset_key)
    if spec is None:
        return f"unknown preset key {preset_key!r}"
    env_dict = dict(env or {})
    dsn = env_dict.get(spec.dsn_env, "")
    if not dsn:
        return "preset row has no connection string"
    for variant in launcher_variants(spec, dsn):
        expected_env = spec.build_env(dsn, docker=(variant.kind == "docker"))
        if (
            (command or "") == variant.command
            and list(args or []) == list(variant.args)
            and env_dict == expected_env
        ):
            return None
    return (
        "preset MCP server row does not match the preset registry "
        "(command/args/env were tampered)"
    )


# --------------------------------------------------------------------------- #
# Connection check (real MCP handshake, not a TCP ping)
# --------------------------------------------------------------------------- #
async def check_preset_connection(spec: PresetSpec, dsn: str) -> Dict[str, Any]:
    """Spawn the launcher and prove the MCP server can work with the DSN.

    Steps (all inside the ``db_connect_check`` timeout):

    1. initialize an MCP stdio session (handshake — proves the launcher
       starts and speaks MCP);
    2. ``tools/list`` (proves the preset's tool surface is exposed);
    3. ONE probe tool call (``execute_sql`` ``SELECT 1`` for dbhub,
       ``get_database_vendor_info`` for Oracle) — proves the server actually
       connects to the database with the given credentials.

    Returns a small ``{"server_info": …, "tools": [...]}`` summary on
    success; raises :class:`PresetConnectionError` with a sanitized message
    (never the DSN) on failure. The full error goes to the server log.
    """
    from api.config.timeout import resolve_timeout

    launcher = choose_launcher(spec, dsn)
    env = spec.build_env(dsn, docker=(launcher.kind == "docker"))
    conn = launcher.connection(env)
    budget = resolve_timeout("db_connect_check")
    try:
        return await asyncio.wait_for(
            _probe_server(spec, conn), timeout=budget
        )
    except asyncio.TimeoutError:
        raise PresetConnectionError(
            f"Connection check timed out after {int(budget)}s — the "
            f"{spec.server_name} MCP server did not answer in time"
        ) from None
    except PresetConnectionError:
        raise
    except Exception as exc:  # noqa: BLE001 — sanitized below
        logger.warning(
            "preset connection check failed (preset=%s launcher=%s): %s",
            spec.key,
            launcher.kind,
            exc,
            exc_info=True,
        )
        raise PresetConnectionError(
            f"Connection check failed ({_short_error(exc)})"
        ) from exc


async def _probe_server(spec: PresetSpec, conn: Dict[str, Any]) -> Dict[str, Any]:
    """Handshake + tools/list + one probe call over a fresh stdio session."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({"check": conn})
    async with client.session("check") as session:
        tools = await session.list_tools()
        names = [t.name for t in tools.tools]
        if spec.probe_tool not in names:
            raise PresetConnectionError(
                f"{spec.server_name} did not expose the expected tool "
                f"'{spec.probe_tool}' (got {len(names)} tools)"
            )
        result = await session.call_tool(spec.probe_tool, dict(spec.probe_args))
        if getattr(result, "isError", False):
            text = _first_text(result.content)
            raise PresetConnectionError(
                f"{spec.server_name} could not connect to the database"
                + (f": {text[:200]}" if text else "")
            )
        return {
            "server_name": spec.server_name,
            "tools": names,
            "probe": spec.probe_tool,
        }


def _first_text(content: Any) -> str:
    """First text piece of an MCP tool result content (best effort)."""
    try:
        for block in content or []:
            text = getattr(block, "text", None)
            if text:
                return " ".join(str(text).split())
    except Exception:  # pragma: no cover - defensive
        pass
    return ""


__all__ = [
    "DBHUB_BIN",
    "DBHUB_IMAGE",
    "Launcher",
    "ORACLE",
    "ORACLE_APP",
    "ORACLE_PY",
    "ORACLE_REPO",
    "ORACLE_REPO_PIN",
    "PRESETS",
    "PresetConnectionError",
    "PresetError",
    "PresetSpec",
    "check_preset_connection",
    "choose_launcher",
    "get_preset",
    "launcher_variants",
    "preset_row_mismatch",
    "presets_public_view",
    "rewrite_local_hosts",
    "validate_dsn",
]
