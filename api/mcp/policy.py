"""Security policy for stdio MCP servers (Wave C hardening).

Shared by ``api.routers.mcp_admin`` (400 on registration/update) and
``api.mcp.manager.build_connection`` (refuse to connect legacy rows), so the
same rules apply at write time AND at connect time.

The stdio transport spawns a LOCAL SUBPROCESS from admin-supplied config.
A plain binary (e.g. ``/usr/local/bin/mcp-server-foo``) is the intended use;
shells, interpreters and script runners are NOT, because through them the
separately-validated ``args``/``env`` fields collapse into arbitrary code
execution (``/bin/sh -c …``, ``python -c …``, ``npx <anything>`` …). Env keys
that reshape process/interpreter behavior (``PATH``, ``LD_*``,
``PYTHONPATH``, …) are likewise rejected — the subprocess inherits the
server's environment, and those keys turn benign binaries into arbitrary-code
launchers.

Import-safe: stdlib only, no I/O at import time.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

#: Basenames that are shells, interpreters or generic runners — with them the
#: validated ``args`` list becomes an arbitrary code payload (``-c``, ``-e`` …).
_INTERPRETER_BASENAMES = frozenset(
    {
        # shells
        "sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "ash", "fish",
        "mksh",
        # interpreters
        "ruby", "irb", "perl", "php", "lua", "luajit", "tclsh", "osascript",
        "expect",
        # generic runners / process trampolines
        "env", "nohup", "timeout", "stdbuf", "setsid", "script", "xargs",
        "sed", "awk", "gawk",
        # package/script runners
        "node", "nodejs", "deno", "bun", "npx", "uvx", "make", "cargo", "go",
    }
)
#: Basenames matched by prefix so version suffixes are covered
#: (``python``, ``python3``, ``python3.12``, ``pypy3`` …).
_INTERPRETER_PREFIXES = ("python", "pypy")

#: Env keys that must never appear in a stdio subprocess environment:
#: interpreter injection (``PYTHONPATH`` …), shell startup files, and the
#: dynamic-loader family is covered by the prefixes below.
_FORBIDDEN_ENV_EXACT = frozenset(
    {
        "PATH", "IFS", "ENV", "BASH_ENV", "ZDOTDIR", "SHELLOPTS", "BASHOPTS",
        "PYTHONPATH", "PYTHONHOME", "NODE_PATH", "NODE_OPTIONS",
        "RUBYOPT", "RUBYLIB", "PERL5OPT", "PERLLIB",
    }
)
#: Env-key prefixes that hijack the dynamic loader (``LD_PRELOAD``,
#: ``LD_LIBRARY_PATH``, ``DYLD_INSERT_LIBRARIES`` …).
_FORBIDDEN_ENV_PREFIXES = ("LD_", "DYLD_")


def stdio_command_error(command: Any) -> Optional[str]:
    """Return an error string when ``command`` may not spawn a stdio MCP server.

    Blocks shells, interpreters and script runners (lowercase basename,
    version suffixes included) and any command containing whitespace.
    ``None`` means "no objection". The HTTP-level shape checks
    (metacharacters, ``..``, control chars) stay in the router — this module
    is transport-policy only and never raises.
    """
    base = os.path.basename(str(command or "").strip()).lower()
    if not base:
        return "stdio command is required"
    if any(ch.isspace() for ch in base):
        return "stdio command must be a single executable path (no whitespace)"
    if base in _INTERPRETER_BASENAMES or base.startswith(_INTERPRETER_PREFIXES):
        return (
            f"stdio command {base!r} is a shell/interpreter/runner and is not "
            "allowed — spawn the MCP server binary directly (no -c/npx/uvx wrappers)"
        )
    return None


def stdio_env_error(env: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return an error string when ``env`` contains forbidden variables.

    Blocks PATH-like injection, interpreter/shell overrides and dynamic-loader
    overrides (prefix match). Comparison is case-insensitive. ``None`` = ok.
    """
    for key in env or {}:
        upper = str(key).upper()
        if upper in _FORBIDDEN_ENV_EXACT or upper.startswith(_FORBIDDEN_ENV_PREFIXES):
            return (
                f"stdio env variable {key!r} is not allowed "
                "(PATH/loader/interpreter overrides are rejected)"
            )
    return None


__all__ = ["stdio_command_error", "stdio_env_error"]
