"""Filesystem read helpers shared by the agent and docgen file tools."""

from __future__ import annotations

import os
from typing import IO


def open_read_nofollow(path: str, *, binary: bool = False, errors: str = "strict") -> IO:
    """Open ``path`` for reading, refusing a symlink on the FINAL component.

    The path-confinement checks in the agent/docgen tools (``realpath`` +
    ``commonpath``) resolve symlinks once and return the resolved path; without
    ``O_NOFOLLOW`` a symlink swapped onto that final component between the
    check and the ``open`` (a TOCTOU race) would still be followed outside the
    confined root. This helper closes that window on the final component.
    Intermediate-directory races remain theoretically possible but require an
    attacker with write access inside the clone — outside this threat model.

    Raises ``OSError`` (``ELOOP`` where supported) when the final component is
    a symlink — callers' existing ``except OSError`` paths handle it like any
    other unreadable file. Platforms without ``O_NOFOLLOW`` keep plain
    ``open`` semantics (best-effort hardening).
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    if binary:
        flags |= getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        if binary:
            return os.fdopen(fd, "rb")
        return os.fdopen(fd, "r", encoding="utf-8", errors=errors)
    except BaseException:
        os.close(fd)
        raise


__all__ = ["open_read_nofollow"]
