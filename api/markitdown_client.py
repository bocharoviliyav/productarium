"""Thin wrapper over the ``markitdown`` package (plan section G / item 2).

Converts uploaded files and Confluence attachments (docx/pdf/pptx/html/xlsx/
images/...) to markdown. The ``markitdown`` import is lazy so this module is
always import-safe even when the package (or one of its optional converter
deps) is missing — callers get a clear placeholder string instead of an
import error, and the app never fails to boot.

Public API::

    from api.markitdown_client import convert_to_markdown
    md = convert_to_markdown("/path/to/file.docx")
    md = convert_to_markdown(raw_bytes, filename="report.pdf")
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Whether the markitdown package imported successfully. Resolved on first use.
_MARKITDOWN_AVAILABLE: Optional[bool] = None
_MARKITDOWN = None  # type: ignore


def _load_markitdown():
    """Lazy-import markitdown. Returns the MarkItDown class or None."""
    global _MARKITDOWN_AVAILABLE, _MARKITDOWN
    if _MARKITDOWN_AVAILABLE is not None:
        return _MARKITDOWN
    try:
        from markitdown import MarkItDown  # type: ignore

        _MARKITDOWN = MarkItDown
        _MARKITDOWN_AVAILABLE = True
        logger.debug("markitdown loaded successfully.")
    except Exception as e:  # pragma: no cover - dep missing
        _MARKITDOWN_AVAILABLE = False
        logger.warning(
            "markitdown is not available; file conversions will return a "
            "placeholder. Install `markitdown` to enable conversions: %s",
            e,
        )
    return _MARKITDOWN


def _placeholder(filename: Optional[str], reason: str) -> str:
    """A clear, non-empty placeholder returned when conversion is unavailable."""
    name = filename or "unknown"
    return f"<!-- markitdown: conversion unavailable for {name!r}: {reason} -->\n"


def convert_to_markdown(
    file_path_or_bytes: Union[str, bytes, "os.PathLike", "io.IOBase"],
    filename: Optional[str] = None,
) -> str:
    """Convert a file or raw bytes to markdown using ``markitdown``.

    Args:
        file_path_or_bytes: A filesystem path (str/PathLike) OR raw bytes. When
            bytes are passed, ``filename`` should be provided so markitdown can
            infer the format from the extension.
        filename: Original filename (used for bytes + as a hint for the format).
            Ignored when a path is passed (the path's own name is used).

    Returns:
        Markdown text. When markitdown (or a converter for the given type) is
        unavailable, returns a placeholder HTML comment string instead of
        raising, so callers and background indexing never crash.
    """
    MarkItDown = _load_markitdown()
    if MarkItDown is None:
        return _placeholder(filename, "markitdown package not installed")

    try:
        converter = MarkItDown()
        if isinstance(file_path_or_bytes, (str, os.PathLike)):
            # Filesystem path — markitdown infers the format from the extension.
            result = converter.convert(str(file_path_or_bytes))
        elif isinstance(file_path_or_bytes, (bytes, bytearray)):
            # Raw bytes — wrap in a BytesIO with a filename hint. markitdown
            # sniffs the format from the stream's `.name` attribute when set.
            stream = io.BytesIO(bytes(file_path_or_bytes))
            if filename:
                stream.name = filename  # type: ignore[attr-defined]
            result = converter.convert(stream)
        elif hasattr(file_path_or_bytes, "read"):
            # Already a file-like object.
            if filename and not getattr(file_path_or_bytes, "name", None):
                try:
                    file_path_or_bytes.name = filename  # type: ignore[attr-defined]
                except Exception:
                    pass
            result = converter.convert(file_path_or_bytes)
        else:
            return _placeholder(filename, f"unsupported input type {type(file_path_or_bytes)!r}")
        text = getattr(result, "text_content", None)
        if text is None:
            # Newer markitdown versions expose `.markdown` or stringify.
            text = getattr(result, "markdown", None) or str(result)
        return text or ""
    except Exception as e:
        logger.warning(
            "markitdown conversion failed for %r: %s",
            filename or getattr(file_path_or_bytes, "name", "<path>"),
            e,
        )
        return _placeholder(filename, f"conversion error: {e}")
