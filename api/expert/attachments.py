"""Chat attachment conversion + prompt blocks (conversation-context only).

Attachments ride the ask-turn prompt; they are never indexed into the
product memory. Shares the markitdown pipeline with the knowledge upload
(graceful UTF-8 fallback for plain-text files).
"""

from __future__ import annotations

import html
import importlib
import logging
import os
import tempfile
from typing import Any, List

logger = logging.getLogger(__name__)

#: Max files accepted per ask turn (mirrors ExpertAskRequest.attachment_ids).
MAX_ATTACHMENTS_PER_ASK = 5
#: Per-file cap on the converted Markdown inlined into the runner query.
MAX_ATTACHMENT_CHARS = 50_000

_DEFAULT_UPLOAD_MAX_BYTES = 50 * 1024 * 1024


def upload_max_bytes() -> int:
    """Resolve the upload size limit: admin setting > env > 50 MiB."""
    try:
        from api.config.settings import get_setting

        raw = get_setting("limits.upload_max_bytes")
        if raw and str(raw).strip().isdigit():
            return int(str(raw).strip())
    except Exception:  # pragma: no cover - settings store down
        pass
    raw_env = os.environ.get("UPLOAD_MAX_BYTES", "")
    if raw_env.strip().isdigit():
        return int(raw_env.strip())
    return _DEFAULT_UPLOAD_MAX_BYTES


def convert_via_markitdown(data: bytes, filename: str) -> tuple:
    """Try api.formats.markitdown.convert_to_markdown with common conventions.

    Returns ``(ok, markdown_or_error)``. We import it lazily and tolerate
    either a path-based or bytes-based ``convert_to_markdown`` signature.
    """
    try:
        md = importlib.import_module("api.formats.markitdown")
    except Exception as e:  # module absent -> degrade
        return (False, f"markitdown unavailable: {e}")
    convert = getattr(md, "convert_to_markdown", None)
    if convert is None:
        return (False, "formats.markitdown.convert_to_markdown not found")

    suffix = os.path.splitext(filename or "")[1]
    tmp_path = None
    try:
        # Persist bytes to a temp file so path-based wrappers work too.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp_path = tf.name
        for args, kwargs in (
            ((tmp_path,), {"filename": filename}),
            ((tmp_path,), {}),
            ((data,), {"filename": filename}),
            ((data,), {}),
        ):
            try:
                result = convert(*args, **kwargs)
            except TypeError:
                continue
            # The wrapper degrades failures to a placeholder comment; that is
            # NOT a real conversion — fall through to the UTF-8/501 path.
            if result and not md.is_placeholder(str(result)):
                return (True, str(result))
        return (False, "markitdown could not convert the file")
    except Exception as e:
        return (False, f"markitdown conversion failed: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def convert_attachment(data: bytes, filename: str) -> str:
    """Convert one upload to a capped Markdown rendition.

    markitdown first; UTF-8 text passthrough when unavailable; ValueError
    when the payload is binary and cannot be rendered.
    """
    ok, output = convert_via_markitdown(data, filename)
    if not ok:
        logger.warning("attachment markitdown degraded: %s", output)
        try:
            output = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(
                "File could not be converted: markitdown is unavailable and "
                "the upload is not UTF-8 text."
            ) from e
    return output[:MAX_ATTACHMENT_CHARS]


def build_attachment_blocks(rows: List[Any]) -> str:
    """Render attachment rows as structured prompt blocks.

    The name attribute is HTML-escaped so a crafted filename cannot forge or
    close the structural prompt tags.
    """
    parts: List[str] = []
    for row in rows:
        name = html.escape((getattr(row, "filename", "") or "attachment")[:256], quote=True)
        content = (getattr(row, "content_md", "") or "").strip()
        parts.append(f'<attachment name="{name}">\n{content}\n</attachment>')
    return "\n\n".join(parts)
