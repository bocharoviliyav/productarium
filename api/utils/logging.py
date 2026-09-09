"""Logging configuration — console-only with logfmt/json output.

Thread-safe non-blocking setup: a ``QueueHandler`` feeds a single
``QueueListener`` thread that owns the real ``StreamHandler``, so log calls
from ANY thread (including ``asyncio.to_thread`` workers used by the LLM /
docgen pipelines) never block on a slow/full stdout pipe.

Environment variables:
    LOG_LEVEL: Log level (default: INFO)
    LOG_FORMAT: ``logfmt`` (default) or ``json``
    LOG_MAX_RECORD_CHARS: Truncate each log record to this many chars
        (default: 8192). Protects against multi-MB records (a vendored
        library dumping the full LLM completion) which, when written from a
        worker thread to a pipe-backed stdout, raise ``BlockingIOError``
        once the pipe buffer fills.
"""

from __future__ import annotations

import json
import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener

logger = logging.getLogger(__name__)


class IgnoreLogChangeDetectedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return "Detected file change in" not in record.getMessage()


class DocgenPollNoiseFilter(logging.Filter):
    """Drop the noisy 2-second docgen polling access lines (200 responses).

    Paths: ``generate/status`` (per-job polling) and ``docgen/active``
    (page-mount restore polling) are hit every ~2s by the UI and drown out
    everything else in uvicorn.access. Non-200 responses stay visible —
    those are real errors worth seeing. Attached at the LOGGER level
    (``uvicorn.access``) because uvicorn's dictConfig replaces handlers but
    leaves logger-level filters intact.
    """

    NOISY_PATHS = ("generate/status", "docgen/active")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        if not any(p in msg for p in self.NOISY_PATHS):
            return True
        return not msg.rstrip().endswith(" 200")


class _TruncatingFormatter(logging.Formatter):
    """Formatter that caps record message length and supports json output."""

    def __init__(self, fmt: str = None, use_json: bool = False, max_chars: int = None):
        super().__init__(fmt)
        self._json = use_json
        self._max_chars = max_chars if max_chars is not None else int(
            os.environ.get("LOG_MAX_RECORD_CHARS", "8192")
        )

    def format(self, record: logging.LogRecord) -> str:
        if self._json:
            text = json.dumps(
                {
                    "ts": self.formatTime(record, self.datefmt),
                    "level": record.levelname,
                    "logger": record.name,
                    "file": f"{record.filename}:{record.lineno}",
                    "msg": record.getMessage(),
                },
                ensure_ascii=False,
            )
        else:
            text = super().format(record)
        if self._max_chars > 0 and len(text) > self._max_chars:
            return text[: self._max_chars] + " ... (log record truncated)"
        return text


def setup_logging():
    """Configure console-only logging behind a non-blocking queue.

    A ``QueueHandler`` puts records onto an unbounded ``SimpleQueue`` (never
    blocks, never raises ``BlockingIOError``) and a single ``QueueListener``
    thread owns the real ``StreamHandler``. This isolates blocking writes to
    one dedicated thread regardless of how many worker threads emit logs.
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    use_json = os.environ.get("LOG_FORMAT", "logfmt").lower() == "json"
    fmt = (
        "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
    )
    formatter = _TruncatingFormatter(fmt=fmt, use_json=use_json)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(IgnoreLogChangeDetectedFilter())

    _log_queue: queue.Queue = queue.SimpleQueue()  # type: ignore[assignment]
    queue_handler = QueueHandler(_log_queue)
    queue_handler.setLevel(log_level)
    listener = QueueListener(
        _log_queue, console_handler, respect_handler_level=True
    )
    listener.start()

    logging.basicConfig(level=log_level, handlers=[queue_handler], force=True)

    # Mute the 2s docgen status-poll spam in uvicorn.access (200s only).
    # Attached to the LOGGER, not a handler: uvicorn's own dictConfig (run at
    # uvicorn.run) replaces the access logger's handlers but preserves
    # logger-level filters, so installing it here survives uvicorn setup.
    # Idempotent: setup_logging(force=True) may run more than once.
    uv_access = logging.getLogger("uvicorn.access")
    uv_access.filters = [
        f for f in uv_access.filters if not isinstance(f, DocgenPollNoiseFilter)
    ]
    uv_access.addFilter(DocgenPollNoiseFilter())

    # Route third-party loggers (langchain, httpx, ...) through the
    # root QueueHandler. They attach their OWN StreamHandler(sys.stdout) with
    # propagate=False, which writes synchronously from worker threads and
    # raises BlockingIOError on pipe-backed stdout. Clear their handlers,
    # enable propagation, and raise their level to WARNING to cut INFO spam.
    for _vendor_name in (
        "langchain", "langchain_core", "langchain_openai",
        "httpx", "httpcore", "openai._base_client",
    ):
        try:
            _vlog = logging.getLogger(_vendor_name)
            _vlog.handlers = []
            _vlog.propagate = True
            _vlog.setLevel(logging.WARNING)
        except Exception:
            pass

    logger.debug(
        "Logging configured: level=%s, format=%s, queue_listener_started=True",
        log_level_str,
        "json" if use_json else "logfmt",
    )
