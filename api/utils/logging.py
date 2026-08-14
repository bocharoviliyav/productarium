"""Logging configuration — console-only with logfmt/json output.

Thread-safe non-blocking setup: a ``QueueHandler`` feeds a single
``QueueListener`` thread that owns the real ``StreamHandler``, so log calls
from ANY thread (including ``asyncio.to_thread`` workers used by adalflow /
RLM / docgen) never block on a slow/full stdout pipe.

Environment variables:
    LOG_LEVEL: Log level (default: INFO)
    LOG_FORMAT: ``logfmt`` (default) or ``json``
    LOG_MAX_RECORD_CHARS: Truncate each log record to this many chars
        (default: 8192). Protects against multi-MB records (adalflow's
        ``log.info(f"output: {output}")`` dumps the full LLM completion)
        which, when written from a worker thread to a pipe-backed stdout,
        raise ``BlockingIOError`` once the pipe buffer fills.
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

    # Route third-party loggers (adalflow, litellm, httpx, ...) through the
    # root QueueHandler. They attach their OWN StreamHandler(sys.stdout) with
    # propagate=False, which writes synchronously from worker threads and
    # raises BlockingIOError on pipe-backed stdout. Clear their handlers,
    # enable propagation, and raise their level to WARNING to cut INFO spam.
    for _vendor_name in (
        "adalflow", "adalflow.core", "adalflow.utils", "adalflow.components",
        "litellm", "litellm.litellm_logging_utils", "litellm.utils",
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
