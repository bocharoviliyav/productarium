"""Logging configuration (former ``api/logging_config.py``).

Thread-safe non-blocking logging setup: a ``QueueHandler`` feeds a single
``QueueListener`` thread that owns the real file + stream handlers, so log
calls from ANY thread (including ``asyncio.to_thread`` workers used by
adalflow / RLM / docgen) never block on a slow/full stdout pipe.
"""

from __future__ import annotations

import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)


class IgnoreLogChangeDetectedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return "Detected file change in" not in record.getMessage()


class _TruncatingFormatter(logging.Formatter):
    """Formatter that caps record message length.

    Some libraries (notably ``adalflow``) log the full LLM output verbatim via
    ``log.info(f"output: {output}")``, which can produce single log records of
    tens/hundreds of KB. Writing a record that large to a pipe-backed stdout
    from a worker thread raises ``BlockingIOError: [Errno 11] write could not
    complete without blocking`` (the pipe buffer fills and a blocking write is
    attempted off the event-loop thread). Truncating the formatted message to
    a sane cap removes the worst case without hiding useful diagnostics. The
    cap is env-tunable; default 8 KB per record.
    """

    def __init__(self, fmt: str = None, max_chars: int = None):
        super().__init__(fmt)
        self._max_chars = max_chars if max_chars is not None else int(
            os.environ.get("LOG_MAX_RECORD_CHARS", "8192")
        )

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self._max_chars > 0 and len(text) > self._max_chars:
            return text[: self._max_chars] + " ... (log record truncated)"
        return text


def setup_logging(format: str = None):
    """
    Configure logging for the application with log rotation.

    Environment variables:
        LOG_LEVEL: Log level (default: INFO)
        LOG_FILE_PATH: Path to log file (default: logs/application.log)
        LOG_MAX_SIZE: Max size in MB before rotating (default: 10MB)
        LOG_BACKUP_COUNT: Number of backup files to keep (default: 5)
        LOG_MAX_RECORD_CHARS: Truncate each log record to this many chars
            (default: 8192). Protects against multi-MB records (adalflow's
            ``log.info(f"output: {output}")`` dumps the full LLM completion)
            which, when written from a worker thread to a pipe-backed stdout,
            raise ``BlockingIOError: [Errno 11] write could not complete without
            blocking`` once the pipe buffer fills.

    Ensures log directory exists, prevents path traversal, and configures
    both rotating file and console handlers behind a ``QueueHandler`` /
    ``QueueListener`` pair so log calls from ANY thread (including the
    ``asyncio.to_thread`` workers used by adalflow / RLM / docgen) NEVER block
    on a slow/full stdout pipe. The queue is drained by a single background
    thread that owns the real file + stream handlers.
    """
    # Determine log directory and default file path
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log_file = log_dir / "application.log"

    # Get log level from environment
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Get log file path
    log_file_path = Path(os.environ.get("LOG_FILE_PATH", str(default_log_file)))

    # Secure path check: must be inside logs/ directory
    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(f"LOG_FILE_PATH '{log_file_path}' is outside the trusted log directory '{log_dir_resolved}'")

    # Ensure parent directories exist
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Get max log file size (default: 10MB)
    try:
        max_mb = int(os.environ.get("LOG_MAX_SIZE", 10))  # 10MB default
        max_bytes = max_mb * 1024 * 1024
    except (TypeError, ValueError):
        max_bytes = 10 * 1024 * 1024  # fallback to 10MB on error

    # Get backup count (default: 5)
    try:
        backup_count = int(os.environ.get("LOG_BACKUP_COUNT", 5))
    except ValueError:
        backup_count = 5

    # Configure format
    log_format = format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"

    # Create the REAL handlers (owned solely by the QueueListener thread).
    file_handler = RotatingFileHandler(resolved_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    console_handler = logging.StreamHandler()

    # Set format for both handlers (truncating so giant records never block).
    formatter = _TruncatingFormatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add filter to suppress "Detected file change" messages
    file_handler.addFilter(IgnoreLogChangeDetectedFilter())
    console_handler.addFilter(IgnoreLogChangeDetectedFilter())

    # --- Thread-safe non-blocking dispatch ----------------------------------
    # ``StreamHandler`` writes synchronously on the calling thread. Under Docker,
    # stdout/stderr are pipes with a finite kernel buffer (~64 KB). A worker
    # thread (adalflow's ``log.info(output)`` inside ``asyncio.to_thread``)
    # that emits a record larger than the free buffer, or that emits while the
    # log consumer is slow, performs a blocking write and raises
    # ``BlockingIOError: [Errno 11] write could not complete without blocking``.
    #
    # ``QueueHandler`` instead just ``put_nowait``s the record onto an
    # unbounded ``queue.SimpleQueue`` (never blocks, never raises BlockingIOError)
    # and a single ``QueueListener`` thread does all the real file/console I/O.
    # That isolates blocking writes to one dedicated thread regardless of how
    # many worker threads emit logs.
    _log_queue: queue.Queue = queue.SimpleQueue()  # type: ignore[assignment]
    queue_handler = QueueHandler(_log_queue)
    queue_handler.setLevel(log_level)
    listener = QueueListener(
        _log_queue, file_handler, console_handler, respect_handler_level=True
    )
    listener.start()

    # Apply logging configuration: ONLY the queue handler is attached to the
    # root logger; the file/console handlers are driven by the listener.
    logging.basicConfig(level=log_level, handlers=[queue_handler], force=True)

    # --- Route third-party loggers through the root QueueHandler -----------
    # adalflow (``adalflow.utils.logger.get_logger``) and litellm
    # (``litellm._logging`` / ``litellm.litellm_logging_utils``) attach their
    # OWN ``StreamHandler(sys.stdout)`` to named loggers with
    # ``propagate=False``. Those direct handlers write synchronously on the
    # calling thread -- from a worker thread (``asyncio.to_thread`` used by
    # adalflow / RLM / cognee) to a pipe-backed stdout that raises
    # ``BlockingIOError: [Errno 11] write could not complete without blocking``
    # once the pipe buffer fills. They also emit noisy INFO records (adalflow's
    # ``Prompt has variables: ['input_str']`` on every Generator construct;
    # litellm's verbose request logging).
    #
    # Fix: clear their direct handlers, set propagate=True (so records flow to
    # the root QueueHandler -> the listener thread's truncating formatter),
    # and raise their level to WARNING so the INFO-level spam never even
    # reaches the queue. ``setup_logging`` runs with ``force=True`` above, so
    # re-running it (e.g. via tests) re-applies this cleanly.
    for _vendor_name in (
        "adalflow", "adalflow.core", "adalflow.utils", "adalflow.components",
        "litellm", "litellm.litellm_logging_utils", "litellm.utils",
        "httpx", "httpcore", "openai._base_client",
    ):
        try:
            _vlog = logging.getLogger(_vendor_name)
            _vlog.handlers = []  # drop their direct StreamHandler/FileHandler
            _vlog.propagate = True  # -> root -> QueueHandler
            _vlog.setLevel(logging.WARNING)
        except Exception:
            pass

    # Log configuration info
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Logging configured: level={log_level_str}, "
        f"file={resolved_path}, max_size={max_bytes} bytes, "
        f"backup_count={backup_count}, queue_listener_started=True"
    )
