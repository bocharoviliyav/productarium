"""Unit tests for api.utils.logging filters (uvicorn.access noise mute)."""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.utils.logging import DocgenPollNoiseFilter  # noqa: E402


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    """Build a record shaped like uvicorn's access log line."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="p",
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:5000", method, path, "1.1", status),
        exc_info=None,
    )


class TestDocgenPollNoiseFilter:
    def test_mutes_200_status_poll(self):
        f = DocgenPollNoiseFilter()
        rec = _access_record(
            "GET",
            "/api/products/p1/codebases/c1/generate/status?job_id=abc",
            200,
        )
        assert f.filter(rec) is False

    def test_mutes_200_active_poll(self):
        f = DocgenPollNoiseFilter()
        rec = _access_record("GET", "/api/products/p1/docgen/active", 200)
        assert f.filter(rec) is False

    def test_keeps_non_200_status_poll(self):
        f = DocgenPollNoiseFilter()
        rec = _access_record(
            "GET",
            "/api/products/p1/codebases/c1/generate/status?job_id=abc",
            404,
        )
        assert f.filter(rec) is True

    def test_keeps_unrelated_200(self):
        f = DocgenPollNoiseFilter()
        rec = _access_record("GET", "/api/products", 200)
        assert f.filter(rec) is True

    def test_setup_logging_installs_filter_idempotently(self):
        from api.utils.logging import setup_logging

        setup_logging()
        setup_logging()
        uv_access = logging.getLogger("uvicorn.access")
        installed = [f for f in uv_access.filters if isinstance(f, DocgenPollNoiseFilter)]
        assert len(installed) == 1
