"""Unit tests for the Confluence integration connector (api.integrations.confluence).

Covers:
- ``test()`` connectivity — direct REST (v2 success, v2→v1 fallback, full
  failure) and MCP mode dispatch.
- ``list_spaces()`` — pagination via ``_links.next``, ``configured_space``
  filtering, empty/not-configured.
- ``pull()`` — single page, recursive tree (depth-bounded), attachment
  conversion via markitdown (monkeypatched), MCP mode, not-configured error.
- ``_auth_headers`` — Basic (Cloud) vs Bearer (Server/DC).
- ``_base()`` — trailing slash + ``/wiki`` stripping.
- ``_get`` / ``_get_bytes`` — HTTP error + non-JSON handling.
- ``_pages_to_markdown`` — heading hierarchy + page_id comments.

All HTTP calls are mocked via monkeypatching ``requests.get`` or the
connector's own ``_get`` / ``_get_bytes`` methods. ``convert_to_markdown``
is monkeypatched so no real markitdown dependency is required.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import base64
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ===========================================================================
# Helpers
# ===========================================================================
class _FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code=200, text="", headers=None, json_data=None, content=b""):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self._json_data = json_data
        self.content = content

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


def _direct_config(**kw) -> Dict[str, Any]:
    """A minimal direct-mode Confluence config."""
    base = {"mode": "direct", "base_url": "https://confluence.example.com", "token": "tok123"}
    base.update(kw)
    return base


def _connector(config=None, **overrides):
    from api.integrations.confluence import ConfluenceConnector

    cfg = config or _direct_config()
    cfg.update(overrides)
    return ConfluenceConnector(cfg)


# ===========================================================================
# is_configured / _is_mcp_mode / _base / _auth_headers
# ===========================================================================
class TestConfigHelpers:
    def test_is_configured_direct_true(self):
        c = _connector()
        assert c.is_configured() is True

    def test_is_configured_direct_missing_token(self):
        c = _connector(base_url="https://x", token="")
        assert c.is_configured() is False

    def test_is_configured_direct_missing_base_url(self):
        c = _connector(base_url="", token="tok")
        assert c.is_configured() is False

    def test_is_configured_mcp_mode(self, monkeypatch):
        c = _connector(mode="mcp")
        fake_client = MagicMock()
        fake_client.is_configured = lambda: True
        monkeypatch.setattr("api.utils.LocalMcpClient", lambda *a, **kw: fake_client)
        assert c.is_configured() is True

    def test_is_configured_mcp_mode_failure(self, monkeypatch):
        c = _connector(mode="mcp")

        def _boom(*a, **kw):
            raise ImportError("no mcp")

        monkeypatch.setattr("api.utils.LocalMcpClient", _boom)
        assert c.is_configured() is False

    def test_is_mcp_mode_direct(self):
        c = _connector(mode="direct")
        assert c._is_mcp_mode() is False

    def test_is_mcp_mode_mcp(self):
        c = _connector(mode="MCP")
        assert c._is_mcp_mode() is True

    def test_is_mcp_mode_default(self):
        c = _connector()
        c.config = {}  # no mode key
        assert c._is_mcp_mode() is False  # defaults to direct

    def test_base_strips_trailing_slash(self):
        c = _connector(base_url="https://x.example.com/")
        assert c._base() == "https://x.example.com"

    def test_base_strips_trailing_wiki(self):
        c = _connector(base_url="https://x.example.com/wiki")
        assert c._base() == "https://x.example.com"

    def test_base_strips_trailing_wiki_with_slash(self):
        c = _connector(base_url="https://x.example.com/wiki/")
        assert c._base() == "https://x.example.com"

    def test_base_empty(self):
        c = _connector(base_url="")
        assert c._base() == ""

    def test_auth_headers_bearer_no_username(self):
        c = _connector(token="my-pat")
        headers = c._auth_headers()
        assert headers["Authorization"] == "Bearer my-pat"
        assert headers["Accept"] == "application/json"

    def test_auth_headers_basic_with_username(self):
        c = _connector(username="user@example.com", token="api-token")
        headers = c._auth_headers()
        assert headers["Authorization"].startswith("Basic ")
        decoded = base64.b64decode(headers["Authorization"][6:]).decode()
        assert decoded == "user@example.com:api-token"

    def test_auth_headers_empty_token(self):
        c = _connector(token="")
        headers = c._auth_headers()
        assert headers["Authorization"] == "Bearer "

    def test_get_config_reads_settings_store(self, isolated_db):
        from api.integrations.confluence import ConfluenceConnector
        from api.config.settings import set_setting

        set_setting("confluence.base_url", "https://stored.example.com")
        set_setting("confluence.token", "stored-tok")
        cfg = ConfluenceConnector.get_config()
        assert cfg["base_url"] == "https://stored.example.com"
        assert cfg["token"] == "stored-tok"
        assert cfg["mode"] == "direct"


# ===========================================================================
# _get / _get_bytes — HTTP layer
# ===========================================================================
class TestHttpLayer:
    def test_get_success(self, monkeypatch):
        c = _connector()
        captured: Dict[str, Any] = {}

        def _get(url, headers=None, params=None, timeout=None, verify=True):
            captured.update(url=url, headers=headers, params=params, timeout=timeout, verify=verify)
            return _FakeResp(200, text='{"results":[]}', headers={"content-type": "application/json"}, json_data={"results": []})

        monkeypatch.setattr("requests.get", _get)
        data = c._get("/wiki/api/v2/spaces", params={"limit": 1})
        assert data == {"results": []}
        assert captured["url"] == "https://confluence.example.com/wiki/api/v2/spaces"
        assert captured["params"] == {"limit": 1}

    def test_get_http_error_raises(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(403, text="Forbidden", headers={})
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        with pytest.raises(ValueError, match="HTTP 403"):
            c._get("/wiki/api/v2/spaces")

    def test_get_empty_response_raises(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(200, text="", headers={"content-type": "text/html"})
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        with pytest.raises(ValueError, match="empty response"):
            c._get("/wiki/api/v2/spaces")

    def test_get_non_json_raises(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(200, text="<html>not json</html>", headers={"content-type": "text/html"})
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        with pytest.raises(ValueError, match="non-JSON"):
            c._get("/wiki/api/v2/spaces")

    def test_get_bytes_success(self, monkeypatch):
        c = _connector()
        captured: Dict[str, Any] = {}

        def _get(url, headers=None, params=None, timeout=None, verify=True):
            captured.update(url=url, headers=headers)
            return _FakeResp(200, text="", headers={}, content=b"binary-data")

        monkeypatch.setattr("requests.get", _get)
        data = c._get_bytes("https://x.example.com/attachment")
        assert data == b"binary-data"
        assert "Authorization" in captured["headers"]

    def test_get_bytes_http_error_raises(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(404, text="Not found", headers={})
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        with pytest.raises(ValueError, match="HTTP 404"):
            c._get_bytes("https://x.example.com/attachment")


# ===========================================================================
# test() — direct mode
# ===========================================================================
class TestConfluenceTest:
    def test_not_configured(self):
        c = _connector(base_url="", token="")
        result = c.test()
        assert result["success"] is False
        assert "not configured" in result["message"]

    def test_direct_v2_success(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(
            200,
            text='{"results":[{"id":"1"}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"id": "1"}]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        result = c.test()
        assert result["success"] is True
        assert "1 space(s)" in result["message"]

    def test_direct_v2_fails_v1_succeeds(self, monkeypatch):
        c = _connector()
        # First call (v2) returns 403, second call (v1) returns 200.
        responses = [
            _FakeResp(403, text="v2 forbidden", headers={}),
            _FakeResp(200, text='{"results":[{"key":"ENG"}]}', headers={"content-type": "application/json"}, json_data={"results": [{"key": "ENG"}]}),
        ]

        def _get(url, **kw):
            return responses.pop(0)

        monkeypatch.setattr("requests.get", _get)
        result = c.test()
        assert result["success"] is True
        assert "1 space(s)" in result["message"]

    def test_direct_v2_and_v1_both_fail(self, monkeypatch):
        c = _connector()
        responses = [
            _FakeResp(500, text="v2 error", headers={}),
            _FakeResp(500, text="v1 error", headers={}),
        ]

        def _get(url, **kw):
            return responses.pop(0)

        monkeypatch.setattr("requests.get", _get)
        result = c.test()
        assert result["success"] is False
        assert "connection failed" in result["message"]

    def test_direct_spaces_key_fallback(self, monkeypatch):
        c = _connector()
        # Some responses use "spaces" instead of "results".
        resp = _FakeResp(
            200,
            text='{"spaces":[{"id":"1"}]}',
            headers={"content-type": "application/json"},
            json_data={"spaces": [{"id": "1"}, {"id": "2"}]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        result = c.test()
        assert result["success"] is True
        assert "2 space(s)" in result["message"]

    def test_direct_results_not_list(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(
            200,
            text='{"results":"not-a-list"}',
            headers={"content-type": "application/json"},
            json_data={"results": "not-a-list"},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        result = c.test()
        assert result["success"] is True
        # Non-list results → n=1
        assert "1 space(s)" in result["message"]

    # --- MCP mode test ---
    def test_mcp_mode_test_success(self, monkeypatch):
        c = _connector(mode="mcp")
        fake_client = MagicMock()
        fake_client.is_configured = lambda: True
        fake_client.test_connections = lambda: {"success": True, "message": "all good"}
        # is_configured() instantiates LocalMcpClient; _mcp_test() calls get_local_mcp_client.
        monkeypatch.setattr("api.utils.LocalMcpClient", lambda *a, **kw: fake_client)
        monkeypatch.setattr("api.utils.get_local_mcp_client", lambda: fake_client)
        result = c.test()
        assert result["success"] is True
        assert "all good" in result["message"]

    def test_mcp_mode_test_failure(self, monkeypatch):
        c = _connector(mode="mcp")

        class _BadClient:
            def is_configured(self):
                return True

            def test_connections(self):
                raise RuntimeError("mcp down")

        monkeypatch.setattr("api.utils.LocalMcpClient", lambda *a, **kw: _BadClient())
        monkeypatch.setattr("api.utils.get_local_mcp_client", lambda: _BadClient())
        result = c.test()
        assert result["success"] is False
        assert "mcp down" in result["message"]


# ===========================================================================
# list_spaces() — direct mode
# ===========================================================================
class TestListSpaces:
    def test_not_configured(self):
        c = _connector(base_url="", token="")
        assert c.list_spaces() == []

    def test_single_page(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(
            200,
            text='{"results":[{"id":"1","key":"ENG","name":"Engineering"}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"id": "1", "key": "ENG", "name": "Engineering"}]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        spaces = c.list_spaces()
        assert len(spaces) == 1
        assert spaces[0]["id"] == "1"
        assert spaces[0]["title"] == "Engineering"
        assert spaces[0]["key"] == "ENG"
        assert spaces[0]["type"] == "space"

    def test_pagination_follows_next(self, monkeypatch):
        c = _connector()
        page1 = _FakeResp(
            200,
            text='{"results":[{"id":"1","key":"A"}],"_links":{"next":"/wiki/api/v2/spaces?cursor=2"}}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"id": "1", "key": "A"}], "_links": {"next": "/wiki/api/v2/spaces?cursor=2"}},
        )
        page2 = _FakeResp(
            200,
            text='{"results":[{"id":"2","key":"B"}],"_links":{}}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"id": "2", "key": "B"}], "_links": {}},
        )
        responses = [page1, page2]
        monkeypatch.setattr("requests.get", lambda *a, **kw: responses.pop(0))
        spaces = c.list_spaces()
        assert len(spaces) == 2
        assert [s["key"] for s in spaces] == ["A", "B"]

    def test_configured_space_filter(self, monkeypatch):
        c = _connector(space="ENG")
        resp = _FakeResp(
            200,
            text='{"results":[{"id":"1","key":"ENG","name":"Eng"},{"id":"2","key":"OPS","name":"Ops"}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [
                {"id": "1", "key": "ENG", "name": "Eng"},
                {"id": "2", "key": "OPS", "name": "Ops"},
            ]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        spaces = c.list_spaces()
        assert len(spaces) == 1
        assert spaces[0]["key"] == "ENG"

    def test_configured_space_filter_by_id(self, monkeypatch):
        c = _connector(space="123")
        resp = _FakeResp(
            200,
            text='{"results":[{"id":"123","key":"ENG"},{"id":"456","key":"OPS"}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [
                {"id": "123", "key": "ENG"},
                {"id": "456", "key": "OPS"},
            ]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        spaces = c.list_spaces()
        assert len(spaces) == 1
        assert spaces[0]["id"] == "123"

    def test_configured_space_no_match_returns_all(self, monkeypatch):
        c = _connector(space="NONEXISTENT")
        resp = _FakeResp(
            200,
            text='{"results":[{"id":"1","key":"ENG"}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"id": "1", "key": "ENG"}]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        spaces = c.list_spaces()
        # When filter matches nothing, returns the unfiltered list.
        assert len(spaces) == 1

    def test_http_failure_returns_partial(self, monkeypatch):
        c = _connector()
        def _boom(*a, **kw):
            raise ConnectionError("down")
        monkeypatch.setattr("requests.get", _boom)
        assert c.list_spaces() == []

    def test_non_dict_results_skipped(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(
            200,
            text='{"results":["str", {"id":"1","key":"A"}, 42]}',
            headers={"content-type": "application/json"},
            json_data={"results": ["str", {"id": "1", "key": "A"}, 42]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        spaces = c.list_spaces()
        assert len(spaces) == 1
        assert spaces[0]["key"] == "A"

    # --- MCP mode list_spaces ---
    def test_mcp_mode_list_spaces(self, monkeypatch):
        c = _connector(mode="mcp")
        # is_configured() instantiates LocalMcpClient.
        fake_client = MagicMock()
        fake_client.is_configured = lambda: True
        monkeypatch.setattr("api.utils.LocalMcpClient", lambda *a, **kw: fake_client)
        tools = [{"id": "confluence:t1", "server": "confluence"}, {"id": "other:t2", "server": "other"}]
        monkeypatch.setattr("api.utils.list_all_mcp_tools", lambda: tools)
        spaces = c.list_spaces()
        assert len(spaces) == 1
        assert spaces[0]["server"] == "confluence"

    def test_mcp_mode_list_spaces_fallback_all(self, monkeypatch):
        c = _connector(mode="mcp")
        fake_client = MagicMock()
        fake_client.is_configured = lambda: True
        monkeypatch.setattr("api.utils.LocalMcpClient", lambda *a, **kw: fake_client)
        tools = [{"id": "x:t1", "server": "x"}, {"id": "y:t2", "server": "y"}]
        monkeypatch.setattr("api.utils.list_all_mcp_tools", lambda: tools)
        spaces = c.list_spaces()
        # No server matches "confluence" → returns all tools.
        assert len(spaces) == 2

    def test_mcp_mode_list_spaces_failure(self, monkeypatch):
        c = _connector(mode="mcp")
        fake_client = MagicMock()
        fake_client.is_configured = lambda: True
        monkeypatch.setattr("api.utils.LocalMcpClient", lambda *a, **kw: fake_client)

        def _boom():
            raise RuntimeError("mcp down")

        monkeypatch.setattr("api.utils.list_all_mcp_tools", _boom)
        assert c.list_spaces() == []


# ===========================================================================
# _fetch_page / _fetch_children / _fetch_attachments / _convert_attachments
# ===========================================================================
class TestPageFetching:
    def test_fetch_page(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(
            200,
            text='{"id":"100","title":"My Page","body":{"value":"<p>hello</p>"},"spaceId":"50"}',
            headers={"content-type": "application/json"},
            json_data={"id": "100", "title": "My Page", "body": {"value": "<p>hello</p>"}, "spaceId": "50"},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        page = c._fetch_page("100")
        assert page["id"] == "100"
        assert page["title"] == "My Page"
        assert page["html"] == "<p>hello</p>"
        assert page["space_id"] == "50"

    def test_fetch_page_missing_fields(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(
            200,
            text='{"id":"100"}',
            headers={"content-type": "application/json"},
            json_data={"id": "100"},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        page = c._fetch_page("100")
        assert page["id"] == "100"
        assert page["title"] == "100"
        assert page["html"] == ""

    def test_fetch_children(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(
            200,
            text='{"results":[{"id":"101"},{"id":"102"}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"id": "101"}, {"id": "102"}]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        children = c._fetch_children("100")
        assert len(children) == 2

    def test_fetch_children_failure(self, monkeypatch):
        c = _connector()
        def _boom(*a, **kw):
            raise ConnectionError("down")
        monkeypatch.setattr("requests.get", _boom)
        assert c._fetch_children("100") == []

    def test_fetch_attachments(self, monkeypatch):
        c = _connector()
        resp = _FakeResp(
            200,
            text='{"results":[{"title":"doc.pdf","_links":{"download":"/dl/1"}}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"title": "doc.pdf", "_links": {"download": "/dl/1"}}]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: resp)
        attachments = c._fetch_attachments("100")
        assert len(attachments) == 1
        assert attachments[0]["title"] == "doc.pdf"

    def test_fetch_attachments_failure(self, monkeypatch):
        c = _connector()
        def _boom(*a, **kw):
            raise ConnectionError("down")
        monkeypatch.setattr("requests.get", _boom)
        assert c._fetch_attachments("100") == []

    def test_convert_attachments_success(self, monkeypatch):
        c = _connector()
        att_resp = _FakeResp(
            200,
            text='{"results":[{"title":"doc.pdf","_links":{"download":"/dl/1"}}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"title": "doc.pdf", "_links": {"download": "/dl/1"}}]},
        )
        dl_resp = _FakeResp(200, text="", headers={}, content=b"%PDF-1.4")
        responses = [att_resp, dl_resp]
        monkeypatch.setattr("requests.get", lambda *a, **kw: responses.pop(0))
        monkeypatch.setattr("api.integrations.confluence.convert_to_markdown", lambda raw, filename: f"md of {filename}")
        converted = c._convert_attachments("100")
        assert len(converted) == 1
        assert converted[0]["filename"] == "doc.pdf"
        assert converted[0]["markdown"] == "md of doc.pdf"

    def test_convert_attachments_absolute_download_url(self, monkeypatch):
        c = _connector()
        att_resp = _FakeResp(
            200,
            text='{"results":[{"title":"doc.pdf","_links":{"download":"https://cdn.example.com/dl/1"}}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"title": "doc.pdf", "_links": {"download": "https://cdn.example.com/dl/1"}}]},
        )
        dl_resp = _FakeResp(200, text="", headers={}, content=b"data")
        responses = [att_resp, dl_resp]
        monkeypatch.setattr("requests.get", lambda *a, **kw: responses.pop(0))
        monkeypatch.setattr("api.integrations.confluence.convert_to_markdown", lambda raw, filename: "md")
        converted = c._convert_attachments("100")
        assert len(converted) == 1

    def test_convert_attachments_no_download_link(self, monkeypatch):
        c = _connector()
        att_resp = _FakeResp(
            200,
            text='{"results":[{"title":"no-link.pdf","_links":{}}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"title": "no-link.pdf", "_links": {}}]},
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: att_resp)
        converted = c._convert_attachments("100")
        assert converted == []

    def test_convert_attachments_download_failure(self, monkeypatch):
        c = _connector()
        att_resp = _FakeResp(
            200,
            text='{"results":[{"title":"doc.pdf","_links":{"download":"/dl/1"}}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"title": "doc.pdf", "_links": {"download": "/dl/1"}}]},
        )
        dl_resp = _FakeResp(404, text="not found", headers={})
        responses = [att_resp, dl_resp]
        monkeypatch.setattr("requests.get", lambda *a, **kw: responses.pop(0))
        converted = c._convert_attachments("100")
        assert len(converted) == 1
        assert "failed to convert" in converted[0]["markdown"]

    def test_convert_attachments_links_via_links_key(self, monkeypatch):
        # Source uses ``_links`` or ``links`` (not ``attachments``).
        c = _connector()
        att_resp = _FakeResp(
            200,
            text='{"results":[{"title":"doc.pdf","links":{"download":"/dl/1"}}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"title": "doc.pdf", "links": {"download": "/dl/1"}}]},
        )
        dl_resp = _FakeResp(200, text="", headers={}, content=b"data")
        responses = [att_resp, dl_resp]
        monkeypatch.setattr("requests.get", lambda *a, **kw: responses.pop(0))
        monkeypatch.setattr("api.integrations.confluence.convert_to_markdown", lambda raw, filename: "md")
        converted = c._convert_attachments("100")
        assert len(converted) == 1
        assert converted[0]["filename"] == "doc.pdf"


# ===========================================================================
# _pages_to_markdown
# ===========================================================================
class TestPagesToMarkdown:
    def test_single_page(self):
        from api.integrations.confluence import ConfluenceConnector

        pages = [{"id": "1", "title": "Root", "html": "<p>content</p>"}]
        md = ConfluenceConnector._pages_to_markdown(pages)
        assert md.startswith("# Root")
        assert "<p>content</p>" in md
        assert "<!-- page_id=1 -->" in md

    def test_multiple_pages_subheadings(self):
        from api.integrations.confluence import ConfluenceConnector

        pages = [
            {"id": "1", "title": "Root", "html": "<p>root</p>"},
            {"id": "2", "title": "Child", "html": "<p>child</p>"},
        ]
        md = ConfluenceConnector._pages_to_markdown(pages)
        assert "# Root" in md
        assert "## Child" in md
        assert "<!-- page_id=1 -->" in md
        assert "<!-- page_id=2 -->" in md

    def test_page_with_empty_html(self):
        from api.integrations.confluence import ConfluenceConnector

        pages = [{"id": "1", "title": "Empty", "html": ""}]
        md = ConfluenceConnector._pages_to_markdown(pages)
        assert "# Empty" in md
        assert "<!-- page_id=1 -->" in md


# ===========================================================================
# pull() — direct mode
# ===========================================================================
class TestConfluencePull:
    def test_not_configured_raises(self):
        c = _connector(base_url="", token="")
        with pytest.raises(ValueError, match="not configured"):
            c.pull("123")

    def test_single_page(self, monkeypatch):
        c = _connector()
        page_resp = _FakeResp(
            200,
            text='{"id":"100","title":"My Page","body":{"value":"<p>hi</p>"},"spaceId":"50"}',
            headers={"content-type": "application/json"},
            json_data={"id": "100", "title": "My Page", "body": {"value": "<p>hi</p>"}, "spaceId": "50"},
        )
        att_resp = _FakeResp(
            200,
            text='{"results":[]}',
            headers={"content-type": "application/json"},
            json_data={"results": []},
        )
        responses = [page_resp, att_resp]
        monkeypatch.setattr("requests.get", lambda *a, **kw: responses.pop(0))
        result = c.pull("100")
        assert result["title"] == "My Page"
        assert result["page_id"] == "100"
        assert result["space_id"] == "50"
        assert result["page_count"] == 1
        assert result["source"] == "confluence"
        assert "<p>hi</p>" in result["markdown"]
        assert result["attachments"] == []

    def test_recursive_tree(self, monkeypatch):
        c = _connector()
        # Root page → child1, child2 → grandchild under child1
        root = _FakeResp(
            200,
            text='{"id":"1","title":"Root","body":{"value":"<p>root</p>"}}',
            headers={"content-type": "application/json"},
            json_data={"id": "1", "title": "Root", "body": {"value": "<p>root</p>"}},
        )
        root_children = _FakeResp(
            200,
            text='{"results":[{"id":"2","title":"Child1"},{"id":"3","title":"Child2"}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"id": "2", "title": "Child1"}, {"id": "3", "title": "Child2"}]},
        )
        child1 = _FakeResp(
            200,
            text='{"id":"2","title":"Child1","body":{"value":"<p>c1</p>"}}',
            headers={"content-type": "application/json"},
            json_data={"id": "2", "title": "Child1", "body": {"value": "<p>c1</p>"}},
        )
        child1_children = _FakeResp(
            200,
            text='{"results":[{"id":"4","title":"Grandchild"}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"id": "4", "title": "Grandchild"}]},
        )
        grandchild = _FakeResp(
            200,
            text='{"id":"4","title":"Grandchild","body":{"value":"<p>gc</p>"}}',
            headers={"content-type": "application/json"},
            json_data={"id": "4", "title": "Grandchild", "body": {"value": "<p>gc</p>"}},
        )
        grandchild_children = _FakeResp(
            200,
            text='{"results":[]}',
            headers={"content-type": "application/json"},
            json_data={"results": []},
        )
        child2 = _FakeResp(
            200,
            text='{"id":"3","title":"Child2","body":{"value":"<p>c2</p>"}}',
            headers={"content-type": "application/json"},
            json_data={"id": "3", "title": "Child2", "body": {"value": "<p>c2</p>"}},
        )
        child2_children = _FakeResp(
            200,
            text='{"results":[]}',
            headers={"content-type": "application/json"},
            json_data={"results": []},
        )
        responses = [root, root_children, child1, child1_children, grandchild, grandchild_children, child2, child2_children]
        monkeypatch.setattr("requests.get", lambda *a, **kw: responses.pop(0))
        result = c.pull("1", opts={"recursive": True})
        assert result["page_count"] == 4
        assert "# Root" in result["markdown"]
        assert "## Child1" in result["markdown"]
        assert "## Grandchild" in result["markdown"]
        assert "## Child2" in result["markdown"]
        # Recursive pull skips attachments.
        assert result["attachments"] == []

    def test_pull_with_attachments(self, monkeypatch):
        c = _connector()
        page = _FakeResp(
            200,
            text='{"id":"1","title":"Root","body":{"value":"<p>root</p>"}}',
            headers={"content-type": "application/json"},
            json_data={"id": "1", "title": "Root", "body": {"value": "<p>root</p>"}},
        )
        atts = _FakeResp(
            200,
            text='{"results":[{"title":"doc.pdf","_links":{"download":"/dl/1"}}]}',
            headers={"content-type": "application/json"},
            json_data={"results": [{"title": "doc.pdf", "_links": {"download": "/dl/1"}}]},
        )
        dl = _FakeResp(200, text="", headers={}, content=b"%PDF")
        responses = [page, atts, dl]
        monkeypatch.setattr("requests.get", lambda *a, **kw: responses.pop(0))
        monkeypatch.setattr("api.integrations.confluence.convert_to_markdown", lambda raw, filename: f"md-{filename}")
        result = c.pull("1")
        assert len(result["attachments"]) == 1
        assert result["attachments"][0]["markdown"] == "md-doc.pdf"


# ===========================================================================
# pull() — MCP mode
# ===========================================================================
class TestConfluenceMcpPull:
    def _setup_mcp(self, monkeypatch):
        """Patch LocalMcpClient so is_configured() returns True in MCP mode."""
        fake_client = MagicMock()
        fake_client.is_configured = lambda: True
        monkeypatch.setattr("api.utils.LocalMcpClient", lambda *a, **kw: fake_client)

    def test_mcp_pull_success(self, monkeypatch):
        c = _connector(mode="mcp")
        self._setup_mcp(monkeypatch)
        pulled = {"title": "MCP Page", "markdown": "# MCP content", "attachments": [{"filename": "a.pdf", "markdown": "md"}]}
        monkeypatch.setattr("api.utils.invoke_mcp_tool", lambda sid, opts=None: pulled)
        result = c.pull("page123")
        assert result["title"] == "MCP Page"
        assert result["markdown"] == "# MCP content"
        assert result["page_id"] == "page123"
        assert result["source"] == "confluence_mcp"
        assert len(result["attachments"]) == 1

    def test_mcp_pull_adds_server_prefix(self, monkeypatch):
        c = _connector(mode="mcp", mcp_server="myconf")
        self._setup_mcp(monkeypatch)
        captured: Dict[str, Any] = {}

        def _invoke(sid, opts=None):
            captured["sid"] = sid
            return {"markdown": "content"}

        monkeypatch.setattr("api.utils.invoke_mcp_tool", _invoke)
        result = c.pull("page123")  # no colon → server prefix added
        assert captured["sid"] == "myconf:page123"
        assert result["markdown"] == "content"

    def test_mcp_pull_keeps_colon_source_id(self, monkeypatch):
        c = _connector(mode="mcp", mcp_server="myconf")
        self._setup_mcp(monkeypatch)
        captured: Dict[str, Any] = {}

        def _invoke(sid, opts=None):
            captured["sid"] = sid
            return {"markdown": "content"}

        monkeypatch.setattr("api.utils.invoke_mcp_tool", _invoke)
        c.pull("confluence:page123")
        assert captured["sid"] == "confluence:page123"

    def test_mcp_pull_missing_fields(self, monkeypatch):
        c = _connector(mode="mcp")
        self._setup_mcp(monkeypatch)
        monkeypatch.setattr("api.utils.invoke_mcp_tool", lambda sid, opts=None: {})
        result = c.pull("page123")
        assert result["title"] == "page123"
        assert result["markdown"] == "{}"
        assert result["attachments"] == []

    def test_mcp_pull_string_result(self, monkeypatch):
        c = _connector(mode="mcp")
        self._setup_mcp(monkeypatch)
        monkeypatch.setattr("api.utils.invoke_mcp_tool", lambda sid, opts=None: "raw string")
        result = c.pull("page123")
        assert result["markdown"] == "raw string"
