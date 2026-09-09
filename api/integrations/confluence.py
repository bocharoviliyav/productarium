"""Confluence connector (plan section G).

Talks to the Confluence Cloud REST API v2 (``<base_url>/wiki/api/v2``) to list
spaces, fetch page bodies (export/storage format), walk child pages, and
download attachments. Binary attachments (docx/pdf/pptx/xlsx/...) are
converted to markdown via :mod:`api.formats.markitdown`.

Auth: Confluence Cloud uses HTTP Basic with ``email:api_token``; Confluence
Server/Data Center uses a Bearer personal-access token. The settings store
returns ``{base_url, token, space, username}`` — when ``username`` is present
we use Basic auth, otherwise Bearer. This keeps the connector usable for both
deployments while reading from a single settings group.

No live credentials are required to import this module — all network calls
live inside ``test``/``list_spaces``/``pull`` and the HTTP layer is a single
:py:meth:`_get` method that tests can monkeypatch. Built against the public
Confluence Cloud v2 spec:
    GET /wiki/api/v2/spaces
    GET /wiki/api/v2/pages/{id}?body-format=export
    GET /wiki/api/v2/pages/{id}/children
    GET /wiki/api/v2/pages/{id}/attachments

(The former "corporate MCP mode" was removed with the legacy hand-written
MCP client; external MCP servers are now managed by the Wave C MCP platform
— see ``api/mcp/`` and ``api/routers/mcp_admin.py``.)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from api.integrations.base import IntegrationConnector
from api.formats.markitdown import convert_to_markdown

logger = logging.getLogger(__name__)

# Max recursion depth when pulling a page tree (opts={"recursive": True}).
_MAX_TREE_DEPTH = 3
# Max child pages fetched per page (keeps pulls bounded).
_MAX_CHILDREN = 50

# Page id inside a friendly URL: .../pages/{id} or .../pages/{id}/Title.
_URL_PAGES_RE = re.compile(r"/pages/(\d+)")
# Page id as an explicit query parameter: ...?pageId={id} (Server/DC viewpage).
_URL_PAGE_ID_QS_RE = re.compile(r"[?&]pageId=(\d+)")


class ConfluenceConnector(IntegrationConnector):
    name = "confluence"
    display_name = "Confluence"
    description = "Pull Confluence spaces/pages via the Confluence REST v2/v1 API."
    requires_credentials = True

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        from api.config.settings import get_confluence_creds

        return get_confluence_creds()

    def is_configured(self) -> bool:
        return bool(self.config.get("base_url") and self.config.get("token"))

    # ---- HTTP layer (single method so tests can monkeypatch it) ----------
    def _auth_headers(self) -> Dict[str, str]:
        token = self.config.get("token") or ""
        username = self.config.get("username")
        if username:
            # Confluence Cloud: Basic auth with email:api_token.
            import base64

            raw = f"{username}:{token}".encode("utf-8")
            return {"Authorization": f"Basic {base64.b64encode(raw).decode('ascii')}"}
        # Confluence Server/DC: Bearer PAT.
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def _base(self) -> str:
        b = (self.config.get("base_url") or "").strip().rstrip("/")
        # Strip trailing /wiki if user entered it in base_url to avoid /wiki/wiki URL duplication
        if b.endswith("/wiki"):
            b = b[:-5].rstrip("/")
        return b

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET a Confluence API path. Returns parsed JSON (dict).

        Raises a clear ValueError on HTTP errors or non-JSON responses.
        """
        import json
        import requests
        from api.config.ssl import requests_verify

        from api.config.timeout import resolve_integration_http_timeout
        url = f"{self._base()}{path}"
        resp = requests.get(
            url, headers=self._auth_headers(), params=params, timeout=resolve_integration_http_timeout(),
            verify=requests_verify(),
        )
        if resp.status_code >= 400:
            # The URL (which may embed credentials) and the response body stay
            # in the server log only — ValueError messages surface verbatim in
            # pull/test responses to regular users (review #5).
            logger.warning(
                "Confluence API %s -> HTTP %s: %s", url, resp.status_code, resp.text[:200]
            )
            raise ValueError(f"Confluence API request failed (HTTP {resp.status_code})")

        text = resp.text or ""
        if not text.strip():
            logger.warning("Confluence API %s returned an empty response.", url)
            raise ValueError("Confluence API returned an empty response.")

        try:
            return resp.json()
        except Exception as err:
            ct = resp.headers.get("content-type", "unknown")
            preview = text[:150].replace("\n", " ")
            logger.warning(
                "Confluence API %s returned non-JSON content (type: %s): %r",
                url, ct, preview,
            )
            raise ValueError(
                "Confluence API returned non-JSON content; "
                "check base_url path and credentials."
            ) from err

    def _get_bytes(self, url: str) -> bytes:
        """Download raw bytes (attachment download link)."""
        import requests
        from api.config.ssl import requests_verify

        from api.config.timeout import resolve_integration_http_timeout
        resp = requests.get(
            url, headers=self._auth_headers(), timeout=resolve_integration_http_timeout(),
            verify=requests_verify(),
        )
        if resp.status_code >= 400:
            logger.warning(
                "Attachment download %s -> HTTP %s", url, resp.status_code
            )
            raise ValueError(
                f"Attachment download failed (HTTP {resp.status_code})"
            )
        return resp.content

    # ---- IntegrationConnector interface ----------------------------------
    def test(self) -> Dict[str, Any]:
        if not self.is_configured():
            return {"success": False, "message": "Confluence base_url/token not configured."}
        try:
            # Try Cloud v2 endpoint first, then Server/DC v1 endpoint fallback
            data = None
            try:
                data = self._get("/wiki/api/v2/spaces", params={"limit": 1})
            except Exception as e_v2:
                try:
                    data = self._get("/rest/api/space", params={"limit": 1})
                except Exception:
                    raise e_v2
            results = (data or {}).get("results") or (data or {}).get("spaces") or []
            n = len(results) if isinstance(results, list) else 1
            return {"success": True, "message": f"Connected to Confluence ({n} space(s) reachable)."}
        except Exception as e:
            return {"success": False, "message": f"Confluence connection failed: {e}"}

    def list_spaces(self) -> List[Dict[str, Any]]:
        if not self.is_configured():
            return []
        configured_space = self.config.get("space")
        out: List[Dict[str, Any]] = []
        try:
            url: Optional[str] = "/wiki/api/v2/spaces?limit=100"
            for _ in range(5):
                if not url:
                    break
                data = self._get(url if url.startswith("/wiki") else url)
                for sp in (data or {}).get("results", []) or []:
                    if not isinstance(sp, dict):
                        continue
                    out.append(
                        {
                            "id": str(sp.get("id") or sp.get("key") or ""),
                            "title": sp.get("name") or sp.get("key") or "",
                            "type": "space",
                            "key": sp.get("key") or "",
                        }
                    )
                nxt = ((data or {}).get("_links") or {}).get("next")
                url = nxt if isinstance(nxt, str) else None
        except Exception as e:
            logger.warning("Confluence list_spaces failed: %s", e)
            return out
        if configured_space:
            filtered = [s for s in out if s.get("key") == configured_space or s.get("id") == configured_space]
            if filtered:
                return filtered
        return out

    # ---- page fetching ---------------------------------------------------
    def _fetch_page(self, page_id: str) -> Dict[str, Any]:
        """Fetch a single page with its body in export (storage HTML) format."""
        data = self._get(f"/wiki/api/v2/pages/{page_id}", params={"body-format": "export"})
        body = (data or {}).get("body") or {}
        # v2 export body shape: {"body": {"representation": "storage", "value": "<html>"}}
        html = ""
        if isinstance(body, dict):
            html = body.get("value") or ""
        return {
            "id": str(data.get("id") or page_id),
            "title": data.get("title") or page_id,
            "html": html,
            "space_id": (data or {}).get("spaceId"),
        }

    def _fetch_children(self, page_id: str, limit: int = _MAX_CHILDREN) -> List[Dict[str, Any]]:
        try:
            data = self._get(f"/wiki/api/v2/pages/{page_id}/children", params={"limit": limit})
            results = (data or {}).get("results", []) or []
            return [r for r in results if isinstance(r, dict)]
        except Exception as e:
            logger.warning("Confluence children of %s failed: %s", page_id, e)
            return []

    def _fetch_attachments(self, page_id: str) -> List[Dict[str, Any]]:
        try:
            data = self._get(f"/wiki/api/v2/pages/{page_id}/attachments", params={"limit": 50})
            results = (data or {}).get("results", []) or []
            return [r for r in results if isinstance(r, dict)]
        except Exception as e:
            logger.warning("Confluence attachments of %s failed: %s", page_id, e)
            return []

    def _convert_attachments(self, page_id: str) -> List[Dict[str, str]]:
        """Download + markitdown-convert attachments for a page."""
        converted: List[Dict[str, str]] = []
        for att in self._fetch_attachments(page_id):
            title = att.get("title") or att.get("filename") or "attachment"
            # v2 attachment download link lives under _links.download or attachments.download.
            download = None
            links = att.get("_links") or att.get("links") or {}
            if isinstance(links, dict):
                download = links.get("download")
            if not download:
                continue
            # Resolve relative links against the base URL.
            if str(download).startswith("/"):
                download = f"{self._base()}{download}"
            try:
                raw = self._get_bytes(download)
                md = convert_to_markdown(raw, filename=title)
                converted.append({"filename": title, "markdown": md})
            except Exception as e:
                logger.warning("Failed to convert Confluence attachment %s: %s", title, e)
                converted.append({"filename": title, "markdown": f"<!-- failed to convert {title}: {e} -->\n"})
        return converted

    def _pull_page_tree(
        self, page_id: str, depth: int, recursive: bool
    ) -> List[Dict[str, Any]]:
        """Fetch a page and (optionally) its descendants. Returns a list of pages."""
        page = self._fetch_page(page_id)
        pages = [page]
        if recursive and depth < _MAX_TREE_DEPTH:
            for child in self._fetch_children(page["id"]):
                if len(pages) >= _MAX_CHILDREN:
                    break
                pages.extend(self._pull_page_tree(str(child.get("id")), depth + 1, recursive))
        return pages

    @staticmethod
    def _pages_to_markdown(pages: List[Dict[str, Any]]) -> str:
        """Concatenate fetched pages into a single markdown document."""
        parts: List[str] = []
        for i, p in enumerate(pages):
            if i == 0:
                parts.append(f"# {p['title']}")
            else:
                parts.append(f"\n## {p['title']}")
            parts.append("")
            parts.append(f"<!-- page_id={p['id']} -->")
            html = p.get("html") or ""
            if html:
                # Keep the storage-format HTML inline; the frontend renders raw
                # HTML via rehype-raw. markitdown is reserved for binary attachments.
                parts.append(html)
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _extract_page_id(source_id: str) -> str:
        """Normalize a pull source into a Confluence page id.

        Accepts either a raw numeric page id or a full page URL (issue #6):

        - ``https://host/wiki/spaces/KEY/pages/12345/Title``  (friendly URL)
        - ``https://host/wiki/spaces/KEY/pages/12345``
        - ``https://host/pages/viewpage.action?pageId=12345`` (Server/DC)

        Anything else non-numeric is passed through verbatim (legacy callers
        may still send string ids).
        """
        src = (source_id or "").strip()
        if not src:
            raise ValueError("Confluence page id or URL must not be empty.")
        if src.isdigit():
            return src
        if "://" in src or src.startswith("/"):
            match = _URL_PAGE_ID_QS_RE.search(src) or _URL_PAGES_RE.search(src)
            if match:
                return match.group(1)
            raise ValueError(
                "Could not find a page id in the Confluence URL "
                "(expected /pages/{id}/… or ?pageId={id})."
            )
        return src

    def pull(self, source_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Pull a Confluence page (and optionally its descendants + attachments).

        ``source_id`` may be a raw page id or a full page URL (see
        :meth:`_extract_page_id`). Returns a structured pages list with `id`,
        `title`, `html`, `parent_id` for KnowledgeNode tree creation.
        """
        if not self.is_configured():
            raise ValueError("Confluence base_url/token not configured.")
        opts = opts or {}
        recursive = bool(opts.get("recursive"))
        page_id = self._extract_page_id(source_id)
        pages = self._pull_page_tree(page_id, depth=0, recursive=recursive)
        if not pages:
            raise ValueError(f"Confluence page {page_id} not found.")
        markdown = self._pages_to_markdown(pages)
        attachments = self._convert_attachments(pages[0]["id"]) if not recursive else []
        root = pages[0]
        return {
            "title": root["title"],
            "markdown": markdown,
            "pages": pages,
            "attachments": attachments,
            "page_id": root["id"],
            "space_id": root.get("space_id"),
            "page_count": len(pages),
            "source": "confluence",
        }
