"""Confluence connector (plan section G).

Talks to the Confluence Cloud REST API v2 (``<base_url>/wiki/api/v2``) to list
spaces, fetch page bodies (export/storage format), walk child pages, and
download attachments. Binary attachments (docx/pdf/pptx/xlsx/...) are
converted to markdown via :mod:`api.markitdown_client`.

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
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from api.integrations.base import IntegrationConnector
from api.markitdown_client import convert_to_markdown

logger = logging.getLogger(__name__)

# Max recursion depth when pulling a page tree (opts={"recursive": True}).
_MAX_TREE_DEPTH = 3
# Max child pages fetched per page (keeps pulls bounded).
_MAX_CHILDREN = 50


class ConfluenceConnector(IntegrationConnector):
    name = "confluence"
    display_name = "Confluence"
    description = "Pull Confluence spaces/pages (Cloud REST API v2) with markitdown attachments."
    requires_credentials = True

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        from api.settings_store import get_confluence_creds

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
        return (self.config.get("base_url") or "").rstrip("/")

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET a Confluence v2 API path. Returns parsed JSON (dict).

        Raises on non-2xx so callers can convert to a {success: False} result.
        Mock this method in tests to avoid real HTTP.
        """
        import requests
        from api.ssl_config import requests_verify

        url = f"{self._base()}{path}"
        resp = requests.get(
            url, headers=self._auth_headers(), params=params, timeout=20,
            verify=requests_verify(),
        )
        if resp.status_code >= 400:
            raise ValueError(f"Confluence API {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _get_bytes(self, url: str) -> bytes:
        """Download raw bytes (attachment download link)."""
        import requests
        from api.ssl_config import requests_verify

        resp = requests.get(
            url, headers=self._auth_headers(), timeout=30,
            verify=requests_verify(),
        )
        if resp.status_code >= 400:
            raise ValueError(f"Attachment download {url} -> HTTP {resp.status_code}")
        return resp.content

    # ---- IntegrationConnector interface ----------------------------------
    def test(self) -> Dict[str, Any]:
        if not self.is_configured():
            return {"success": False, "message": "Confluence base_url/token not configured."}
        try:
            data = self._get("/wiki/api/v2/spaces", params={"limit": 1})
            n = len((data or {}).get("results", []) or [])
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
            # Follow cursor pagination (bounded).
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
                # v2 cursor pagination: _links.next
                nxt = ((data or {}).get("_links") or {}).get("next")
                url = nxt if isinstance(nxt, str) else None
        except Exception as e:
            logger.warning("Confluence list_spaces failed: %s", e)
            return out
        # If a specific space is configured, filter to it (still return all if none match).
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

    def pull(self, source_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Pull a Confluence page (and optionally its descendants + attachments).

        Args:
            source_id: A Confluence page id.
            opts: ``{"recursive": bool}`` — also pull descendant pages (bounded
                to depth 3 / 50 pages). Default False.
        """
        if not self.is_configured():
            raise ValueError("Confluence base_url/token not configured.")
        opts = opts or {}
        recursive = bool(opts.get("recursive"))
        pages = self._pull_page_tree(source_id, depth=0, recursive=recursive)
        if not pages:
            raise ValueError(f"Confluence page {source_id} not found.")
        markdown = self._pages_to_markdown(pages)
        # Convert attachments for the root page (child-page attachments are
        # included only when recursive; kept bounded to avoid huge pulls).
        attachments = self._convert_attachments(pages[0]["id"]) if not recursive else []
        root = pages[0]
        return {
            "title": root["title"],
            "markdown": markdown,
            "attachments": attachments,
            "page_id": root["id"],
            "space_id": root.get("space_id"),
            "page_count": len(pages),
            "source": "confluence",
        }
