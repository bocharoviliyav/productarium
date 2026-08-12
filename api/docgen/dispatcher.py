"""Artifact documentation dispatcher (routes by ``artifact.type``).

Split out of the former ``api/artifact_docgen.py`` (Step 4). Routes the
artifact-type enum (codebase|spec|links|documentation|guides) and maps legacy
types (openapi/asyncapi/testcase) to the new (type, kind) pairs via
``LEGACY_ARTIFACT_TYPE_MAP`` so calls from clients still using the legacy
vocabulary keep working.

The 7 sub-generators are imported by name into this module's namespace so tests
that monkeypatch them on the dispatcher module (``adg.generate_codebase_docs =
...``) take effect — the dispatcher looks them up as module globals at call
time, not via a captured import-time binding.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from api.utils import setup_logging
from api.models import LEGACY_ARTIFACT_TYPE_MAP
from api.docgen._common import _index_in_background  # re-exported for test compat
from api.docgen.codebase import generate_codebase_docs
from api.docgen.spec import (
    generate_openapi_docs,
    generate_asyncapi_docs,
    generate_testcase_docs,
)
from api.docgen.simple import (
    generate_links_docs,
    generate_documentation_docs,
    generate_guides_docs,
)

setup_logging()
logger = logging.getLogger(__name__)


async def generate_artifact_documentation(
    artifact: Any,
    product: Any,
    provider: str = None,
    model: Optional[str] = None,
    language: str = "ru",
) -> str:
    """Dispatch documentation generation by ``artifact.type`` (and ``kind``).

    Routes the new artifact-type enum (codebase|spec|links|documentation|guides)
    and maps legacy types (openapi/asyncapi/testcase) to the new (type, kind)
    pairs via ``LEGACY_ARTIFACT_TYPE_MAP`` so calls from clients still using the
    legacy vocabulary keep working:
    - ``codebase``         -> 7 wiki sections (RLM/standard LLM).
    - ``spec``             -> by ``kind``: ``asyncapi`` -> asyncapi render+LLM,
      otherwise (``openapi``) -> openapi render+LLM.
    - ``links``            -> links index page (no heavy gen).
    - ``documentation``    -> by ``kind``: ``testcase`` -> testcase render+LLM,
      otherwise manual/generated MD passthrough + optional LLM enrichment.
    - ``guides``           -> manual/generated MD passthrough.

    Returns the generated markdown and persists it onto ``artifact.generated_docs``
    + ``artifact.pages``. All generated content is indexed into the product-scoped
    cognee dataset ``prod_{product_id}`` (item 1 cognee-first). All backends
    degrade gracefully: cognee indexing is non-blocking, and RLM/LLM failures
    fall back to deterministic renders.
    """
    atype = (getattr(artifact, "type", "") or "").strip().lower()
    kind = (getattr(artifact, "kind", "") or "").strip().lower()
    # Route legacy types (openapi/asyncapi/testcase) to the new (type, kind).
    if atype in LEGACY_ARTIFACT_TYPE_MAP:
        atype, default_kind = LEGACY_ARTIFACT_TYPE_MAP[atype]
        kind = kind or default_kind

    if atype == "codebase":
        return await generate_codebase_docs(artifact, product, provider, model, language)
    if atype == "spec":
        if kind == "asyncapi":
            return await generate_asyncapi_docs(artifact, product, provider, model, language)
        return await generate_openapi_docs(artifact, product, provider, model, language)
    if atype == "links":
        return await generate_links_docs(artifact, product, provider, model, language)
    if atype == "documentation":
        if kind == "testcase":
            return await generate_testcase_docs(artifact, product, provider, model, language)
        return await generate_documentation_docs(artifact, product, provider, model, language)
    if atype == "guides":
        return await generate_guides_docs(artifact, product, provider, model, language)
    raise ValueError(f"Unsupported artifact type: {atype!r}")


__all__ = [
    "generate_artifact_documentation",
    "generate_codebase_docs",
    "generate_openapi_docs",
    "generate_asyncapi_docs",
    "generate_testcase_docs",
    "generate_links_docs",
    "generate_documentation_docs",
    "generate_guides_docs",
    "_index_in_background",
]
