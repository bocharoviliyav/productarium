"""Documentation generation pipeline.

Submodules:
- ``_common``    — shared helpers (event loop, LLM wrapper, persistence, indexing).
- ``codebase``    — codebase docgen (RLM long-context + standard-LLM fallback).
- ``spec``        — OpenAPI/AsyncAPI/Testcase docgen (stdlib render + LLM enrich).
- ``simple``      — links/documentation/guides docgen (passthrough + LLM enrich).
- ``dispatcher``  — routes by ``artifact.type`` to the 7 sub-generators.
- ``wiki``        — sequential 7-section wiki generator (prompt dispatch + context).
- ``summary``     — AI product summary over artifacts + knowledge nodes.
- ``jobs``        — async 202 + poll job registry for artifact doc generation.

The public entry point is :func:`generate_artifact_documentation` (re-exported
from :mod:`api.docgen.dispatcher`). ``set_main_event_loop`` (re-exported from
:mod:`api.docgen._common`) is called once from ``api.api.startup_event``.
"""

from api.docgen._common import set_main_event_loop
from api.docgen.dispatcher import (
    generate_artifact_documentation,
    generate_codebase_docs,
    generate_openapi_docs,
    generate_asyncapi_docs,
    generate_testcase_docs,
    generate_links_docs,
    generate_documentation_docs,
    generate_guides_docs,
    _index_in_background,
)

__all__ = [
    "generate_artifact_documentation",
    "generate_codebase_docs",
    "generate_openapi_docs",
    "generate_asyncapi_docs",
    "generate_testcase_docs",
    "generate_links_docs",
    "generate_documentation_docs",
    "generate_guides_docs",
    "set_main_event_loop",
    "_index_in_background",
]
