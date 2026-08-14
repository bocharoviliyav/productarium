"""Documentation generation pipeline.

Submodules:
- ``_common``   — shared helpers (event loop, LLM wrapper, persistence, indexing).
- ``codebase``  — codebase docgen (RLM long-context + standard-LLM fallback).
- ``spec``      — OpenAPI/AsyncAPI docgen (stdlib render + LLM enrich).
- ``wiki``      — sequential 7-section wiki generator (prompt dispatch + context).
- ``summary``   — AI product summary over codebases/specs + knowledge nodes.
- ``jobs``      — async 202 + poll job registry for codebase/spec doc generation.

Each generate endpoint calls its generator directly (no polymorphic dispatcher).
``set_main_event_loop`` (re-exported from :mod:`api.docgen._common`) is called
once from ``api.api.startup_event``.
"""

from api.docgen._common import set_main_event_loop, _index_in_background
from api.docgen.codebase import generate_codebase_docs
from api.docgen.spec import generate_openapi_docs, generate_asyncapi_docs

__all__ = [
    "generate_codebase_docs",
    "generate_openapi_docs",
    "generate_asyncapi_docs",
    "set_main_event_loop",
    "_index_in_background",
]
