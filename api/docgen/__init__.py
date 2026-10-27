"""Documentation generation pipeline.

Submodules:
- ``_common``   — shared helpers (event loop, LLM wrapper, persistence, indexing).
- ``codebase``  — codebase docgen (deepagents orchestrator + subagents, standard-LLM
  fallback, verification pipeline, adaptive small-context caps).
- ``spec``      — OpenAPI/AsyncAPI docgen (stdlib render + LLM enrich).
- ``summary``   — AI product summary over codebases/specs + knowledge nodes.
- ``jobs``      — async 202 + poll job registry with a live progress model.

Each generate endpoint calls its generator directly (no polymorphic
dispatcher — import generators from their submodules). Re-exported here:
``set_main_event_loop`` (called once from ``api.api.startup_event``) and
``_index_in_background`` (used by the knowledge router).
"""

from api.docgen._common import set_main_event_loop, _index_in_background

__all__ = [
    "set_main_event_loop",
    "_index_in_background",
]
