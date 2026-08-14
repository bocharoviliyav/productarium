"""Expert agent package — product-scoped chat + document generation.

Split out of the former ``api/expert_agent.py`` (Step 6 of the backend
decomposition). Submodules:
- ``llm``: ``_ExpertLLM`` wrapper, ``_safe_build_llm``, ``_extract_chunk_fields``,
  ``_resolve_expert_model``.
- ``knowledge``: ``_retrieve_product_knowledge``, ``_fallback_artifact_docs``,
  ``_product_name_by_id``, ``_format_history``.
- ``prompt``: tunables (``RLM_MIN_CHARS`` etc.), ``EXPERT_SYSTEM_PROMPT`` /
  ``EXPERT_DOC_PROMPT``, ``_clean_llm_text``, ``_chunk_text``, ``_build_prompt``.
- ``generate``: ``_rlm_generate``, ``_generate_answer``, ``_stream_answer``,
  ``_resolve_use_rlm``.
- ``chat``: ``run_expert_chat``, ``run_expert_doc``,
  ``_run_expert_chat_collect`` / ``_run_expert_chat_stream``.

Public API: ``run_expert_chat``, ``run_expert_doc``. Internal names are
re-exported here for backward-compatible access (``api.expert.<name>``).
Patch-then-call tests must patch on the use-site submodule where the calling
function looks up the dependency — e.g. ``api.expert.generate._safe_build_llm``
(not here), because ``_generate_answer`` resolves it from ``generate``'s globals.
"""

from __future__ import annotations

from api.expert.chat import (  # noqa: F401
    _run_expert_chat_collect,
    _run_expert_chat_stream,
    run_expert_chat,
    run_expert_doc,
)
from api.expert.generate import (  # noqa: F401
    _generate_answer,
    _resolve_use_rlm,
    _rlm_generate,
    _stream_answer,
)
from api.expert.knowledge import (  # noqa: F401
    _fallback_artifact_docs,
    _format_history,
    _product_name_by_id,
    _retrieve_product_knowledge,
)
from api.expert.llm import (  # noqa: F401
    _ExpertLLM,
    _ThinkingStreamParser,
    _extract_chunk_fields,
    _resolve_expert_model,
    _safe_build_llm,
    _strip_thinking_tags,
)
from api.expert.prompt import (  # noqa: F401
    EXPERT_DOC_PROMPT,
    EXPERT_SYSTEM_PROMPT,
    KNOWLEDGE_MAX_CHARS,
    RLM_MIN_CHARS,
    STREAM_CHUNK_SIZE,
    _build_prompt,
    _chunk_text,
    _clean_llm_text,
)
from api.expert.prompt import _safe_replace  # noqa: F401
from api.expert.types import (  # noqa: F401
    EVENT_ANSWERING,
    EVENT_CONTENT,
    EVENT_ERROR,
    EVENT_REASONING,
    EVENT_RETRIEVING,
    EVENT_STATUS,
    EVENT_THINKING,
    ExpertStreamEvent,
)

__all__ = [
    "run_expert_chat",
    "run_expert_doc",
    "ExpertStreamEvent",
    "EXPERT_SYSTEM_PROMPT",
    "EXPERT_DOC_PROMPT",
]
