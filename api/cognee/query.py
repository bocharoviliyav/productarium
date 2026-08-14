from __future__ import annotations

import asyncio
import logging

from api.cognee._runtime import _COGNEE_AVAILABLE, cognee
from api.cognee.config import apply_cognee_runtime_config

logger = logging.getLogger(__name__)

def _recall_response_to_text(result) -> str:
    """Extract a display string from a cognee V1 ``RecallResponse`` object.

    cognee V1 ``recall()`` returns a list of Pydantic ``RecallResponse``
    objects (a discriminated union on the ``source`` field), NOT plain
    strings. The variants carry their text under different attribute names:

    - ``ResponseGraphEntry`` (source="graph")        -> ``.text``
    - ``ResponseQAEntry``     (source="session")      -> ``.answer``
    - ``ResponseGraphContextEntry`` (source="graph_context") -> ``.content``
    - ``ResponseSessionContextEntry`` (source="session_context") -> ``.content``
    - ``ResponseAgentTraceEntry`` (source="trace")    -> various fields

    We probe by attribute so this stays robust to cognee minor-version
    renames without importing the (potentially absent) model classes. A
    plain ``str`` result (older cognee, or a remote-client unwrap) is
    returned as-is. Anything else falls back to ``str(result)``.
    """
    if isinstance(result, str):
        return result
    for attr in ("text", "answer", "content"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    out = str(result)
    return out if out and out.strip() else ""


async def query_cognee(query: str, dataset_name: str, top_k: int = 20) -> str:
    """Query the cognee knowledge graph for a product and return retrieved
    context as a single string. NEVER raises; returns "" on any error.

    V1 note: ``recall()`` searches ALL user-readable datasets when
    ``datasets`` is omitted, which would break product isolation. We pass
    ``datasets=[dataset_name]`` to scope retrieval to the product's dataset.

    recall's GRAPH_COMPLETION makes an LLM completion call internally; on a
    slow/contended local model that can hang indefinitely. We cap it with
    ``resolve_cognee_recall_timeout`` (default 300s) and return "" on timeout
    so the expert path falls back to artifact docs instead of stalling the
    SSE stream until the HTTP proxy times out.
    """
    if not _COGNEE_AVAILABLE:
        return ""
    apply_cognee_runtime_config()
    try:
        from cognee import SearchType
        from api.config.timeout import resolve_cognee_recall_timeout
        recall_timeout = resolve_cognee_recall_timeout()
        logger.info("Querying Cognee knowledge graph (dataset: %s, timeout: %.0fs)...", dataset_name, recall_timeout)
        results = await asyncio.wait_for(
            cognee.recall(
                query_text=query,
                query_type=SearchType.GRAPH_COMPLETION,
                datasets=[dataset_name],
                top_k=top_k,
            ),
            timeout=recall_timeout,
        )
        if not results:
            return ""
        # cognee V1 recall() returns list[RecallResponse] (Pydantic objects,
        # not strings); extract a display string from each variant.
        pieces = [_recall_response_to_text(r) for r in results]
        pieces = [p for p in pieces if p]
        return "\n\n".join(pieces)
    except asyncio.TimeoutError:
        logger.warning(
            "Cognee recall timed out for dataset %r; falling back to artifact docs.",
            dataset_name,
        )
        return ""
    except Exception as e:
        logger.error("Error querying Cognee: %s", e, exc_info=True)
        return ""
