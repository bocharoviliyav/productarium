"""Utilities package (abstraction layer 11).

Re-exports public names from the focused submodules so existing
``from api.utils import X`` call sites keep working after the split:
  * ``api.utils.logging``  — ``setup_logging`` etc.
  * ``api.utils.llm_tokens`` — ``get_model_context_window``, ``_count_tokens``, ...

(The former ``api.utils.mcp`` re-exports were removed together with the
legacy hand-written MCP client; MCP is handled by ``api/mcp/`` since Wave C.)
"""

from api.utils.logging import (
    IgnoreLogChangeDetectedFilter,
    _TruncatingFormatter,
    setup_logging,
)
from api.utils.llm_tokens import (
    _MODEL_CTX_CACHE,
    _count_tokens,
    get_model_context_window,
)

__all__ = [
    # logging
    "IgnoreLogChangeDetectedFilter",
    "_TruncatingFormatter",
    "setup_logging",
    # llm_tokens
    "_MODEL_CTX_CACHE",
    "_count_tokens",
    "get_model_context_window",
]
