"""Utilities package (abstraction layer 11).

Re-exports public names from the focused submodules so existing
``from api.utils import X`` call sites keep working after the split:
  * ``api.utils.logging``  — ``setup_logging`` etc.
  * ``api.utils.mcp``      — ``LocalMcpClient``, ``invoke_mcp_tool``, ...
  * ``api.utils.llm_tokens`` — ``get_model_context_window``, ``_count_tokens``, ...
"""

from api.utils.logging import (
    IgnoreLogChangeDetectedFilter,
    _TruncatingFormatter,
    setup_logging,
)
from api.utils.mcp import (
    LocalMcpClient,
    get_local_mcp_client,
    invoke_mcp_tool,
    list_all_mcp_tools,
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
    # mcp
    "LocalMcpClient",
    "get_local_mcp_client",
    "invoke_mcp_tool",
    "list_all_mcp_tools",
    # llm_tokens
    "_MODEL_CTX_CACHE",
    "_count_tokens",
    "get_model_context_window",
]
