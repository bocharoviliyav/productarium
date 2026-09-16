"""Agent layer for Productarium (LangChain/LangGraph migration).

Wave B (expert agent) modules:

- :mod:`api.agents.tools` — product-scoped agent tools
  (``knowledge_recall``, ``codebase_file_read``, ``spec_read``,
  ``link_read``, ``node_read``); ``product_id`` is closure-bound, never an
  LLM-controlled argument.
- :mod:`api.agents.runtime` — process-wide LangGraph checkpointer
  (Postgres → SQLite file → in-memory fallback chain).
- :mod:`api.agents.expert` — the expert agent itself
  (``create_react_agent``) plus the typed stream mapper
  (``run_agent_chat_stream``).

Later waves add: MCP platform (C), docgen agents (D), Deep Research (E).

The heavy langchain/langgraph imports live inside the submodules'
functions, so importing this package stays cheap; import the submodules
themselves (``api.agents.expert`` etc.) for the concrete machinery.
"""

from __future__ import annotations

__all__ = ["expert", "runtime", "tools"]
