"""LLM client layer for Productarium (LangChain-based).

The single source for chat models over the OpenAI-compatible server:

- :func:`api.llm.client.build_chat_model` — ChatOpenAI factory (auth
  placeholder handling, corporate TLS, central timeouts, model params).
- :class:`api.llm.generate.GenerateLLM` — non-streaming generation with
  backoff retry (replaces the adalflow ``_StandardLLM`` / ``_ExpertLLM`` /
  ``_SummaryLLM`` wrappers).
- :func:`api.llm.stream.stream_chat` / :func:`stream_chat_fields` —
  token-streaming helpers (content + reasoning deltas) for the SSE paths.
"""

from api.llm.client import (
    NO_AUTH_PLACEHOLDER,
    build_chat_model,
    build_http_async_client,
    is_local_endpoint,
    is_no_auth_placeholder,
    resolve_api_key,
)
from api.llm.generate import GenerateLLM
from api.llm.stream import stream_chat, stream_chat_fields

__all__ = [
    "NO_AUTH_PLACEHOLDER",
    "GenerateLLM",
    "build_chat_model",
    "build_http_async_client",
    "is_local_endpoint",
    "is_no_auth_placeholder",
    "resolve_api_key",
    "stream_chat",
    "stream_chat_fields",
]
