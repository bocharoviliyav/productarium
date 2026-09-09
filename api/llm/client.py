"""Factory for ChatOpenAI chat models over the OpenAI-compatible server.

This module replaces the former adalflow ``OpenAIClient``
(``api/clients/openai_client.py``) with langchain-openai's :class:`ChatOpenAI`
as the single LLM client. Every supported server (LM Studio, llama.cpp, vLLM,
text-generation-webui, a corporate AI gateway) exposes the OpenAI-compatible
``/v1`` API, so one client covers all cases.

Behavior carried over from the former client:

- **API-key placeholder handling**: the values ``not-needed`` / ``not_needed``
  mean "no auth". The OpenAI SDK requires a non-empty key, so the placeholder
  is passed and the ``Authorization`` header is dropped at send time via an
  httpx event hook. Keyless operation is only allowed when the placeholder
  was set EXPLICITLY (the ``api_key`` argument or the
  ``LOCAL_OPENAI_API_KEY`` environment variable — a clear "no auth" signal
  for any endpoint, e.g. a corporate gateway) OR when the endpoint is local.
  The module-level config default (``api.config.LOCAL_OPENAI_API_KEY``) does
  NOT count as an explicit signal, so a missing key on a remote endpoint —
  almost certainly a misconfiguration — raises at build time so it is
  noticed.
- **Corporate TLS**: the httpx client is built with ``httpx_verify()`` from
  :mod:`api.config.ssl` so a corporate CA bundle / skip-verify setting reaches
  the enterprise AI gateway.
- **Timeouts**: the per-request timeout resolves through
  :mod:`api.config.timeout` (admin > env > default; default 3600s, floor 60s)
  so long-running generations on a local model are not aborted mid-flight.
- **Model params**: temperature / top_p / seed come from
  ``api/config/generator.json`` via :func:`api.config.get_model_config`;
  admin-configured ``models.<task>.{model,base_url,api_key}`` is threaded
  through by the callers (expert / docgen / summary).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import httpx
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Value the OpenAI SDK is given when no real key is configured. The SDK
# requires a non-empty api_key, so the placeholder is passed here and the
# Authorization header is stripped at send time (see ``_drop_auth_header``).
NO_AUTH_PLACEHOLDER = "not-needed"

# Explicit "no auth" placeholders accepted from env/admin configuration.
_NO_AUTH_PLACEHOLDERS = ("not-needed", "not_needed")

# Hosts considered local for the keyless-endpoint policy. An endpoint on the
# loopback interface may legitimately run without authentication.
_LOCAL_HOSTS = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "[::]",
    "[::1]",
    "host.docker.internal",
)

# Model parameters passed as first-class ChatOpenAI constructor fields. Any
# other configured parameter falls through to ``model_kwargs`` (extra body
# params on the /chat/completions call).
_CHAT_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "n",
        "stop",
        "logprobs",
        "top_logprobs",
    }
)


def is_local_endpoint(base_url: Optional[str]) -> bool:
    """True when the base URL points at a local/loopback endpoint.

    Exact hostname comparison (P0-10): the previous substring check let
    ``http://localhost.attacker.com`` pass as "local" and inherit the
    keyless-endpoint policy. Parse the URL and compare the hostname EXACTLY
    against ``_LOCAL_HOSTS`` (IPv6 literals unbracketed; ``urlsplit`` returns
    them bare, the tuple keeps bracketed entries for readability); anything
    unparseable or host-less is not local.
    """
    if not base_url:
        return False
    try:
        from urllib.parse import urlsplit

        hostname = urlsplit(base_url).hostname
    except ValueError:
        return False
    if not hostname:
        return False
    host = hostname.lower()
    return host in _LOCAL_HOSTS or f"[{host}]" in _LOCAL_HOSTS


def is_no_auth_placeholder(api_key: Optional[str]) -> bool:
    """True ONLY for the explicit 'no auth' placeholder values.

    A missing/empty key is NOT a placeholder signal: treating it as one
    silently turned misconfigured remote endpoints into no-auth clients.
    Use :func:`resolve_api_key` for "is there a real key" semantics.
    """
    if not api_key:
        return False
    from api.config.settings import _sanitize_api_key

    clean = _sanitize_api_key(api_key)
    return bool(clean) and clean.lower() in _NO_AUTH_PLACEHOLDERS


def resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    """Resolve the effective API key, honoring the 'no-auth' placeholder.

    Custom keys (UUIDs, hex tokens, ``sk-*``, JWTs) are sanitized and returned
    as-is. Placeholders (``not-needed`` / ``not_needed``) or empty strings
    return ``None``, signalling the client to strip the Bearer header for
    unauthenticated endpoints.
    """
    if not api_key:
        return None
    from api.config.settings import _sanitize_api_key

    clean = _sanitize_api_key(api_key)
    if not clean or clean.lower() in _NO_AUTH_PLACEHOLDERS:
        return None
    return clean


async def _drop_auth_header(request: httpx.Request) -> None:
    """httpx async request event hook: drop the Authorization header.

    The OpenAI SDK sets ``Authorization: Bearer <api_key>`` on every request
    at send time, so merely passing a placeholder key is not enough to
    suppress it for no-auth gateways. This hook removes the header entirely
    when no real key was resolved.

    MUST be a coroutine: httpx awaits every event hook on ``AsyncClient``
    (``await hook(request)``), so a sync hook returning ``None`` raises
    ``TypeError: object NoneType can't be used in 'await' expression`` and
    every request fails with a misleading "Connection error".
    """
    request.headers.pop("authorization", None)
    request.headers.pop("Authorization", None)


def _ssl_verify() -> Union[bool, str]:
    """Resolve the TLS verify value (corporate CA bundle / skip-verify).

    Reads admin settings + env via :mod:`api.config.ssl` so a runtime save
    takes effect when the client is (re)built. Returns True / False / a CA
    bundle path.
    """
    try:
        from api.config.ssl import httpx_verify

        return httpx_verify()
    except Exception:  # pragma: no cover - defensive
        return True


def build_http_async_client(strip_auth: bool = False) -> httpx.AsyncClient:
    """Build the shared httpx async client (corporate TLS + long timeout).

    Args:
        strip_auth: Register the Authorization-dropping event hook (used when
            only the no-auth placeholder key was resolved).
    """
    from api.config.timeout import resolve_llm_request_timeout

    client = httpx.AsyncClient(
        verify=_ssl_verify(),
        timeout=resolve_llm_request_timeout(),
    )
    if strip_auth:
        client.event_hooks["request"].append(_drop_auth_header)
    return client


def _resolve_credentials(
    base_url: Optional[str],
    api_key: Optional[str],
) -> str:
    """Resolve the api_key value to hand to the OpenAI SDK.

    Returns the sanitized real key, or the no-auth placeholder when allowed.
    Raises ``ValueError`` for a missing key on a remote endpoint.

    Only an EXPLICIT signal allows keyless operation on a remote endpoint:
    the placeholder set via the ``api_key`` argument or the
    ``LOCAL_OPENAI_API_KEY`` environment variable. The module-level config
    default (``api.config.LOCAL_OPENAI_API_KEY``) is a fallback for LOCAL
    endpoints only — it must not silently turn a misconfigured remote
    endpoint into a no-auth client.
    """
    from api.config import LOCAL_OPENAI_API_KEY, LOCAL_OPENAI_BASE_URL

    resolved_base_url = base_url or os.environ.get("LOCAL_OPENAI_BASE_URL") or LOCAL_OPENAI_BASE_URL
    explicit_key = api_key or os.environ.get("LOCAL_OPENAI_API_KEY")
    api_key_raw = explicit_key or LOCAL_OPENAI_API_KEY

    resolved_key = resolve_api_key(api_key_raw)
    if resolved_key:
        return resolved_key

    # No real key resolved. Allowed only when the user EXPLICITLY set the
    # placeholder (arg/env — a clear "no auth" signal) or the endpoint is
    # local.
    if (
        explicit_key
        and is_no_auth_placeholder(explicit_key)
    ) or is_local_endpoint(resolved_base_url):
        logger.info(
            "Using placeholder API key for the no-auth endpoint: %s",
            resolved_base_url,
        )
        return NO_AUTH_PLACEHOLDER

    raise ValueError(
        "An API key (LOCAL_OPENAI_API_KEY or models.<task>.api_key) must be "
        f"set for remote endpoints ({resolved_base_url})"
    )


def strip_non_tool_message_names(messages: Any) -> Any:
    """Return ``messages`` with the optional ``name`` dropped from every
    non-tool message.

    Local OpenAI-compatible servers validate the chat schema strictly:
    LM Studio (and several vLLM builds) reject a request with HTTP 400
    ``"name" is only valid on role="tool" messages`` whenever any message
    with ``role != "tool"`` carries a ``name`` field. Such names reach the
    outgoing payload even though we never set them ourselves:

    - langchain-openai parses a server response into
      ``AIMessage(name=...)`` when the server includes a ``name`` in the
      assistant message, and then *re-sends* it on every subsequent
      request of the same conversation;
    - third-party agent middleware (deepagents subagent handoffs) may
      attach names when merging transcripts.

    On the wire ``name`` is optional author metadata (deprecated even by
    OpenAI); tool association runs through ``tool_call_id``. Dropping it
    for non-tool messages is therefore loss-free, while tool/function
    messages keep theirs (``name`` is the documented tool name there and
    strict servers accept it on those roles).

    Messages are copied, never mutated: the same message objects live in
    LangGraph state and must not be modified in place.
    """
    stripped: list = []
    for message in messages or []:
        if getattr(message, "type", "") in ("tool", "function"):
            stripped.append(message)
            continue
        name = getattr(message, "name", None)
        extra = getattr(message, "additional_kwargs", None) or {}
        if not name and "name" not in extra:
            stripped.append(message)
            continue
        clean_extra = {k: v for k, v in extra.items() if k != "name"}
        try:
            stripped.append(
                message.model_copy(
                    update={"name": None, "additional_kwargs": clean_extra}
                )
            )
        except AttributeError:  # pragma: no cover - pydantic v1 fallback
            stripped.append(
                message.copy(update={"name": None, "additional_kwargs": clean_extra})
            )
    return stripped


class ServerCompatChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that never sends ``name`` on non-tool messages.

    Wraps every request entry point (sync/async, streaming/not) with
    :func:`strip_non_tool_message_names` so strict local servers (LM
    Studio, llama.cpp, vLLM) cannot reject an agent-loop conversation
    with ``400 "name" is only valid on role="tool" messages`` — the error
    that killed whole orchestrated docgen runs (deepagents orchestrator
    transcripts) and forced the expensive python-parallel fallback.
    """

    def _generate(self, messages, *args: Any, **kwargs: Any):
        return super()._generate(strip_non_tool_message_names(messages), *args, **kwargs)

    async def _agenerate(self, messages, *args: Any, **kwargs: Any):
        return await super()._agenerate(
            strip_non_tool_message_names(messages), *args, **kwargs
        )

    def _stream(self, messages, *args: Any, **kwargs: Any):
        yield from super()._stream(
            strip_non_tool_message_names(messages), *args, **kwargs
        )

    async def _astream(self, messages, *args: Any, **kwargs: Any):
        async for chunk in super()._astream(
            strip_non_tool_message_names(messages), *args, **kwargs
        ):
            yield chunk


def build_chat_model(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **overrides: Any,
) -> Any:
    """Build a :class:`langchain_openai.ChatOpenAI` for the configured endpoint.

    Args:
        model: Model name; falls back to the default model from
            ``api/config/generator.json``.
        base_url: Endpoint base URL (admin-configured per task, e.g.
            ``models.docgen.base_url``); defaults to ``LOCAL_OPENAI_BASE_URL``.
        api_key: API key (admin-configured per task); placeholder-aware.
        **overrides: Explicit model-parameter overrides (e.g.
            ``temperature=0``) that win over the JSON config.

    Returns:
        A configured ``ChatOpenAI`` bound to an httpx client that honors the
        corporate TLS config, the central request timeout, and the no-auth
        placeholder policy.

    Raises:
        ValueError: When no real key resolves for a remote (non-local)
            endpoint, or when no model can be resolved.
    """
    from api.config import LOCAL_OPENAI_BASE_URL, get_model_config
    from api.config.timeout import resolve_llm_request_timeout

    cfg = get_model_config(model)
    params: Dict[str, Any] = dict(cfg.get("model_kwargs") or {})
    # get_model_config resolves the default model when none is given and
    # always places the final model name inside model_kwargs.
    resolved_model = params.pop("model", None)
    if not resolved_model:
        raise ValueError("No model could be resolved from the generator config")
    params.update(overrides)

    resolved_base_url = base_url or os.environ.get("LOCAL_OPENAI_BASE_URL") or LOCAL_OPENAI_BASE_URL
    resolved_key = _resolve_credentials(base_url, api_key)

    http_async_client = build_http_async_client(
        strip_auth=resolved_key == NO_AUTH_PLACEHOLDER
    )

    ctor_kwargs: Dict[str, Any] = {
        k: v for k, v in params.items() if k in _CHAT_FIELDS
    }
    extra_kwargs: Dict[str, Any] = {
        k: v for k, v in params.items() if k not in _CHAT_FIELDS
    }

    kwargs: Dict[str, Any] = dict(
        model=resolved_model,
        api_key=resolved_key,
        base_url=resolved_base_url,
        timeout=resolve_llm_request_timeout(),
        http_async_client=http_async_client,
        **ctor_kwargs,
    )
    if extra_kwargs:
        kwargs["model_kwargs"] = extra_kwargs

    return ServerCompatChatOpenAI(**kwargs)
