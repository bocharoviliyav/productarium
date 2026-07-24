"""Central SSL/TLS configuration for HTTP clients.

Lets an admin point every outbound HTTPS call (``requests``, ``httpx``,
the OpenAI SDK, and cognee's aiohttp adapters) at a corporate CA bundle,
or skip certificate verification entirely, for an enterprise AI gateway
whose cert is signed by an internal CA not present in the default trust
store.

Two knobs, both runtime-configurable via the admin panel AND env vars:

- **CA bundle path** (``ssl.ca_bundle`` setting / ``SSL_CA_BUNDLE`` /
  ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` env). When set, the path is
  pushed into ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` / ``CURL_CA_BUNDLE``
  so ``ssl.create_default_context()`` (used by httpx + cognee's aiohttp
  ``create_secure_ssl_context``), ``requests``, and ``curl`` all trust the
  corporate root cert. Verification stays ON.

- **Skip verification** (``ssl.verify`` setting / ``SSL_VERIFY`` env). When
  explicitly ``false``, certificate verification is disabled for every
  client. This is insecure (MITM risk) and only a fallback when no CA file
  is available (e.g. the corporate cert lives only in the OS keychain and
  has not been exported). A WARNING is logged.

Defaults (empty ``.env``, no admin config): verification ON, default trust
store. Local HTTP Ollama is unaffected (no TLS).

The admin-panel values win over env vars so a runtime save takes effect
without a restart for any client that reads the value per call
(``requests`` calls, ``OpenAIClient`` httpx client). cognee's aiohttp SSL
context is built once at cognee import, so for cognee the env vars must be
set before cognee is imported (``apply_ssl_env`` is called early in
``main.py`` and ``cognee_manager`` imports ``apply_ssl_env``); a
``apply_cognee_ssl_patch`` monkeypatch handles the skip-verify case at
runtime.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Admin setting key for the CA bundle path (settings_store group "ssl").
_SSL_CA_BUNDLE_KEY = "ssl.ca_bundle"
# Admin setting key for the verify toggle (settings_store group "ssl").
_SSL_VERIFY_KEY = "ssl.verify"

_TRUTHY = ("1", "true", "t", "yes", "y", "on")
_FALSY = ("0", "false", "f", "no", "n", "off")


def _to_bool(value: Optional[Union[str, bool]], default: bool = True) -> bool:
    """Parse a truthy/falsy string/bool. Unknown values fall back to ``default``."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return default


def _setting(key: str) -> Optional[str]:
    """Read a setting from the store (best-effort; None if DB/cognee down)."""
    try:
        from api.settings_store import get_setting  # lazy: avoids circular import
        return get_setting(key)
    except Exception as e:  # pragma: no cover - DB / import unavailable
        logger.debug("ssl_config: get_setting(%r) failed: %s", key, e)
        return None


def get_ca_bundle() -> Optional[str]:
    """Resolve the CA bundle path. Admin store wins, then env vars.

    Honors ``SSL_CA_BUNDLE`` / ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE``.
    Returns a filesystem path string or ``None`` (use default trust store).
    """
    path = _setting(_SSL_CA_BUNDLE_KEY)
    if not path:
        # ``SSL_CERT_FILE`` is honored by ssl.create_default_context (httpx,
        # cognee aiohttp) and by the OpenAI SDK's httpx transport.
        # ``REQUESTS_CA_BUNDLE`` / ``CURL_CA_BUNDLE`` are honored by requests.
        path = (
            os.environ.get("SSL_CA_BUNDLE")
            or os.environ.get("SSL_CERT_FILE")
            or os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("CURL_CA_BUNDLE")
        )
    if path:
        path = str(path).strip()
        if not path:
            return None
        if not os.path.isfile(path):
            logger.warning(
                "SSL CA bundle path %r does not exist or is not a file; "
                "ignoring (TLS verification will use the default trust store).",
                path,
            )
            return None
    return path or None


def get_verify() -> bool:
    """Resolve whether TLS verification is enabled. Admin store wins.

    Default ``True`` (verify). Explicitly setting ``ssl.verify=false`` (or
    ``SSL_VERIFY=false``) disables verification. A non-existent CA bundle
    path does NOT disable verification — it falls back to the default trust
    store and verification stays on.
    """
    stored = _setting(_SSL_VERIFY_KEY)
    if stored is not None:
        return _to_bool(stored, default=True)
    env_val = os.environ.get("SSL_VERIFY")
    if env_val is not None:
        return _to_bool(env_val, default=True)
    return True


def requests_verify() -> Union[bool, str]:
    """Value to pass as ``verify=`` to ``requests`` calls.

    Returns the CA bundle path (str) when set + verification on, ``False``
    when verification is disabled, or ``True`` (default trust store).
    """
    if not get_verify():
        return False
    ca = get_ca_bundle()
    return ca if ca else True


def httpx_verify() -> Union[bool, str]:
    """Value to pass as ``verify=`` to ``httpx.Client`` / ``AsyncClient``.

    Same semantics as :func:`requests_verify`; httpx accepts a path string
    or a bool.
    """
    return requests_verify()


def apply_litellm_ssl() -> None:
    """Propagate the skip-verify / CA-bundle state onto litellm.

    litellm (used by cognee's cognify structured-output path) builds its OWN
    httpx/openai-SDK client (``OpenAIChatCompletion._get_async_http_client``),
    NOT the adalflow ``OpenAIClient`` that receives ``verify=`` explicitly. To
    honor skip-verify / a corporate CA bundle, litellm reads
    ``os.getenv("SSL_VERIFY", litellm.ssl_verify)`` in ``get_ssl_verify``
    (``litellm/llms/custom_httpx/http_handler.py``) and, when verification is
    on, falls back to ``os.getenv("SSL_CERT_FILE")`` for the CA bundle.

    Gap this closes: ``apply_ssl_env`` previously set neither ``SSL_VERIFY``
    nor ``litellm.ssl_verify`` in skip-verify mode, so the cognify/litellm
    path kept verifying the corporate TLS cert and failed with
    ``OpenAIException - Connection error`` while adalflow docgen (explicit
    ``verify=False``) and the admin ``requests``-based model test worked.

    litellm also CACHES its openai client (``set_cached_openai_client``), so
    this MUST run before the first litellm call. It is called from
    :func:`apply_ssl_env` (which runs at ``cognee_manager`` import + in
    ``main.py``) and from :func:`apply_cognee_ssl_patch`. Best-effort and
    never raises: litellm may be absent in a dev env without cognee.
    """
    try:
        import litellm  # type: ignore
    except Exception:  # pragma: no cover - litellm optional (cognee dep)
        return
    try:
        if not get_verify():
            # Skip-verify: litellm reads SSL_VERIFY env first, then
            # litellm.ssl_verify. Set BOTH so a cached client built between
            # calls still sees the disabled state.
            os.environ.setdefault("SSL_VERIFY", "false")
            litellm.ssl_verify = False
            return
        # Verify ON: restore litellm default and let SSL_CERT_FILE (set in
        # apply_ssl_env) supply the corporate CA bundle for the openai route.
        litellm.ssl_verify = True
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("apply_litellm_ssl failed (non-fatal): %s", e)


def apply_ssl_env() -> None:
    """Push the CA bundle path into env vars honored by httpx/cognee/requests.

    Call this as early as possible (before httpx/cognee/openai clients are
    constructed) so the default-trust-store consumers pick up the corporate
    root cert. Safe to call repeatedly; never raises.

    Sets ``SSL_CERT_FILE`` (honored by ``ssl.create_default_context`` ->
    httpx + cognee aiohttp ``create_secure_ssl_context``),
    ``REQUESTS_CA_BUNDLE`` + ``CURL_CA_BUNDLE`` (requests), but only when an
    admin/env CA bundle is configured AND verification is enabled. Does NOT
    unset pre-existing env values from the process environment.
    """
    try:
        if not get_verify():
            # Skip-verify mode: do not force a CA bundle on the environment.
            # Per-call clients (requests/httpx) get verify=False directly.
            # cognee's aiohttp context is patched separately.
            #
            # Propagate skip-verify to litellm too: its cognify
            # structured-output path builds its OWN httpx/openai client (not
            # the adalflow client that takes verify= explicitly) and reads
            # SSL_VERIFY / litellm.ssl_verify. Without this the cognify path
            # keeps verifying the corporate cert -> "Connection error".
            apply_litellm_ssl()
            #
            # Suppress urllib3's per-request InsecureRequestWarning spam: with
            # skip-verify ON, every unverified HTTPS call (Confluence/MCP/git /
            # corporate gateway) logs a WARNING that floods the logs. The user
            # has explicitly opted into skip-verify via the admin panel/env, so
            # these warnings add no signal. Filtering here (at the first SSL
            # bootstrap) keeps the rest of the process quiet for the lifetime
            # of the run.
            try:
                import warnings as _warnings
                from urllib3.exceptions import InsecureRequestWarning  # type: ignore
                _warnings.filterwarnings(
                    "ignore", category=InsecureRequestWarning
                )
                logger.info(
                    "SSL skip-verify active; urllib3 InsecureRequestWarning "
                    "suppressed (unverified HTTPS requests will not be logged)."
                )
            except Exception as _we:  # pragma: no cover - urllib3 optional
                logger.debug("could not suppress InsecureRequestWarning: %s", _we)
            return
        ca = get_ca_bundle()
        if ca:
            # Don't overwrite an explicitly-exported env value with the same
            # resolved path, but DO set the keys so all consumers honor it.
            os.environ.setdefault("SSL_CERT_FILE", ca)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
            os.environ.setdefault("CURL_CA_BUNDLE", ca)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("apply_ssl_env failed (non-fatal): %s", e)


def apply_cognee_ssl_patch() -> None:
    """Patch cognee's aiohttp SSL context for skip-verify / CA bundle.

    cognee's ``cognee.shared.utils.create_secure_ssl_context`` returns
    ``ssl.create_default_context()``. With an admin CA bundle we already
    export ``SSL_CERT_FILE`` (honored by ``ssl.create_default_context``),
    so the CA case is covered by :func:`apply_ssl_env`. For the skip-verify
    case we monkeypatch the function to return an unverified context so
    cognee's Ollama/OpenAI-compatible aiohttp embedders stop failing with
    ``unable to get local issuer certificate``.

    Safe to call when cognee is unavailable (no-op). Never raises.
    """
    # Re-assert the litellm skip-verify/CA state on every call so a runtime
    # flip of ssl.verify (admin panel) propagates to litellm's cached client
    # before the next cognify call. apply_ssl_env() runs once at import; this
    # covers the case where the admin toggles ssl.verify AFTER startup.
    apply_litellm_ssl()
    try:
        import cognee.shared.utils as _cutils  # type: ignore
    except Exception:
        return  # cognee not installed / not imported yet
    try:
        if get_verify():
            # Verification ON: restore the original function if we previously
            # patched it, so a runtime flip back to verify=True is honored.
            orig = getattr(_cutils.create_secure_ssl_context, "__ssl_orig__", None)
            if orig is not None:
                _cutils.create_secure_ssl_context = orig  # type: ignore
                logger.info("cognee SSL: verification re-enabled (restored default context).")
            return

        import ssl as _ssl

        def _unverified_context() -> "ssl.SSLContext":
            ctx = _ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            return ctx

        if not hasattr(_cutils.create_secure_ssl_context, "__ssl_orig__"):
            _unverified_context.__ssl_orig__ = _cutils.create_secure_ssl_context  # type: ignore[attr-defined]
        _cutils.create_secure_ssl_context = _unverified_context  # type: ignore
        logger.warning(
            "cognee SSL: certificate verification DISABLED "
            "(ssl.verify=false). This is insecure (MITM risk)."
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("apply_cognee_ssl_patch failed (non-fatal): %s", e)
