"""Backwards-compatibility shim. Real module: api.config.ssl."""
from api.config import ssl as _real
from api.config.ssl import *  # noqa: F401, F403


def __getattr__(name):
    return getattr(_real, name)
