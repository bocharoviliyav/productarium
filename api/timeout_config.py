"""Backwards-compatibility shim. Real module: api.config.timeout."""
from api.config import timeout as _real
from api.config.timeout import *  # noqa: F401, F403


def __getattr__(name):
    return getattr(_real, name)
