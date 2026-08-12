"""Backwards-compatibility shim. Real module: api.config.abstraction."""
from api.config import abstraction as _real
from api.config.abstraction import *  # noqa: F401, F403


def __getattr__(name):
    return getattr(_real, name)
