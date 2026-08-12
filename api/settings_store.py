"""Backwards-compatibility shim. Real module: api.config.settings."""
from api.config import settings as _real
from api.config.settings import *  # noqa: F401, F403


def __getattr__(name):
    return getattr(_real, name)
