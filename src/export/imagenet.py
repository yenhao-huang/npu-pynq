"""Compatibility alias for src.deprecate.export.imagenet."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("src.deprecate.export.imagenet")
