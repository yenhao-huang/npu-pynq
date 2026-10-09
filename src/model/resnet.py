"""Compatibility alias for src.deprecate.model.resnet."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("src.deprecate.model.resnet")
