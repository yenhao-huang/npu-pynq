"""Category packages.

Importing a category package runs its `register_category` call and then
imports its backends, whose decorators attach them. `load` is idempotent and
is called by the registry before any lookup, so no surface has to know the
list of categories -- adding a folder here is the whole integration step.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

_loaded = False

#: Backends are optional by design: a machine without Vivado should still get
#: working lint and sim rather than an import error at startup.
_OPTIONAL_IMPORT_ERRORS: list[str] = []


def load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    here = Path(__file__).parent
    for entry in sorted(pkgutil.iter_modules([str(here)])):
        if not entry.ispkg:
            continue
        try:
            importlib.import_module(f"{__name__}.{entry.name}")
        except Exception as exc:  # pragma: no cover - surfaced via `ic doctor`
            _OPTIONAL_IMPORT_ERRORS.append(f"{entry.name}: {exc}")


def import_errors() -> list[str]:
    load()
    return list(_OPTIONAL_IMPORT_ERRORS)
