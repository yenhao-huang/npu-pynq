"""The registry: one source of truth for every agent-facing surface.

A *category* is a folder under `ic_core/tools/`. It owns the schema -- the
pydantic models for its operations -- and nothing else. A *backend* is one file
inside that folder that knows how to drive one external program and translate
its output into the category's `Out` model.

Everything else in this repository is generated from what is registered here:

| Surface            | Derived from                                  |
| ------------------ | --------------------------------------------- |
| MCP tool list      | `Op.In` / `Op.Out` JSON Schema                 |
| CLI subcommands    | the same models, reflected into arguments      |
| HTTP endpoints     | the same models, handed to FastAPI             |
| `meta.json` inputs | `Op.In` fields + backend name and version      |
| sync vs job path   | `Op.long_running`                              |

Adding a backend touches one file. Adding a category touches one folder.
Neither touches the daemon, the CLI, or the MCP server.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Type

from pydantic import BaseModel

from .errors import BackendUnavailable, UnknownBackend, UnknownOp


@dataclass(frozen=True)
class Op:
    """One agent-callable operation belonging to a category.

    `name` is globally unique, so it doubles as the MCP tool name and the CLI
    subcommand. Most categories have exactly one op named after the category;
    `debug` has four because an agent queries a waveform in four different ways.
    """

    name: str
    In: Type[BaseModel]
    Out: Type[BaseModel]
    summary: str
    long_running: bool = False
    #: Record a run directory on disk. Off for cheap read-only queries that
    #: would otherwise litter `.ic/runs/` with one folder per waveform probe.
    records_run: bool = True


@dataclass
class Category:
    name: str
    summary: str
    ops: list[Op]
    default_backend: str
    backends: dict[str, "Backend"] = field(default_factory=dict)

    def op(self, name: str) -> Op:
        for candidate in self.ops:
            if candidate.name == name:
                return candidate
        raise UnknownOp(f"category {self.name!r} has no op {name!r}")


@dataclass
class Backend:
    name: str
    category: str
    impl: Callable[[], object]
    #: Command that prints the backend's version, recorded into `meta.json`.
    version_cmd: list[str] | None = None
    #: Executable that must exist on PATH for this backend to be usable.
    requires: str | None = None

    _version_cache: str | None = field(default=None, repr=False)

    def available(self) -> bool:
        return self.requires is None or shutil.which(self.requires) is not None

    def version(self) -> str:
        if self._version_cache is not None:
            return self._version_cache
        version = "unknown"
        if self.version_cmd:
            try:
                proc = subprocess.run(
                    self.version_cmd, capture_output=True, text=True, timeout=30
                )
                text = (proc.stdout or proc.stderr).strip().splitlines()
                if text:
                    version = text[0].strip()
            except (OSError, subprocess.SubprocessError):
                version = "unknown"
        self._version_cache = version
        return version


CATEGORIES: dict[str, Category] = {}


def register_category(category: Category) -> Category:
    CATEGORIES[category.name] = category
    return category


def backend(category: str, name: str, *, requires: str | None = None,
            version_cmd: list[str] | None = None) -> Callable:
    """Class decorator that plugs a backend into its category.

    The class implements one method per op it supports, named after the op.
    Ops it does not implement fall through to `UnknownBackend` at call time,
    which is how `vcd` can serve every `debug` op while a future partial
    backend serves only some.
    """

    def wrap(cls):
        entry = Backend(
            name=name,
            category=category,
            impl=cls,
            version_cmd=version_cmd,
            requires=requires,
        )
        CATEGORIES[category].backends[name] = entry
        cls.__ic_backend__ = entry
        return cls

    return wrap


def load_all() -> None:
    """Import every category package so the decorators above have run."""
    from . import tools  # noqa: F401

    tools.load()


def get_category(name: str) -> Category:
    load_all()
    if name not in CATEGORIES:
        raise UnknownOp(f"unknown category {name!r}", known=sorted(CATEGORIES))
    return CATEGORIES[name]


def find_op(op_name: str) -> tuple[Category, Op]:
    """Resolve a globally unique op name to its category."""
    load_all()
    for category in CATEGORIES.values():
        for op in category.ops:
            if op.name == op_name:
                return category, op
    raise UnknownOp(f"unknown op {op_name!r}", known=sorted(all_op_names()))


def all_op_names() -> list[str]:
    load_all()
    return [op.name for c in CATEGORIES.values() for op in c.ops]


def iter_ops():
    load_all()
    for category in CATEGORIES.values():
        for op in category.ops:
            yield category, op


def resolve_backend(category: Category, name: str | None) -> Backend:
    chosen = name or category.default_backend
    if chosen not in category.backends:
        raise UnknownBackend(
            f"{category.name} has no backend {chosen!r}",
            known=sorted(category.backends),
        )
    entry = category.backends[chosen]
    if not entry.available():
        raise BackendUnavailable(
            f"{category.name} backend {chosen!r} needs {entry.requires!r} on PATH",
            backend=chosen,
            requires=entry.requires,
        )
    return entry
