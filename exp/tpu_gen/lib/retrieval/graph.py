"""Macro-aware dependency retrieval over the TPU-Gen RTL library."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from tpugen_types import FlowError

from .preprocess import collect_defines, resolve, strip_comments

_MODULE_DECL = re.compile(r"^\s*module\s+([A-Za-z_]\w*)", re.M)


@dataclass
class RTLLibrary:
    """Every module in the .v library, indexed by module name."""

    root: Path
    module_file: dict[str, Path] = field(default_factory=dict)
    file_text: dict[Path, str] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "RTLLibrary":
        if not root.is_dir():
            raise FlowError(f"RTL library not found: {root}")
        lib = cls(root=root)
        for path in sorted(root.glob("*.v")):
            text = path.read_text(errors="replace")
            lib.file_text[path] = text
            for name in _MODULE_DECL.findall(text):
                if name in lib.module_file and lib.module_file[name] != path:
                    raise FlowError(
                        f"module {name} declared in both "
                        f"{lib.module_file[name].name} and {path.name}"
                    )
                lib.module_file[name] = path
        if not lib.module_file:
            raise FlowError(f"no Verilog modules found under {root}")
        return lib

    def instantiations(self, path: Path, defines: dict[str, str]) -> set[str]:
        """Module names instantiated in a file once macros are resolved."""
        body = resolve(self.file_text[path], defines)
        declared = set(_MODULE_DECL.findall(body))
        found: set[str] = set()
        for name in self.module_file:
            if name in declared:
                continue
            # <module> [#(params)] <instance> (
            pattern = rf"\b{re.escape(name)}\s*(?:#\s*\([^;]*?\)\s*)?[A-Za-z_]\w*\s*\("
            if re.search(pattern, body):
                found.add(name)
        return found


@dataclass
class RetrievalResult:
    filelist: list[Path]
    modules: list[str]
    defines: dict[str, str]
    missing: list[str]
    mode: str


def retrieve(
    vh_text: str,
    library: RTLLibrary,
    top: str,
    *,
    mode: str = "rag",
) -> RetrievalResult:
    """Resolve a header to the exact set of .v files the design needs.

    mode="rag"  -- dependency closure from the top module (the paper's claim)
    mode="full" -- the whole library, as a control for measuring what RAG saves
    """
    defines = collect_defines(vh_text)

    if mode == "full":
        files = sorted(set(library.module_file.values()))
        return RetrievalResult(
            filelist=files,
            modules=sorted(library.module_file),
            defines=defines,
            missing=[],
            mode="full",
        )
    if mode != "rag":
        raise FlowError(f"unknown retrieval mode {mode!r}")

    if top not in library.module_file:
        raise FlowError(f"top module {top!r} is not in {library.root}")

    seen: set[str] = set()
    missing: list[str] = []
    queue = [top]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        path = library.module_file.get(name)
        if path is None:
            missing.append(name)
            continue
        queue.extend(library.instantiations(path, defines))

    files = sorted({library.module_file[m] for m in seen if m in library.module_file})
    return RetrievalResult(
        filelist=files,
        modules=sorted(seen),
        defines=defines,
        missing=sorted(missing),
        mode="rag",
    )


def selector_macros(defines: dict[str, str]) -> list[str]:
    """Value-less macros: the ones that actually select a variant."""
    return sorted(k for k, v in defines.items() if v.strip() == "")
