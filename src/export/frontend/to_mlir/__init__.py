"""``to_mlir``: ExportedPrograms -> ``torch.mlir`` (the torch dialect)."""

from __future__ import annotations

from pathlib import Path

import torch

from .torch_mlir.to_torch_dialect import to_torch_dialect


def to_mlir(programs: dict[str, torch.export.ExportedProgram], workdir: Path) -> Path:
    """Write ``torch.mlir`` under ``workdir`` and return its path."""
    workdir.mkdir(parents=True, exist_ok=True)
    # UTF-8 and LF on every host, because a Linux LLVM parses this file.
    path = workdir / 'torch.mlir'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        to_torch_dialect(programs).operation.print(file=f)
    return path


__all__ = ['to_mlir']
