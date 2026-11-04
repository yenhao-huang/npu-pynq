"""``to_mlir``: ExportedPrograms -> ``model.mlir`` (linalg on tensors)."""

from __future__ import annotations

from pathlib import Path

import torch

from .torch_mlir.to_linalg_dialect import ToMlirError, to_linalg_dialect
from .torch_mlir.to_torch_dialect import to_torch_dialect


def to_mlir(programs: dict[str, torch.export.ExportedProgram], workdir: Path) -> Path:
    """Write ``torch.mlir`` and ``model.mlir`` under ``workdir``; return ``model.mlir``."""
    workdir.mkdir(parents=True, exist_ok=True)
    # torch.mlir is written first, because the lowering rewrites ``module`` in place.
    # Both files use UTF-8 and LF on every host, because a Linux LLVM parses them.
    module = to_torch_dialect(programs)
    with open(workdir / 'torch.mlir', 'w', encoding='utf-8', newline='\n') as f:
        module.operation.print(file=f)
    return to_linalg_dialect(module, workdir / 'model.mlir')


__all__ = ['ToMlirError', 'to_mlir']
