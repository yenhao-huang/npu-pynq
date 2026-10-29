"""``to_mlir``: ExportedPrograms -> ``model.mlir`` (linalg on tensors)."""

from __future__ import annotations

from pathlib import Path

import torch

from .iree.to_linalg_dialect import ToMlirError, to_linalg_dialect
from .iree.to_torch_dialect import to_torch_dialect


def to_mlir(programs: dict[str, torch.export.ExportedProgram], workdir: Path) -> Path:
    """Write ``torch.mlir`` and ``model.mlir`` under ``workdir``; return ``model.mlir``."""
    workdir.mkdir(parents=True, exist_ok=True)
    torch_mlir = workdir / 'torch.mlir'
    torch_mlir.write_text(to_torch_dialect(programs))
    return to_linalg_dialect(torch_mlir, workdir / 'model.mlir')


__all__ = ['ToMlirError', 'to_mlir']
