"""torch dialect -> linalg on tensors (upstream dialects only).

Runs torch-mlir's two pipelines in process, through its Python bindings. The
result holds only upstream dialects (linalg, tensor, arith, math, func, cf),
which the back end (LLVM 22) parses.
"""

from __future__ import annotations

from pathlib import Path

PIPELINE = ('builtin.module(torch-function-to-torch-backend-pipeline,'
            'torch-backend-to-linalg-on-tensors-backend-pipeline)')


class ToMlirError(RuntimeError):
    pass


def to_linalg_dialect(module, out: Path) -> Path:
    """Lower the torch-dialect ``module`` in place and write it to ``out``."""
    from torch_mlir import ir
    from torch_mlir.passmanager import PassManager

    with module.context:
        try:
            PassManager.parse(PIPELINE).run(module.operation)
        except ir.MLIRError as error:
            raise ToMlirError(f'torch -> linalg failed:\n{str(error)[-4000:]}') from error
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        module.operation.print(file=f)
    return out
