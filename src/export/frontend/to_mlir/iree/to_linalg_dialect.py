"""torch dialect -> linalg on tensors (upstream dialects only).

Runs torch-mlir's two pipelines with the ``iree-opt`` of the IREE compiler
wheel. The result holds only upstream dialects (linalg, tensor, arith, math,
func, cf), which the back end (LLVM 22) parses.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

PIPELINE = ('builtin.module(torch-function-to-torch-backend-pipeline,'
            'torch-backend-to-linalg-on-tensors-backend-pipeline)')
UPSTREAM = ('linalg', 'tensor', 'arith', 'math', 'func', 'cf', 'builtin', 'scf', 'affine')


class ToMlirError(RuntimeError):
    pass


def iree_opt() -> str:
    import iree.compiler.tools as tools

    return str(tools.binaries.find_tool('iree-opt'))


def to_linalg_dialect(torch_mlir: Path, out: Path) -> Path:
    proc = subprocess.run([iree_opt(), str(torch_mlir), f'--pass-pipeline={PIPELINE}', '-o', str(out)],
                          capture_output=True, text=True)
    if proc.returncode:
        raise ToMlirError(f'torch -> linalg failed:\n{proc.stderr[-4000:]}')
    return out
