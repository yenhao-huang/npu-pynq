# torch-mlir: ExportedProgram -> torch dialect -> linalg

The ExportedProgram is given (`1_exported_program.pt2`, readable as
`1_exported_program.txt`): a one-layer W8A8 model, `relu(Linear(8 -> 4))`.
Two torch-mlir steps take it to linalg, the form `pynqo-backend` reads.

| Step | File | Level | Tool |
|---|---|---|---|
| given | `1_exported_program.pt2`, `.txt` | ExportedProgram | `torch.export` |
| 1 | `2_torch_dialect.mlir` | torch dialect | `torch_mlir.fx` (FxImporter) |
| 2 | `3_linalg.mlir` | linalg on tensors | `torch-mlir-opt` |

## Setup (WSL, with the worktree's Windows venv)

The venv holds Windows programs, so call them by path with `.exe`; activating
it in bash does not put them on `PATH`.

```bash
cd /mnt/c/Users/User/Desktop/agent_workspace/npu/worktrees/npu-issue119-a/docs/manual/torch_mlir_arch
V=../../../.venv/Scripts
```

## 1. ExportedProgram -> torch dialect (to_torch_dialect)

torch-mlir has no command-line importer for an ExportedProgram; its Python
`fx.export_and_import` is the tool.

```bash
$V/python.exe - <<'EOF'
import torch
from torch_mlir import fx
ep = torch.export.load('1_exported_program.pt2')
module = fx.export_and_import(ep, output_type='raw', decomposition_table={}, func_name='forward')
open('2_torch_dialect.mlir', 'w', newline='\n').write(str(module))
EOF
```

## 2. torch dialect -> linalg (to_linalg_dialect)

```bash
$V/torch-mlir-opt.exe 2_torch_dialect.mlir \
  --pass-pipeline='builtin.module(torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline)' \
  -o 3_linalg.mlir
```

`-o` takes a path relative to this directory or under `/mnt/c/`; the Windows
program cannot see WSL paths such as `/tmp`.

## Check

```bash
grep -n "aten._int_mm\|linalg.matmul" 1_exported_program.txt 2_torch_dialect.mlir 3_linalg.mlir
```

```text
1_exported_program.txt:12:    _int_mm = torch.ops.aten._int_mm.default(_to_copy, b_fc_qweight);  ...
2_torch_dialect.mlir:34:    %11 = torch.aten._int_mm %9, %10 : !torch.vtensor<[2,8],si8>, !torch.vtensor<[8,4],si8> -> !torch.vtensor<[2,4],si32>
3_linalg.mlir:77:    %16 = linalg.matmul ins(%13, %cst_6 : tensor<2x8xi8>, tensor<8x4xi8>) outs(%15 : tensor<2x4xi32>) -> tensor<2x4xi32>
```

Recorded with torch 2.11.0 and torch-mlir 20260930.602 on 2026-11-01.
