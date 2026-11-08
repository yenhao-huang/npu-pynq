# pynqo-middle-end and pynqo-backend

The two C++ tools of the export. Both run on the host (Linux or WSL, LLVM 22);
`src/export/pipeline.py` builds and runs them.

```
torch.mlir --pynqo-middle-end--> model.mlir --pynqo-backend--> package
```

## pynqo-middle-end

Source `src/export/middle-end/`, built in `build/pynqo-middle-end/`.

```
pynqo-middle-end <torch.mlir> -o <model.mlir> [--emit=linalg|torch]
```

| | |
| --- | --- |
| Input | `ir/torch.mlir`: the torch dialect, written by `to_mlir` |
| Output | `ir/model.mlir`: linalg on tensors, weights inline as constants |
| Passes | FuseConvBatchNorm, DecomposeIntMm, AnyDimToAnyDims, then torch-mlir's two pipelines to linalg |
| `--emit=torch` | stop after our passes, to inspect them |

## pynqo-backend

Source `src/export/backend/`, built in `build/pynqo-backend/`.

```
pynqo-backend <model.mlir> --out <dir> --devices cpu,npu --cpus arm-cortex-a9,native
            [--llvm-bin /usr/lib/llvm-22/bin] [--rows 16 --columns 16 --max-k 256] [--keep-ir]
```

| | |
| --- | --- |
| Input | `ir/model.mlir` |
| Output | in `<dir>`: `weights.bin`, `programs.bin/.json/.txt` (NPU only), `<cpu>/model.so` per `--cpus`, `compile.json` |
| `--devices` | `cpu`: everything is CPU code; `cpu,npu`: INT8 matmuls on constant weights run on the NPU |
| `--cpus` | `arm-cortex-a9` for the board (cross-compiled), `native` for this host |
| `--keep-ir` | also write `partitioned.mlir` and each CPU's LLVM IR |

`pipeline.py` then adds argument names, the tokenizer and model metadata to
`compile.json` to make `manifest.json`.
