# pynqo-middle-end and pynqo-backend

The two C++ tools of the export. Both run on the host (Linux or WSL, LLVM 22);
`src/export/pipeline.py` builds and runs them.

```
torch.mlir --pynqo-middle-end--> model.mlir + delegates/<device>.mlir --pynqo-backend--> package
```

## pynqo-middle-end

Source `src/export/middle-end/`, built in `build/pynqo-middle-end/`.

```
pynqo-middle-end <torch.mlir> -o <model.mlir> [--emit=linalg|torch] [--devices cpu,npu]
                 [--delegates-dir <dir>]
```

| | |
| --- | --- |
| Input | `ir/torch.mlir`: the torch dialect, written by `to_mlir` |
| Output | `ir/model.mlir`: the CPU graph, linalg on tensors, weights inline as constants, a call to a bodiless `@__delegate_<id>` per device subgraph; `ir/delegates/<device>.mlir`: that device's subgraphs, one `@<device>_<id>` each |
| Passes | FuseConvBatchNorm, DecomposeIntMm, AnyDimToAnyDims, torch-mlir's two pipelines to linalg, pynqo-assign-devices, pynqo-partition-devices, then canonicalize, CSE and elementwise fusion on the CPU graph |
| `--devices` | `cpu` (default): nothing is assigned; `cpu,npu`: INT8 matmuls on constant weights with a zero accumulator go to the NPU |
| `--delegates-dir` | where the device modules go (default `delegates/` next to `-o`) |
| `--emit=torch` | stop after our passes, to inspect them |

## pynqo-backend

Source `src/export/backend/`, built in `build/pynqo-backend/`.

```
pynqo-backend <model.mlir> --out <dir> --devices cpu,npu --cpus arm-cortex-a9,native
            [--delegates-dir <dir>] [--llvm-bin /usr/lib/llvm-22/bin]
            [--rows 16 --columns 16 --max-k 256] [--keep-ir]
```

| | |
| --- | --- |
| Input | `ir/model.mlir` and, for each device it calls, `<delegates-dir>/<device>.mlir` (default `delegates/` next to the model) |
| Output | in `<dir>`: `weights.bin`, `programs.bin/.json/.txt` (NPU only), `<cpu>/model.so` per `--cpus`, `compile.json` |
| `--devices` | must equal the middle end's `--devices`; each delegate compiles its device's module |
| `--cpus` | `arm-cortex-a9` for the board (cross-compiled), `native` for this host |
| `--keep-ir` | also write `cpu.mlir` (the CPU graph with `__delegate__` calls and externalized weights) and each CPU's LLVM IR |

`pipeline.py` then adds argument names, the tokenizer and model metadata to
`compile.json` to make `manifest.json`.
