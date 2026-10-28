# SmolLM2 INT8 ONNX from Shared Source

The pinned ONNX model exports through ONNX-MLIR import, export-side CPU
conversions, and the existing NPU partition/lower pipeline. The final package
contains ARM/host libraries, original INT8 weights, NPU programs and tokenizer
assets. It runs on PYNQ-Z1 without an ORT library or C operator bridge.

Use the [WSL deployment guide](../../../examples/smollm_onnx/README.md) or
[macOS deployment guide](../../../examples/smollm_onnx/README.macos.md).
The canonical compiler instructions, plan and acceptance evidence are under
[docs/plans/onnx-mlir-incoporate](../../plans/onnx-mlir-incoporate/README.md).

## Source ownership

| Path | Responsibility |
| --- | --- |
| `src/export/compilers/onnx_mlir/` | Pinned importer, generic-IR parser, identity checks and verified CPU conversions |
| `src/export/graphs/smollm/{onnx,model}.py` | Model specialization, fused projections, prefill/decode and KV ABI |
| `src/export/compilers/mlir/` | Shared tensor builder, CPU operations and LLVM 22 code generation |
| `src/vendors/npu_pynq/compiler/` | Unchanged NPU partition, ISA programs and tiled weights |
| `src/runtime/core/{package,engine,tokenizer}.py` | Loading, inference, sampling and tokenization |
| `src/vendors/npu_pynq/driver/` | Existing physical memory and NPU execution |
| `src/export/core/onnx_source.py` | Downloads and independent reference assets |
| `src/test/onnx_mlir_acceptance.py` | Full-source ORT fixtures and host/physical acceptance |
| `src/runtime/cpu/ort.py` | Optional Python ORT operators for host references |

## Export

```bash
export NPU_ONNX_MODEL="$PWD/dist/smollm2-onnx"
python -m src.export.cli prepare-onnx fetch
python -m src.export.cli export smollm \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --out dist/smollm-package --prefill-lengths 36,40,42
```

An LLVM/MLIR 22 environment and the pinned ONNX-MLIR Docker image are required;
see the compiler guide. Import happens before partition. Compiled ARM code
dispatches ISA programs at runtime. The board does not import ONNX or MLIR.

## Independent reference

Optional host graph interpretation remains separate from compiled inference:

```bash
python -m pip install onnxruntime==1.20.1
python -m src.export.cli prepare-onnx prepare
python -m src.export.cli prepare-onnx operators
python -m src.runtime.cli reference --backend cpu \
  --prompt "Explain why the sky is blue." --max-new-tokens 24
```

The board's optional graph reference uses NumPy floating operators and NPU
integer contractions. Current production preparation and packages do not need
these reference operator sessions.

## Verified behavior

Physical PYNQ-Z1 acceptance passed all three prefill lengths and three decode
steps each. KV state and greedy IDs were exact; max logit error was `4.2e-5`.
The output-free [notebook](../../../examples/smollm_onnx/smollm_onnx.ipynb)
generated 32 tokens matching the independent unchanged-source reference.
See [the acceptance record](../../plans/onnx-mlir-incoporate/acceptance.md).

The four production ORT C/build files and unused ORT-only experiment wrappers
were removed after validation. Historical results under `exp/` describe their
original revisions; current public examples and acceptance use shared `src/`.
No NPU vendor, driver, ISA or RTL modifications were needed.
