# Compiled ONNX SmolLM2 INT8 demo

Prepare and compile the pinned upstream ONNX model using
[the shared-source instructions](../../docs/acceptance/smollm-onnx.md), then
open [smollm_onnx.ipynb](smollm_onnx.ipynb) in the root PYNQ kernel.

Set `NPU_REPO`, `NPU_COMPILED_PACKAGE` and `NPU_OVERLAY` to the repository,
compiled package and matching ISA overlay. Assets live outside `exp/`.
The demo imports only `src/`; deleting the experiment directory is supported.
Exact chat prompt lengths need compiled prefill specializations (default 36,
40 and 42). The ARM CPU simulator must run in a separate process from physical
inference because the compiled library owns one NPU runtime state.
