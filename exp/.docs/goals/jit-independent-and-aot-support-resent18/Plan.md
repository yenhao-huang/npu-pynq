# Independent JIT and ResNet18 AOT

Issue: #119, branch `npu/issue119-a`. Continue the existing working tree without
resetting its staged or unstaged software-stack refactor.

## Design

- Move the precompiled-model interpreter, its ResNet example, references and
  focused tests from `src/deprecate` into a self-contained `src/jit` package.
- Keep the hardware numeric reference available under `src/test/model` without
  importing JIT. No active source, example, CI entry point or test may need JIT.
- Use the existing ResNet MLIR frontend and `src/runtime/core/Package` for the
  public AOT example. Board inference loads previously compiled libraries; it
  does not quantize weights or lower convolution at inference time.
- Preserve the notebook's six stages: connect, overlay, choose image,
  preprocess/preview, inference, prediction. Record elapsed wall time by stage.
- Separate reusable image inference from the notebook and command-line demo.

## Acceptance and experiment loop

1. Hypothesis: moving all interpreter consumers isolates JIT. Audit imports in
   both directions and execute active tests with the JIT directory absent.
2. Hypothesis: the ResNet frontend compiles and runs independently. Exercise
   graph construction, compilation and inference against a host reference;
   distinguish compiler, simulated-NPU and physical-board evidence.
3. Execute notebook stages against a real photograph, record individual block
   durations and total, preserving all failures and limitations.
4. Record commands and outcomes in `Result.md` and `Reproduce.md` here.
5. Physical PYNQ-Z1 execution is mandatory (added by the user): run the compiled
   image model, record a correct real-image prediction, NPU counters, each
   notebook block duration and their total. Host simulation cannot close this goal.

Generated models and compiler products remain ignored. Historical experiment
logs and human-owned documents are not rewritten.
