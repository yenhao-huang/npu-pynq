# Isolate JIT and run the ResNet18 image example ahead of time

## Why

Issue #119's active export/runtime architecture is ahead of time, but the public
ResNet18 demo still imports the pre-MLIR interpreter. Removing that interpreter
therefore breaks public examples and the hardware numeric tests.

## What changes

- Move the interpreter and every executable consumer into optional `src/jit`.
- Keep JIT self-contained, including its DMA/overlay interfaces and references.
- Keep public ResNet18 on the existing MLIR frontend and compiled Package ABI.
- Provide reusable image preprocessing, prediction, and top-5 decoding in runtime.
- Preserve the six notebook interaction stages and measure their elapsed time.
- Compile host and ARM libraries in CD and accept a real image on PYNQ.

The JIT and AOT quantization schemes differ. No bitwise agreement between those
schemes is implied; compare AOT against its independent W8A8 NumPy reference.
Existing dirty worktree changes are retained. No RTL or numeric contract change
is required. The real board must execute the AOT image example before closure.
