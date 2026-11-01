# Optional just-in-time stack

This package owns the pre-MLIR quantized graph exporter and runtime, the original
ResNet18 notebook under `examples/resnet18`, its tests under `tests`, and the
independent integer reference under `reference`. DMA and overlay support are
frozen locally so the package imports no active export/runtime/test modules.

Run from the repository root:

```bash
python -m unittest discover -s src/jit/tests -v
python -m unittest discover -s src/jit/examples/resnet18/tests -v
```

The active `src/export`, `src/runtime`, `src/vendors/npu_pynq/spec`, `src/test`, public examples
and delivery workflows do not depend on this package. It can be removed in its
entirety without changing the AOT software stack. Historical records may mention
old paths; they are not rewritten as executable dependencies.
