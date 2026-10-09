# Deprecated software stack

This package retains the pre-MLIR model contracts, ResNet export pipeline, and
model runtime. New model export uses `src.compiler` and `src.inference`; compiled
packages run through `src.runtime.compiled` and `src.runtime.c.npu_rt`.

Existing examples and release tooling import the archived modules directly.
Do not add new features here. `src.runtime.npu` and `verify_overlay` remain in
the active runtime because the board ISA check and delivery tools still use them.
