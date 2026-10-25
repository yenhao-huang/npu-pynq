"""Host-side export: model -> package. Needs MLIR/LLVM; never runs on the board.

    cli.py      export() pipeline and the command line
    core/       model loading, per-layer weight quantization, ModelGraph, ONNX sources
    quant/      quantization numerics (int8.py) and tools (AWQ, fake-quant)
    graphs/     one frontend per model; registry in graphs/__init__.py
    compilers/  target-independent MLIR builder, operators and LLVM toolchain
    vendors/    per-target passes, weight packing and CPU target; registry in vendors/__init__.py

Imports point down: cli -> everything; graphs -> core, quant, compilers;
vendors -> core, compilers; core and compilers never import graphs or vendors.
src/runtime never imports src/export.
"""
