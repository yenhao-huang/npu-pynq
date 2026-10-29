"""Host-side export: model -> package. Needs MLIR/LLVM; never runs on the board.

    cli.py        export(), compile_graph() and the command line
    pipeline.py   front end: load -> quantize -> to_mlir
    ir/           data between stages: options, SourceModel, QuantizedModel, ModelGraph
    load/         stage 1: read sources without changing any number
    quantize/     stage 2: quantization numerics, weight plans, AWQ, fake-quant
    model_arch/   stage 3: one Architecture per model; registry in model_arch/__init__.py
    compilers/    target-independent MLIR builder, operators and LLVM toolchain
    sources/      pinned model sources and the scripts that download them

Imports point down: cli, pipeline -> everything; model_arch -> ir, load,
quantize, compilers; quantize -> ir; load -> ir, compilers/onnx_mlir; compilers never import
quantize, load or model_arch (test_export_layering.py). src/vendors sits
below src/export; src/runtime never imports src/export.
"""
