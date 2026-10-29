"""Host-side export: model -> package. Needs MLIR/LLVM; never runs on the board.

    cli.py        export() (front end, then compile) and the command line
    compile.py    compile_graph(): ModelGraph -> package (partition, pack, encode, codegen)
    pipeline.py   front end: load -> quantize -> to_mlir
    ir/           data between stages: options, SourceModel, QuantizedModel, ModelGraph
    load/         stage 1: read sources without changing any number
    quantize/     stage 2: quantization numerics, weight plans, AWQ
    model_arch/   stage 3: one Architecture per model (a directory when it brings source
                  pins and downloads); registry in model_arch/__init__.py
    compilers/    target-independent MLIR builder, operators and LLVM toolchain

Imports point down: cli, compile, pipeline -> everything; model_arch -> ir,
load, quantize, compilers; quantize -> ir; load -> ir, compilers/onnx_mlir;
compilers never import quantize, load or model_arch (test_export_layering.py). src/vendors sits
below src/export; src/runtime never imports src/export.
"""
