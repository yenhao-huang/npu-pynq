"""Host-side export: a model and a config -> a package. Never runs on the board.

    cli.py         the command line: build <config>
    pipeline.py    load -> quantize -> export_graph -> to_mlir -> pynqo-middle-end -> pynqo-backend
    configs/       one YAML file per model: source, quantization, graph, devices
    frontend/      Python: load, quantization (torchao), export_graph, to_mlir
    middle-end/    C++ (LLVM 22): pynqo-middle-end; rewrites on the torch dialect, then linalg
    backend/       C++ (LLVM 22): pynqo-backend; partition, delegates, weights, CPU codegen

src/vendors sits below src/export; src/runtime never imports src/export.
"""
