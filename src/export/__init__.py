"""Host-side export: a model and a config -> a package. Never runs on the board.

    cli.py         the command line: build <config>
    pipeline.py    load -> export_graph -> quantize -> to_mlir -> pynqo-middle-end -> pynqo-backend
    frontend/      Python: load, export_graph, quantization (an FX pass), to_mlir
    middle-end/    C++ (LLVM 22): pynqo-middle-end; rewrites on the torch dialect, then linalg
    backend/       C++ (LLVM 22): pynqo-backend; partition, delegates, weights, CPU codegen

A config is one YAML file per demo (source, quantization, graph, devices), kept
with that demo: examples/<demo>/export.yml.

src/vendors sits below src/export; src/runtime never imports src/export.
"""
