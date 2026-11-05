# Run in torch-mlir's source directory after download. An installed LLVM may have
# been built with Python bindings, which turns them on here, and torch-mlir's
# tests depend on those modules. The middle end needs neither.
file(READ CMakeLists.txt text)
string(REPLACE "if(MLIR_ENABLE_BINDINGS_PYTHON)"
               "set(MLIR_ENABLE_BINDINGS_PYTHON OFF)\nif(MLIR_ENABLE_BINDINGS_PYTHON)" text "${text}")
string(REPLACE "add_subdirectory(test)" "" text "${text}")
string(REPLACE "add_dependencies(check-torch-mlir-all check-torch-mlir)" "" text "${text}")
file(WRITE CMakeLists.txt "${text}")
