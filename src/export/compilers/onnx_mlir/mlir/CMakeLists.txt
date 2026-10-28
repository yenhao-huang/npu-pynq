cmake_minimum_required(VERSION 3.24)
project(OnnxImportAdapter LANGUAGES C CXX)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

add_executable(onnx-import-adapter lib/Adapter.cpp)
target_compile_features(onnx-import-adapter PRIVATE cxx_std_17)
target_include_directories(onnx-import-adapter SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITION_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_options(onnx-import-adapter PRIVATE ${LLVM_DEFINITION_LIST})
target_link_libraries(onnx-import-adapter PRIVATE MLIR LLVM)
