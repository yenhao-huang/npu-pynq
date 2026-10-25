# npu-opt: the NPU dialect and its partition/lower passes, built out of tree
# against an installed MLIR (LLVM 22).
#
#   cmake -S src/compiler/mlir -B build/npu-opt -DMLIR_DIR=<llvm>/lib/cmake/mlir
#   cmake --build build/npu-opt
cmake_minimum_required(VERSION 3.20)
project(npu-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/Npu)
set(LLVM_TARGET_DEFINITIONS include/Npu/NpuOps.td)
mlir_tablegen(include/Npu/NpuOps.h.inc -gen-op-decls)
mlir_tablegen(include/Npu/NpuOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/Npu/NpuOpsDialect.h.inc -gen-dialect-decls -dialect=npu)
mlir_tablegen(include/Npu/NpuOpsDialect.cpp.inc -gen-dialect-defs -dialect=npu)
add_public_tablegen_target(NpuOpsIncGen)

add_executable(npu-opt
  tools/npu-opt.cpp
  lib/NpuDialect.cpp
  lib/NpuEncoder.cpp
  lib/Partition.cpp
  lib/Lower.cpp)
add_dependencies(npu-opt NpuOpsIncGen)
llvm_update_compile_flags(npu-opt)
target_link_libraries(npu-opt PRIVATE
  MLIROptLib MLIRIR MLIRParser MLIRPass MLIRSupport MLIRTransforms
  MLIRArithDialect MLIRBufferizationDialect MLIRFuncDialect
  MLIRLinalgDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect
  MLIRTensorDialect)
