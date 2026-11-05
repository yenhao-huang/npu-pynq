# npu-middle-end: torch dialect -> linalg on tensors, against an installed LLVM 22
# with MLIR. It builds torch-mlir from source in this tree, without building
# LLVM, because torch-mlir's wheels bind a different LLVM and hold no C++ API;
# src/export/pipeline.py builds it when stale.
#
# To build by hand, use the same options as pipeline.py:
#
#   cmake -S src/export/middle-end -B build/npu-middle-end -G Ninja \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_C_COMPILER=clang-22 -DCMAKE_CXX_COMPILER=clang++-22 \
#         -DMLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir
#   cmake --build build/npu-middle-end --target npu-middle-end -j 1
#
# Keep the job count low: with GCC, one torch-mlir file took 5.8 GB, so Ninja's
# default job count can exhaust a 10 GB WSL.
cmake_minimum_required(VERSION 3.20)
project(npu-middle-end LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# The torch-mlir commit whose LLVM is closest to the 22.x branch. A newer one
# needs a newer LLVM; what it adds that our models need is in torch/transforms/.
set(TORCH_MLIR_COMMIT ac33bab4f3c1071d12b6c4c2f26bbcf87d1d2ab9)
set(TORCH_MLIR_OUT_OF_TREE_BUILD ON CACHE BOOL "" FORCE)
set(TORCH_MLIR_ENABLE_STABLEHLO OFF CACHE BOOL "" FORCE)
set(TORCH_MLIR_ENABLE_TOSA OFF CACHE BOOL "" FORCE)
set(TORCH_MLIR_ENABLE_REFBACKEND OFF CACHE BOOL "" FORCE)
set(TORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(torch_mlir
  URL https://github.com/llvm/torch-mlir/archive/${TORCH_MLIR_COMMIT}.tar.gz
  URL_HASH SHA256=6e8d1c1bf57ec45a1345cf42ebd6f7dfa8c3652e753a7c1bf4520960eb59d506
  PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/patch_torch_mlir.cmake)
FetchContent_MakeAvailable(torch_mlir)

add_llvm_executable(npu-middle-end
  npu_middle_end.cpp
  torch/transforms/Constants.cpp
  torch/transforms/AnyDimToAnyDims.cpp
  torch/transforms/DecomposeIntMm.cpp
  torch/transforms/FuseConvBatchNorm.cpp)
llvm_update_compile_flags(npu-middle-end)
target_include_directories(npu-middle-end PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${torch_mlir_SOURCE_DIR}/include
  ${torch_mlir_BINARY_DIR}/include)
target_link_libraries(npu-middle-end PRIVATE
  MLIRParser
  MLIRPass
  MLIRTransforms
  TorchMLIRInitAll
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses)
