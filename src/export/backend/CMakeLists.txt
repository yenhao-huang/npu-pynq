# pynqo-backend: the back end, built out of tree against an installed LLVM 22
# with MLIR. It links the vendor's dialect library (src/vendors/npu_pynq/
# custom_dialect); src/export/pipeline.py builds it when stale.
#
#   cmake -S src/export/backend -B build/pynqo-backend -G Ninja \
#         -DMLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir
#   cmake --build build/pynqo-backend -j 1
cmake_minimum_required(VERSION 3.20)
project(pynqo-backend LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../vendors/npu_pynq/custom_dialect
                 ${CMAKE_CURRENT_BINARY_DIR}/custom_dialect)

add_executable(pynqo-backend
  pynqo_backend.cpp
  partition.cpp
  device_register.cpp
  package.cpp
  cpu/toolchain.cpp
  delegates/npu/lower.cpp)
llvm_update_compile_flags(pynqo-backend)
target_include_directories(pynqo-backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pynqo-backend PRIVATE NpuDialect MLIR LLVM)
