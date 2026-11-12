# pynqo-backend: the back end, built out of tree against an installed LLVM 22
# with MLIR. It links each vendor's dialect library (src/vendors/npu_pynq/
# custom_dialect, src/vendors/risc_v_npu/custom_dialect); src/export/
# pipeline.py builds it when stale. The risc_v_npu delegate compiles RV32IMC
# in process and needs LLVM's RISCV target; without it the back end still
# builds, and --devices cpu,risc_v_npu reports the missing target.
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
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../vendors/risc_v_npu/custom_dialect
                 ${CMAKE_CURRENT_BINARY_DIR}/risc_v_npu_custom_dialect)

add_executable(pynqo-backend
  pynqo_backend.cpp
  partition.cpp
  device_register.cpp
  package.cpp
  cpu/toolchain.cpp
  delegates/npu/lower.cpp
  delegates/risc_v_npu/lower.cpp
  delegates/risc_v_npu/rv_codegen.cpp)
llvm_update_compile_flags(pynqo-backend)
target_include_directories(pynqo-backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pynqo-backend PRIVATE NpuDialect RvNpuDialect MLIR LLVM)

# LLVM's RISCV target is part of the LLVM shared library when it was built
# (the apt.llvm.org packages build every target).
if("RISCV" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(pynqo-backend PRIVATE PYNQO_HAVE_RISCV)
  message(STATUS "pynqo-backend: risc_v_npu delegate enabled (LLVM RISCV target found)")
else()
  message(WARNING "pynqo-backend: LLVM was built without the RISCV target; "
                  "the risc_v_npu delegate is disabled")
endif()
