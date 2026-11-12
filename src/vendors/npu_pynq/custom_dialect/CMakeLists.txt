# Two static libraries (LLVM 22 / MLIR), added with add_subdirectory() by a
# tool that has already found MLIR:
#
#   NpuMatch    npu::isClaimable, the NPU's claim rule. Upstream linalg and
#               arith headers only, and no link libraries: each consumer
#               supplies MLIR in its own form. pynqo-middle-end links it with
#               static MLIR components, pynqo-backend (through NpuDialect)
#               with the monolithic MLIR library. Never link MLIR here; two
#               link styles in one binary register everything twice.
#   NpuDialect  the npu dialect, npu::claim and --npu-claim, and the npu
#               delegate's Pack and Encode steps; linked by pynqo-backend only.
#               pynqo-middle-end adds this folder EXCLUDE_FROM_ALL and never
#               builds it.

add_library(NpuMatch STATIC lib/Match.cpp)
target_include_directories(NpuMatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(NpuMatch PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
llvm_update_compile_flags(NpuMatch)

set(LLVM_TARGET_DEFINITIONS include/Npu/NpuOps.td)
mlir_tablegen(include/Npu/NpuOps.h.inc -gen-op-decls)
mlir_tablegen(include/Npu/NpuOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/Npu/NpuOpsDialect.h.inc -gen-dialect-decls -dialect=npu)
mlir_tablegen(include/Npu/NpuOpsDialect.cpp.inc -gen-dialect-defs -dialect=npu)
add_public_tablegen_target(NpuOpsIncGen)

add_library(NpuDialect STATIC
  lib/NpuDialect.cpp
  lib/NpuEncoder.cpp
  lib/Claim.cpp
  lib/Pack.cpp
  lib/Encode.cpp)
add_dependencies(NpuDialect NpuOpsIncGen)
target_include_directories(NpuDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)
llvm_update_compile_flags(NpuDialect)
target_link_libraries(NpuDialect PUBLIC NpuMatch MLIR)
