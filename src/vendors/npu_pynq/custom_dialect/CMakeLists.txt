# The npu dialect, --npu-claim, and the npu delegate's Pack and Encode steps,
# as a static library the back end's pynqo-backend links (LLVM 22 / MLIR).
# Added with add_subdirectory() by src/export/backend/CMakeLists.txt, which
# has already found MLIR.

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
target_link_libraries(NpuDialect PUBLIC MLIR)
