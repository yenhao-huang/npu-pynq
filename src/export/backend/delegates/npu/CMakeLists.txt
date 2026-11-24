# Two static libraries (LLVM 22 / MLIR), added with add_subdirectory() by a
# tool that has already found MLIR:
#
#   NpuMatch    npu::isClaimable, the NPU's claim rule. Upstream linalg and
#               arith headers only, and no link libraries: each consumer
#               supplies MLIR in its own form. pynqo-middle-end links it with
#               static MLIR components, pynqo-backend (through NpuCodegen)
#               with the monolithic MLIR library. Never link MLIR here; two
#               link styles in one binary register everything twice.
#   NpuCodegen  the npu delegate's Pack and Encode steps (B tiles, ISA
#               programs, programs.*), plain C++ called on the claimed
#               linalg.matmul; linked by pynqo-backend only. pynqo-middle-end
#               adds this folder EXCLUDE_FROM_ALL and never builds it.

add_library(NpuMatch STATIC lib/Match.cpp)
target_include_directories(NpuMatch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(NpuMatch PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
llvm_update_compile_flags(NpuMatch)

add_library(NpuCodegen STATIC
  lib/NpuEncoder.cpp
  lib/Pack.cpp
  lib/Encode.cpp)
target_include_directories(NpuCodegen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
llvm_update_compile_flags(NpuCodegen)
target_link_libraries(NpuCodegen PUBLIC NpuMatch MLIR)
