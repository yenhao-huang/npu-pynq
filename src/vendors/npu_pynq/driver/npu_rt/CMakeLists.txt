# libnpu_rt.so, the NPU delegate backend. Built on the PYNQ-Z1 itself with the
# g++ and CMake its image ships (no cross-compiler, no sysroot), and natively
# on the export machine, where only the simulator backend is used.
#
#   cmake -S src/vendors/npu_pynq/driver/npu_rt -B build/npu_rt -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/npu_rt
cmake_minimum_required(VERSION 3.10)
project(npu_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(npu_rt SHARED load.cpp execute.cpp pynq.cpp sim.cpp)
target_include_directories(npu_rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
