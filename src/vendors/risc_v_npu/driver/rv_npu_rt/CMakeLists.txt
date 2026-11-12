# librv_npu_rt.so, the risc_v_npu delegate backend (CVA6 scalar core). Built on
# the PYNQ-Z1 itself with the g++ and CMake its image ships, as libnpu_rt is.
# It runs only on the board: there is no simulator backend.
#
#   cmake -S src/vendors/risc_v_npu/driver/rv_npu_rt -B build/rv_npu_rt -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/rv_npu_rt
cmake_minimum_required(VERSION 3.10)
project(rv_npu_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(rv_npu_rt SHARED load.cpp execute.cpp pynq.cpp profile.cpp)
target_include_directories(rv_npu_rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
