//===- NpuPasses.h - the npu delegate's compile steps -----------*- C++ -*-===//
//
// The NpuCodegen library: plain C++ the npu delegate
// (src/export/backend/delegates/npu/) calls on each claimed linalg.matmul
// (npu::isClaimable, Npu/NpuMatch.h), given M, N, K, its constant weight and
// tk = min(K, MAX_K). The NPU runs K in chunks of tk; libnpu_rt sums the
// chunks, so the delegate's result is the full M x N product.
//
//===----------------------------------------------------------------------===//
#ifndef NPU_NPUPASSES_H
#define NPU_NPUPASSES_H

#include "Npu/NpuEncoder.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::npu {

// Pack: a [K, N] INT8 weight as the B tiles the NPU reads (spec/layout.py).
std::vector<int8_t> packB(const GemmPlan &plan, const int8_t *b);

// Encode: one claimed matmul as an ISA program, split into segments whose weight
// blocks fit `segmentBytes`. ADDR_B words hold offsets inside each segment's
// block; the loader relocates them.
struct Segment {
  int64_t wordOffset, length, niLo, niHi, weightOffset, bBytes;
  std::vector<int64_t> relocations;
};
struct Program {
  int64_t id;
  GemmPlan plan;
  int64_t weightOffset;
  std::vector<Segment> segments;
};
Program encode(int64_t id, const GemmPlan &plan, int64_t weightOffset,
               int64_t segmentBytes, std::vector<uint64_t> &words);

// programs.bin (little-endian words), programs.txt (the table libnpu_rt
// reads) and programs.json (the same, for people and tools).
bool writePrograms(const std::string &dir, const Limits &limits,
                   int64_t segmentBytes, const std::vector<uint64_t> &words,
                   const std::vector<Program> &programs, std::string &error);

} // namespace mlir::npu

#endif // NPU_NPUPASSES_H
