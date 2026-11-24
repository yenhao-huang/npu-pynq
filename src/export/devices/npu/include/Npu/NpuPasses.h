//===- NpuPasses.h - NPU passes and the delegate's compile steps -*- C++ -*-===//
#ifndef NPU_NPUPASSES_H
#define NPU_NPUPASSES_H

#include "Npu/NpuDialect.h"
#include "Npu/NpuEncoder.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::npu {

// Rewrite one claimable matmul (npu::isClaimable, Npu/NpuMatch.h) into
// npu.matmul {tk = min(K, maxK)} with the same M x N result; return it. The
// matmul and its zero accumulator, if nothing else reads it, are erased.
MatmulOp claim(linalg::MatmulOp op, int64_t maxK);

// --npu-claim: every INT8 x INT8 -> INT32 linalg.matmul whose right operand
// is a constant and whose accumulator is zero becomes npu.matmul
// (isClaimable + claim).
std::unique_ptr<Pass> createClaimPass(int64_t maxK = 256);
void registerClaimPass();

// Pack: a [K, N] INT8 weight as the B tiles the NPU reads (spec/layout.py).
std::vector<int8_t> packB(const GemmPlan &plan, const int8_t *b);

// Encode: one npu.matmul as an ISA program, split into segments whose weight
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
