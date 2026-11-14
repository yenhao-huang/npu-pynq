//===- NpuMatch.h - the NPU's claim rule, without the npu dialect -*- C++ -*-===//
//
// The NpuMatch library: only the predicate that decides which operations the
// NPU runs. It needs upstream linalg, tensor and arith and nothing generated, so the
// middle end (pynqo-middle-end) links it without the npu dialect, the
// encoder, or the monolithic MLIR library. --npu-claim uses the same
// predicate.
//
//===----------------------------------------------------------------------===//
#ifndef NPU_NPUMATCH_H
#define NPU_NPUMATCH_H

namespace mlir {
class Operation;
namespace npu {

// An INT8 x INT8 -> INT32 linalg.matmul the NPU runs: no user-defined maps,
// two inputs, static 2-D operands, a right operand (the weight) defined by
// an arith.constant, and an accumulator that is linalg.fill of 0 on
// tensor.empty (the NPU returns the product alone). Pattern match only;
// nothing is rewritten.
bool isClaimable(Operation *op);

} // namespace npu
} // namespace mlir

#endif // NPU_NPUMATCH_H
