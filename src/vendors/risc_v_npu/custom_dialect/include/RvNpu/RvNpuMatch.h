//===- RvNpuMatch.h - risc_v_npu's claim rule, without the dialect -*- C++ -*-===//
//
// The RvNpuMatch library: only the predicate that decides which operations
// the risc_v_npu device runs. It needs upstream linalg, tensor and arith and
// nothing generated, so the middle end (pynqo-middle-end) links it without
// the rvnpu dialect, the RISC-V lowering, or the monolithic MLIR library.
// --rvnpu-claim and the risc_v_npu delegate use the same predicate.
//
// Copied from src/vendors/npu_pynq/custom_dialect/include/Npu/NpuMatch.h
// (risc_v_npu keeps its own copy; plan 2026-1112-003 section 6.2). Keep the
// two in step: both devices run the same matrix core.
//
//===----------------------------------------------------------------------===//
#ifndef RVNPU_RVNPUMATCH_H
#define RVNPU_RVNPUMATCH_H

namespace mlir {
class Operation;
namespace rvnpu {

// An INT8 x INT8 -> INT32 linalg.matmul the device runs: no user-defined
// maps, two inputs, static 2-D operands, a right operand (the weight) defined
// by an arith.constant, and an accumulator that is linalg.fill of 0 on
// tensor.empty (the device returns the product alone). Pattern match only;
// nothing is rewritten.
bool isClaimable(Operation *op);

} // namespace rvnpu
} // namespace mlir

#endif // RVNPU_RVNPUMATCH_H
