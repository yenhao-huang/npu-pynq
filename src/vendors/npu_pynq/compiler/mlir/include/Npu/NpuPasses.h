//===- NpuPasses.h - NPU partitioning and lowering --------------*- C++ -*-===//
#ifndef NPU_NPUPASSES_H
#define NPU_NPUPASSES_H

namespace mlir::npu {
// --npu-partition: rewrite INT8 contractions whose right operand is a weight
// into npu.matmul (NPU tasks); report the split.
void registerPartitionPass();
// --npu-lower: bind weights to the arena, encode each NPU task as an ISA
// program, and replace it with a call into the NPU runtime.
void registerLowerPass();
} // namespace mlir::npu

#endif // NPU_NPUPASSES_H
