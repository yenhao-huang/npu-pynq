//===- NpuPasses.h - NPU partitioning and lowering --------------*- C++ -*-===//
#ifndef NPU_NPUPASSES_H
#define NPU_NPUPASSES_H

namespace mlir::npu {
// --npu-partition: rewrite INT8 contractions whose right operand is a weight
// into npu.matmul (NPU tasks), outline each into the npu.module (the NPU
// subgraph) behind an npu.call (the host subgraph); report the split.
void registerPartitionPass();
// --npu-encode: encode each task of the npu.module as an ISA program.
void registerEncodePass();
// --npu-lower-host: replace each npu.call with a call into the NPU runtime
// and bind weights to the arena.
void registerHostLowerPass();
} // namespace mlir::npu

#endif // NPU_NPUPASSES_H
