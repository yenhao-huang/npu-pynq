//===- pipeline.h - CPU graph to the LLVM dialect for one target -*- C++ -*-===//
#ifndef BACKEND_CPU_LOWERING_PIPELINE_H
#define BACKEND_CPU_LOWERING_PIPELINE_H

#include "device_register.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace backend {

// Lower `module` (linalg on tensors, with __delegate__ calls) in place to the
// LLVM dialect for `target`: unit-stride interchange, bufferization, loops,
// LLVM dialect. Public functions keep MLIR's C interface. `profileKernels`
// times every kernel (cpu/profiling/kernel_probes.h) and writes
// `<dir>/kernels.json`.
mlir::LogicalResult lowerToLlvm(mlir::ModuleOp module, const CpuTarget &target,
                                const std::string &dir, bool profileKernels,
                                std::string &error);

} // namespace backend

#endif // BACKEND_CPU_LOWERING_PIPELINE_H
