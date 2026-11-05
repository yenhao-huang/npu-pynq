//===- Passes.h - rewrites on the torch dialect, before lowering -*- C++ -*-===//
//
// Each pass is one file and matches layers while they are still recognisable;
// lowering breaks them into linalg.generic loops. A pass changes nothing when
// its pattern is absent.
//
//===----------------------------------------------------------------------===//
#ifndef MIDDLE_END_TORCH_TRANSFORMS_PASSES_H
#define MIDDLE_END_TORCH_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace middle_end {

// convolution + BatchNorm -> one layer, with the BatchNorm folded into constants.
std::unique_ptr<mlir::Pass> createFuseConvBatchNormPass();
// aten._int_mm -> aten.mm, which this torch-mlir lowers to linalg.matmul.
std::unique_ptr<mlir::Pass> createDecomposeIntMmPass();
// aten.any.dim -> aten.any.dims, which this torch-mlir lowers.
std::unique_ptr<mlir::Pass> createAnyDimToAnyDimsPass();

} // namespace middle_end

#endif // MIDDLE_END_TORCH_TRANSFORMS_PASSES_H
