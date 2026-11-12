//===- Match.cpp - the contractions the NPU runs --------------------------===//
//
// The claim rule, shared by the middle end's device assignment
// (pynqo-assign-devices) and --npu-claim. Upstream dialects only.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuMatch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace {

bool isIntMatrix(Value v, unsigned width) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.hasStaticShape() && t.getRank() == 2 &&
         t.getElementType().isInteger(width);
}

} // namespace

bool mlir::npu::isClaimable(Operation *op) {
  auto matmul = dyn_cast<linalg::MatmulOp>(op);
  if (!matmul || matmul.hasUserDefinedMaps() || matmul.getInputs().size() != 2)
    return false;
  Value x = matmul.getInputs()[0], w = matmul.getInputs()[1];
  Value acc = matmul.getOutputs()[0];
  return isIntMatrix(x, 8) && isIntMatrix(w, 8) && isIntMatrix(acc, 32) &&
         w.getDefiningOp<arith::ConstantOp>();
}
