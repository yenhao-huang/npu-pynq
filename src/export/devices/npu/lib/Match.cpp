//===- Match.cpp - the contractions the NPU runs --------------------------===//
//
// The claim rule, shared by the middle end's device assignment
// (pynqo-assign-devices) and --npu-claim. Upstream dialects only.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuMatch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace {

bool isIntMatrix(Value v, unsigned width) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.hasStaticShape() && t.getRank() == 2 &&
         t.getElementType().isInteger(width);
}

// A zero accumulator: linalg.fill of an integer constant 0 into an
// operand-free tensor.empty. The NPU's result is the product alone, so the
// delegate's only input is the activation.
bool isZeroInit(Value v) {
  auto fill = v.getDefiningOp<linalg::FillOp>();
  if (!fill || fill.getInputs().size() != 1 || fill.getOutputs().size() != 1)
    return false;
  auto zero = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto value = zero ? dyn_cast<IntegerAttr>(zero.getValue()) : IntegerAttr();
  auto empty = fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>();
  return value && value.getValue().isZero() && empty && empty->getNumOperands() == 0;
}

} // namespace

bool mlir::npu::isClaimable(Operation *op) {
  auto matmul = dyn_cast<linalg::MatmulOp>(op);
  if (!matmul || matmul.hasUserDefinedMaps() || matmul.getInputs().size() != 2)
    return false;
  Value x = matmul.getInputs()[0], w = matmul.getInputs()[1];
  Value acc = matmul.getOutputs()[0];
  return isIntMatrix(x, 8) && isIntMatrix(w, 8) && isIntMatrix(acc, 32) &&
         w.getDefiningOp<arith::ConstantOp>() && isZeroInit(acc);
}
