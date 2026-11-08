//===- Constants.cpp - float32 tensors of torch.vtensor.literal ------------===//
#include "torch/transforms/Constants.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace middle_end {

std::optional<std::vector<float>> readFloats(Value value) {
  auto literal = value.getDefiningOp<ValueTensorLiteralOp>();
  if (!literal)
    return std::nullopt;
  ElementsAttr attr = literal.getValue();
  auto type = dyn_cast<RankedTensorType>(attr.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32())
    return std::nullopt;
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(attr)) {
    AsmResourceBlob *blob = resource.getRawHandle().getBlob();
    if (!blob)
      return std::nullopt;
    ArrayRef<float> data = blob->getDataAs<float>();
    if (static_cast<int64_t>(data.size()) != type.getNumElements())
      return std::nullopt;
    return std::vector<float>(data.begin(), data.end());
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    auto values = dense.getValues<float>();
    return std::vector<float>(values.begin(), values.end());
  }
  return std::nullopt;
}

std::string resourceName(Value literal) {
  auto op = literal.getDefiningOp<ValueTensorLiteralOp>();
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(op.getValue()))
    return resource.getRawHandle().getKey().str();
  return "dense";
}

Value FusedConstants::literal(PatternRewriter &rewriter, Location loc, StringRef name,
                              ArrayRef<float> data, Value like) {
  auto it = made.find(name);
  ElementsAttr attr;
  if (it != made.end()) {
    attr = it->second;
  } else {
    auto type = cast<RankedTensorType>(
        like.getDefiningOp<ValueTensorLiteralOp>().getValue().getType());
    attr = DenseResourceElementsAttr::get(
        type, name, HeapAsmResourceBlob::allocateAndCopyInferAlign(data));
    made[name] = attr;
  }
  return ValueTensorLiteralOp::create(rewriter, loc, like.getType(), attr);
}

} // namespace middle_end
