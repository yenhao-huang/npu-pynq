//===- DecomposeQuantOps.cpp - quant.* custom ops -> torch-dialect arithmetic -===//
//
// The front end's quantization (frontend/quantization/fx/custom_ops.py) leaves two
// custom ops in the graph, which torch-mlir imports as generic torch.operator
// ops; its backend pipeline rejects those. This pass expands each into the
// ATen ops of its arithmetic, op for op as the eager implementation computes
// it, so a compiled package computes what the eager PyTorch graph computes:
//
//   quant.quantize_per_tensor(x, scale, qmax) -> q          x: [M, K] f32, scale: [1] f32
//     q     = x * (reciprocal(scale) * 1.0)
//             |> round (half to even) |> clamp(-qmax, qmax) |> _to_copy(int8)
//
//   quant.dequantize(acc, x_scale, w_scale) -> y              acc: [M, N] i32
//     y     = _to_copy(acc, f32) * unsqueeze(x_scale, 1) * unsqueeze(w_scale, 0)
//     (x_scale is [M], or [1] for a per-tensor scale)
//
// The dequantize expansion is the form FuseConvBatchNorm matches after
// _int_mm, so this pass runs before it.
//
//===----------------------------------------------------------------------===//
#include "torch/transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace {

constexpr llvm::StringLiteral kQuantizePerTensor = "torch.quant.quantize_per_tensor";
constexpr llvm::StringLiteral kDequantize = "torch.quant.dequantize";

// A ranked value tensor of rank `rank` with a known dtype, or null.
ValueTensorType rankedTensor(Value value, size_t rank) {
  auto type = dyn_cast<ValueTensorType>(value.getType());
  if (!type || !type.hasSizes() || !type.hasDtype() || type.getSizes().size() != rank)
    return nullptr;
  return type;
}

// `type` with a dimension of size 1 inserted at `dim`, as aten.unsqueeze makes it.
ValueTensorType unsqueezed(ValueTensorType type, int64_t dim) {
  SmallVector<int64_t> sizes(type.getSizes());
  sizes.insert(sizes.begin() + dim, 1);
  return cast<ValueTensorType>(type.getWithSizesAndDtype(ArrayRef<int64_t>(sizes), type.getDtype()));
}

Value unsqueeze(PatternRewriter &rewriter, Location loc, Value value, int64_t dim) {
  auto type = cast<ValueTensorType>(value.getType());
  Value d = ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(dim));
  return AtenUnsqueezeOp::create(rewriter, loc, unsqueezed(type, dim), value, d);
}

// aten._to_copy(value, dtype) with every other argument at its default.
Value toCopy(PatternRewriter &rewriter, Location loc, Value value, ValueTensorType resultType) {
  Value dtype = getDtypeIntValueForType(rewriter, loc, resultType.getDtype());
  Value none = ConstantNoneOp::create(rewriter, loc);
  Value nonBlocking = ConstantBoolOp::create(rewriter, loc, false);
  return Aten_ToCopyOp::create(rewriter, loc, resultType, value, dtype, none, none, none,
                               nonBlocking, none);
}

Value floatConstant(PatternRewriter &rewriter, Location loc, double value) {
  return ConstantFloatOp::create(rewriter, loc, rewriter.getF64FloatAttr(value));
}

Value intConstant(PatternRewriter &rewriter, Location loc, int64_t value) {
  return ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(value));
}

struct DecomposeQuantizePerTensor : OpRewritePattern<OperatorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override {
    if (op.getName() != kQuantizePerTensor || op.getNumOperands() != 3 || op.getNumResults() != 1)
      return failure();
    Value x = op.getOperand(0), scale = op.getOperand(1);
    int64_t qmax;
    if (!matchPattern(op.getOperand(2), m_TorchConstantInt(&qmax)) || qmax <= 0)
      return rewriter.notifyMatchFailure(op, "qmax must be a constant");
    auto xType = rankedTensor(x, 2);
    auto qType = rankedTensor(op.getResult(0), 2);
    auto scaleType = rankedTensor(scale, 1);
    if (!xType || !qType || !scaleType || !xType.getDtype().isF32() || !scaleType.getDtype().isF32())
      return rewriter.notifyMatchFailure(op, "expects [M, K] f32 and [1] f32 in, [M, K] out");
    Location loc = op.getLoc();

    // q = to_int8(clamp(round(x * (1 / scale)), -qmax, qmax))
    // Python's 1.0 / scale is reciprocal, then a multiply by 1.0.
    Value reciprocal = AtenReciprocalOp::create(rewriter, loc, scaleType, scale);
    Value inverse = AtenMulScalarOp::create(rewriter, loc, scaleType, reciprocal,
                                            floatConstant(rewriter, loc, 1.0));
    Value scaled = AtenMulTensorOp::create(rewriter, loc, xType, x, inverse);
    Value rounded = AtenRoundOp::create(rewriter, loc, xType, scaled);
    Value saturated = AtenClampOp::create(rewriter, loc, xType, rounded, intConstant(rewriter, loc, -qmax),
                                          intConstant(rewriter, loc, qmax));
    rewriter.replaceOp(op, toCopy(rewriter, loc, saturated, qType));
    return success();
  }
};

struct DecomposeDequantize : OpRewritePattern<OperatorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override {
    if (op.getName() != kDequantize || op.getNumOperands() != 3 || op.getNumResults() != 1)
      return failure();
    Value acc = op.getOperand(0), xScale = op.getOperand(1), wScale = op.getOperand(2);
    auto yType = rankedTensor(op.getResult(0), 2);
    if (!rankedTensor(acc, 2) || !rankedTensor(xScale, 1) || !rankedTensor(wScale, 1) || !yType ||
        !yType.getDtype().isF32())
      return rewriter.notifyMatchFailure(op, "expects [M, N], [M], [N] in and [M, N] f32 out");
    Location loc = op.getLoc();

    // y = (float(acc) * x_scale[:, None]) * w_scale[None, :]
    Value y = toCopy(rewriter, loc, acc, yType);
    y = AtenMulTensorOp::create(rewriter, loc, yType, y, unsqueeze(rewriter, loc, xScale, 1));
    y = AtenMulTensorOp::create(rewriter, loc, yType, y, unsqueeze(rewriter, loc, wScale, 0));

    rewriter.replaceOp(op, y);
    return success();
  }
};

struct DecomposeQuantOpsPass : PassWrapper<DecomposeQuantOpsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeQuantOpsPass)

  StringRef getArgument() const final { return "opt-decompose-quant-ops"; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeQuantizePerTensor, DecomposeDequantize>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> middle_end::createDecomposeQuantOpsPass() {
  return std::make_unique<DecomposeQuantOpsPass>();
}
