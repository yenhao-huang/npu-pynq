//===- DecomposeQuantOps.cpp - quant.* custom ops -> torch-dialect arithmetic -===//
//
// The front end's quantization (frontend/quantization/fx/ops.py) leaves two
// custom ops in the graph, which torch-mlir imports as generic torch.operator
// ops; its backend pipeline rejects those. This pass expands each into the
// ATen ops of its arithmetic, op for op as the eager implementation computes
// it, so a compiled package gives the same values as before the ops existed:
//
//   quant.quantize_per_token(x, eps, qmax) -> (q, scale)     x: [M, K] f32
//     scale = amax(abs(x), dim -1) |> clamp(min = eps) |> mul.Scalar(1 / qmax)
//     q     = x * unsqueeze(reciprocal(scale) * 1.0, 1)
//             |> round (half to even) |> clamp(-qmax, qmax) |> _to_copy(int8)
//
//   quant.dequantize(acc, x_scale, w_scale) -> y              acc: [M, N] i32
//     y     = _to_copy(acc, f32) * unsqueeze(x_scale, 1) * unsqueeze(w_scale, 0)
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

constexpr llvm::StringLiteral kQuantizePerToken = "torch.quant.quantize_per_token";
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

struct DecomposeQuantizePerToken : OpRewritePattern<OperatorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override {
    if (op.getName() != kQuantizePerToken || op.getNumOperands() != 3 || op.getNumResults() != 2)
      return failure();
    Value x = op.getOperand(0);
    double eps;
    int64_t qmax;
    if (!matchPattern(op.getOperand(1), m_TorchConstantFloat(&eps)) ||
        !matchPattern(op.getOperand(2), m_TorchConstantInt(&qmax)) || qmax <= 0)
      return rewriter.notifyMatchFailure(op, "eps and qmax must be constants");
    auto xType = rankedTensor(x, 2);
    auto qType = rankedTensor(op.getResult(0), 2);
    auto scaleType = rankedTensor(op.getResult(1), 1);
    if (!xType || !qType || !scaleType || !xType.getDtype().isF32() || !scaleType.getDtype().isF32())
      return rewriter.notifyMatchFailure(op, "expects [M, K] f32 in, [M, K] and [M] f32 out");
    Location loc = op.getLoc();

    // scale = clamp(amax(|x|, -1), min = eps) * (1 / qmax)
    Value abs = AtenAbsOp::create(rewriter, loc, xType, x);
    Value lastDim = PrimListConstructOp::create(
        rewriter, loc, ListType::get(IntType::get(op.getContext())),
        ValueRange{intConstant(rewriter, loc, -1)});
    Value keepDim = ConstantBoolOp::create(rewriter, loc, false);
    Value amax = AtenAmaxOp::create(rewriter, loc, scaleType, abs, lastDim, keepDim);
    Value none = ConstantNoneOp::create(rewriter, loc);
    Value clamped =
        AtenClampOp::create(rewriter, loc, scaleType, amax, floatConstant(rewriter, loc, eps), none);
    Value scale = AtenMulScalarOp::create(rewriter, loc, scaleType, clamped,
                                          floatConstant(rewriter, loc, 1.0 / qmax));

    // q = to_int8(clamp(round(x * (1 / scale)[:, None]), -qmax, qmax))
    // Python's 1.0 / scale is reciprocal, then a multiply by 1.0.
    Value reciprocal = AtenReciprocalOp::create(rewriter, loc, scaleType, scale);
    Value inverse = AtenMulScalarOp::create(rewriter, loc, scaleType, reciprocal,
                                            floatConstant(rewriter, loc, 1.0));
    Value scaled = AtenMulTensorOp::create(rewriter, loc, xType, x, unsqueeze(rewriter, loc, inverse, 1));
    Value rounded = AtenRoundOp::create(rewriter, loc, xType, scaled);
    Value saturated = AtenClampOp::create(rewriter, loc, xType, rounded, intConstant(rewriter, loc, -qmax),
                                          intConstant(rewriter, loc, qmax));
    Value q = toCopy(rewriter, loc, saturated, qType);

    rewriter.replaceOp(op, ValueRange{q, scale});
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
    patterns.add<DecomposeQuantizePerToken, DecomposeDequantize>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> middle_end::createDecomposeQuantOpsPass() {
  return std::make_unique<DecomposeQuantOpsPass>();
}
