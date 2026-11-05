//===- FuseConvBatchNorm.cpp - fold BatchNorm into the layer before it ----===//
//
// torch-mlir cannot lower _native_batch_norm_legit_no_training: it reaches the
// module as a generic torch.operator. Any model with BatchNorm therefore needs
// this pass to lower at all. The BatchNorm follows one of two forms:
//
//   float   convolution(x, weight, bias) -> BatchNorm
//           the weight and bias take the BatchNorm
//
//   W8A8    _int_mm -> to float -> * x_scale -> * w_scale [-> + bias]
//           -> view -> permute(0, 3, 1, 2) -> BatchNorm
//           (W8A8Conv2d's forward.) w_scale and the bias take the BatchNorm; the
//           INT8 weights the NPU reads do not change.
//
// A pair is left alone unless every value involved is a constant and each
// intermediate result has no other user. The arithmetic is that of PyTorch's
// fuse_conv_bn_weights, in float32:
//
//   multiplier = gamma / sqrt(var + eps)
//   weight'    = weight * multiplier              per output channel
//   bias'      = (bias - mean) / sqrt(var + eps) * gamma + beta
//
//===----------------------------------------------------------------------===//
#include "torch/transforms/Constants.h"
#include "torch/transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include <cmath>

using namespace mlir;
using namespace mlir::torch::Torch;

namespace {

constexpr llvm::StringLiteral kBatchNorm = "torch.aten._native_batch_norm_legit_no_training";

// Operand positions in the ATen schema of the BatchNorm.
enum { BnInput, BnGamma, BnBeta, BnMean, BnVar, BnMomentum, BnEps };

bool isNone(Value value) { return isa<mlir::torch::Torch::NoneType>(value.getType()); }

// BatchNorm's constants: one value per channel.
struct BatchNorm {
  std::vector<float> rsqrt, gamma, beta, mean;

  static std::optional<BatchNorm> read(OperatorOp op) {
    double eps;
    if (!matchPattern(op.getOperand(BnEps), m_TorchConstantFloat(&eps)))
      return std::nullopt;
    auto mean = middle_end::readFloats(op.getOperand(BnMean));
    auto var = middle_end::readFloats(op.getOperand(BnVar));
    if (!mean || !var || mean->size() != var->size())
      return std::nullopt;
    BatchNorm bn;
    bn.mean = *mean;
    bn.rsqrt.resize(var->size());
    for (size_t i = 0; i < var->size(); ++i)
      bn.rsqrt[i] = 1.0f / std::sqrt((*var)[i] + static_cast<float>(eps));
    // A BatchNorm without affine parameters scales by 1 and shifts by 0.
    bn.gamma.assign(mean->size(), 1.0f);
    bn.beta.assign(mean->size(), 0.0f);
    if (!isNone(op.getOperand(BnGamma))) {
      auto gamma = middle_end::readFloats(op.getOperand(BnGamma));
      if (!gamma || gamma->size() != mean->size())
        return std::nullopt;
      bn.gamma = *gamma;
    }
    if (!isNone(op.getOperand(BnBeta))) {
      auto beta = middle_end::readFloats(op.getOperand(BnBeta));
      if (!beta || beta->size() != mean->size())
        return std::nullopt;
      bn.beta = *beta;
    }
    return bn;
  }

  // weight[channel, ...] * multiplier; `weight` holds `channels` equal rows.
  std::vector<float> scaled(ArrayRef<float> weight) const {
    std::vector<float> out(weight.begin(), weight.end());
    size_t perChannel = out.size() / mean.size();
    for (size_t c = 0; c < mean.size(); ++c)
      for (size_t i = 0; i < perChannel; ++i)
        out[c * perChannel + i] *= gamma[c] * rsqrt[c];
    return out;
  }

  // An absent bias is zero.
  std::vector<float> shifted(const std::optional<std::vector<float>> &bias) const {
    std::vector<float> out(mean.size());
    for (size_t c = 0; c < mean.size(); ++c)
      out[c] = ((bias ? (*bias)[c] : 0.0f) - mean[c]) * rsqrt[c] * gamma[c] + beta[c];
    return out;
  }
};

struct FuseConvBatchNorm : OpRewritePattern<OperatorOp> {
  FuseConvBatchNorm(MLIRContext *context, middle_end::FusedConstants &constants)
      : OpRewritePattern(context), constants(constants) {}

  LogicalResult matchAndRewrite(OperatorOp bn, PatternRewriter &rewriter) const override {
    if (bn.getName() != kBatchNorm || bn.getNumOperands() != 7 || bn.getNumResults() != 3)
      return failure();
    // running_mean and running_var are returned, unused, in an eval BatchNorm.
    if (!bn.getResult(1).use_empty() || !bn.getResult(2).use_empty())
      return failure();
    Value input = bn.getOperand(BnInput);
    if (auto conv = input.getDefiningOp<AtenConvolutionOp>())
      return fuseFloat(conv, bn, rewriter);
    if (auto permute = input.getDefiningOp<AtenPermuteOp>())
      return fuseW8A8(permute, bn, rewriter);
    return failure();
  }

private:
  static void finish(OperatorOp bn, Value result, PatternRewriter &rewriter) {
    rewriter.replaceAllUsesWith(bn.getResult(0), result);
    rewriter.eraseOp(bn);
  }

  // A bias operand: nullopt in `out` for none; false if it is not a float constant.
  static bool readBias(Value value, std::optional<std::vector<float>> &out) {
    if (isNone(value))
      return true;
    out = middle_end::readFloats(value);
    return out.has_value();
  }

  LogicalResult fuseFloat(AtenConvolutionOp conv, OperatorOp bn,
                          PatternRewriter &rewriter) const {
    bool transposed;
    if (!conv.getResult().hasOneUse() ||
        !matchPattern(conv.getTransposed(), m_TorchConstantBool(&transposed)) || transposed)
      return failure();
    auto weight = middle_end::readFloats(conv.getWeight());
    std::optional<std::vector<float>> bias;
    auto params = BatchNorm::read(bn);
    if (!weight || !params || !readBias(conv.getBias(), bias))
      return failure();
    if (weight->size() % params->mean.size() != 0 ||
        (bias && bias->size() != params->mean.size()))
      return failure();

    std::string key = middle_end::resourceName(conv.getWeight());
    rewriter.setInsertionPoint(conv);
    Value newWeight = constants.literal(rewriter, conv.getLoc(), key + "_fused_weight",
                                        params->scaled(*weight), conv.getWeight());
    // A convolution without a bias gains one, shaped like BatchNorm's mean.
    Value newBias = constants.literal(rewriter, conv.getLoc(), key + "_fused_bias",
                                      params->shifted(bias), bn.getOperand(BnMean));
    rewriter.modifyOpInPlace(conv, [&] {
      conv.getWeightMutable().assign(newWeight);
      conv.getBiasMutable().assign(newBias);
    });
    finish(bn, conv.getResult(), rewriter);
    return success();
  }

  LogicalResult fuseW8A8(AtenPermuteOp permute, OperatorOp bn,
                         PatternRewriter &rewriter) const {
    SmallVector<int64_t> dims;
    if (!permute.getResult().hasOneUse() ||
        !matchPattern(permute.getDims(), m_TorchListOfConstantInts(dims)) ||
        dims != SmallVector<int64_t>{0, 3, 1, 2})
      return failure();
    auto view = permute.getSelf().getDefiningOp<AtenViewOp>();
    if (!view || !view.getResult().hasOneUse())
      return failure();
    auto add = view.getSelf().getDefiningOp<AtenAddTensorOp>();
    if (add && !add.getResult().hasOneUse())
      return failure();
    int64_t alpha;
    if (add && (!matchPattern(add.getAlpha(), m_TorchConstantInt(&alpha)) || alpha != 1))
      return failure();
    auto mul = (add ? add.getSelf() : view.getSelf()).getDefiningOp<AtenMulTensorOp>();
    if (!mul || !mul.getResult().hasOneUse())
      return failure();
    // Make sure this is the rescale of a contraction: (float(_int_mm) * x_scale) * w_scale.
    auto xRescaled = mul.getSelf().getDefiningOp<AtenMulTensorOp>();
    auto toFloat = xRescaled ? xRescaled.getSelf().getDefiningOp<Aten_ToCopyOp>() : nullptr;
    if (!toFloat || !toFloat.getSelf().getDefiningOp<Aten_IntMmOp>())
      return failure();
    auto unsqueeze = mul.getOther().getDefiningOp<AtenUnsqueezeOp>();
    int64_t unsqueezeDim;
    if (!unsqueeze || !unsqueeze.getResult().hasOneUse() ||
        !matchPattern(unsqueeze.getDim(), m_TorchConstantInt(&unsqueezeDim)) || unsqueezeDim != 0)
      return failure();

    auto scale = middle_end::readFloats(unsqueeze.getSelf());
    std::optional<std::vector<float>> bias;
    auto params = BatchNorm::read(bn);
    if (!scale || !params || (add && !readBias(add.getOther(), bias)))
      return failure();
    if (add && !bias)
      return failure();
    if (scale->size() != params->mean.size() || (bias && bias->size() != params->mean.size()))
      return failure();

    std::string key = middle_end::resourceName(unsqueeze.getSelf());
    rewriter.setInsertionPoint(unsqueeze);
    Value newScale = constants.literal(rewriter, unsqueeze.getLoc(), key + "_fused_scale",
                                       params->scaled(*scale), unsqueeze.getSelf());
    rewriter.modifyOpInPlace(unsqueeze, [&] { unsqueeze.getSelfMutable().assign(newScale); });
    rewriter.setInsertionPoint(add ? add.getOperation() : view.getOperation());
    Value newBias = constants.literal(rewriter, view.getLoc(), key + "_fused_bias",
                                      params->shifted(bias), bn.getOperand(BnMean));
    if (add) {
      rewriter.modifyOpInPlace(add, [&] { add.getOtherMutable().assign(newBias); });
    } else {
      // A contraction without a bias gains one, added before the view as W8A8Conv2d would.
      Value one = ConstantIntOp::create(rewriter, view.getLoc(), rewriter.getI64IntegerAttr(1));
      Value added = AtenAddTensorOp::create(rewriter, view.getLoc(), mul.getType(), mul.getResult(),
                                            newBias, one);
      rewriter.modifyOpInPlace(view, [&] { view.getSelfMutable().assign(added); });
    }
    finish(bn, permute.getResult(), rewriter);
    return success();
  }

  middle_end::FusedConstants &constants;
};

struct FuseConvBatchNormPass : PassWrapper<FuseConvBatchNormPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseConvBatchNormPass)

  StringRef getArgument() const final { return "npu-fuse-conv-batch-norm"; }

  void runOnOperation() override {
    middle_end::FusedConstants constants;
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseConvBatchNorm>(&getContext(), constants);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> middle_end::createFuseConvBatchNormPass() {
  return std::make_unique<FuseConvBatchNormPass>();
}
