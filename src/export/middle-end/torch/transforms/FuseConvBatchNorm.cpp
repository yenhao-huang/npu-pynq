//===- FuseConvBatchNorm.cpp - fold BatchNorm into the layer before it ----===//
//
// torch-mlir cannot lower _native_batch_norm_legit_no_training: it reaches the
// module as a generic torch.operator. Any model with BatchNorm therefore needs
// this pass to lower at all.
//
// In inference, BatchNorm is a per-channel multiply and add:
//
//   BN(y)      = (y - mean) / sqrt(var + eps) * gamma + beta
//   multiplier = gamma / sqrt(var + eps)
//
// so the layer before it can absorb it, with the arithmetic of PyTorch's
// fuse_conv_bn_weights in float32:
//
//   weight' = weight * multiplier                       per output channel
//   bias'   = (bias - mean) / sqrt(var + eps) * gamma + beta
//
// The layer before the BatchNorm takes one of two forms:
//
//   float   convolution(x, weight, bias) -> BatchNorm
//           The convolution's weight and bias are replaced.
//
//   W8A8    _int_mm -> to float -> * x_scale -> * w_scale [-> + bias]
//           -> view -> permute(0, 3, 1, 2) -> BatchNorm
//           (W8A8Conv2d's forward.) w_scale takes the multiplier and the bias
//           is replaced; the INT8 weights and _int_mm do not change.
//
// Each case is handled in three steps: match, compute, rewrite. A BatchNorm is
// left alone unless every value involved is a constant and each intermediate
// result has no other user.
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

using Floats = std::vector<float>;

//===----------------------------------------------------------------------===//
// Small matching helpers
//===----------------------------------------------------------------------===//

bool isNone(Value value) { return isa<mlir::torch::Torch::NoneType>(value.getType()); }

// True when `op` exists and its result feeds exactly one op, so rewriting it
// cannot change any other user.
template <typename OpT> bool feedsOnlyOneOp(OpT op) {
  return op && op.getResult().hasOneUse();
}

bool isConstantInt(Value value, int64_t expected) {
  int64_t actual;
  return matchPattern(value, m_TorchConstantInt(&actual)) && actual == expected;
}

// A bias operand: true with `out` empty for None, true with `out` set for a
// float32 constant, false for anything else.
bool readOptionalBias(Value value, std::optional<Floats> &out) {
  if (isNone(value))
    return true;
  out = middle_end::readFloats(value);
  return out.has_value();
}

//===----------------------------------------------------------------------===//
// BatchNorm's constants and the arithmetic of the fold
//===----------------------------------------------------------------------===//

struct BatchNorm {
  Floats rsqrt; // 1 / sqrt(var + eps)
  Floats gamma, beta, mean;

  size_t channels() const { return mean.size(); }

  // Reads the constants of `op`, or nullopt when any of them is not a constant.
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
    for (size_t c = 0; c < var->size(); ++c)
      bn.rsqrt[c] = 1.0f / std::sqrt((*var)[c] + static_cast<float>(eps));

    // A BatchNorm without affine parameters scales by 1 and shifts by 0.
    bn.gamma.assign(bn.channels(), 1.0f);
    bn.beta.assign(bn.channels(), 0.0f);
    if (!readAffine(op.getOperand(BnGamma), bn.gamma) ||
        !readAffine(op.getOperand(BnBeta), bn.beta))
      return std::nullopt;
    return bn;
  }

  // weight' = weight * gamma / sqrt(var + eps), per output channel.
  // `weight` is `channels()` equal blocks, one per output channel.
  Floats foldIntoWeight(ArrayRef<float> weight) const {
    Floats out(weight.begin(), weight.end());
    size_t perChannel = out.size() / channels();
    for (size_t c = 0; c < channels(); ++c)
      for (size_t i = 0; i < perChannel; ++i)
        out[c * perChannel + i] *= gamma[c] * rsqrt[c];
    return out;
  }

  // bias' = (bias - mean) / sqrt(var + eps) * gamma + beta. No bias is zero.
  Floats foldIntoBias(const std::optional<Floats> &bias) const {
    Floats out(channels());
    for (size_t c = 0; c < channels(); ++c)
      out[c] = ((bias ? (*bias)[c] : 0.0f) - mean[c]) * rsqrt[c] * gamma[c] + beta[c];
    return out;
  }

private:
  // gamma or beta: keeps the default for None, fails unless one value per channel.
  static bool readAffine(Value value, Floats &out) {
    if (isNone(value))
      return true;
    auto values = middle_end::readFloats(value);
    if (!values || values->size() != out.size())
      return false;
    out = *values;
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Matching the layer before the BatchNorm
//===----------------------------------------------------------------------===//

// float: convolution(x, weight, bias) -> BatchNorm
struct FloatConv {
  AtenConvolutionOp conv;
  Floats weight;
  std::optional<Floats> bias;
};

std::optional<FloatConv> matchFloatConv(Value input, size_t channels) {
  FloatConv m;
  m.conv = input.getDefiningOp<AtenConvolutionOp>();
  if (!feedsOnlyOneOp(m.conv))
    return std::nullopt;
  // A transposed convolution lays its weight out by input channel first.
  bool transposed;
  if (!matchPattern(m.conv.getTransposed(), m_TorchConstantBool(&transposed)) || transposed)
    return std::nullopt;

  auto weight = middle_end::readFloats(m.conv.getWeight());
  if (!weight || weight->size() % channels != 0)
    return std::nullopt;
  m.weight = *weight;
  if (!readOptionalBias(m.conv.getBias(), m.bias) || (m.bias && m.bias->size() != channels))
    return std::nullopt;
  return m;
}

// W8A8: _int_mm -> to float -> * x_scale -> * unsqueeze(w_scale) [-> + bias]
//       -> view -> permute(0, 3, 1, 2) -> BatchNorm
struct W8A8Conv {
  AtenUnsqueezeOp wScale;  // unsqueeze(w_scale, 0)
  AtenMulTensorOp rescale; // (float(_int_mm) * x_scale) * wScale
  AtenAddTensorOp addBias; // rescale + bias; null when the layer has no bias
  AtenViewOp view;
  AtenPermuteOp permute;
  Floats scale;
  std::optional<Floats> bias;
};

// True when `value` is float(_int_mm(...)) * x_scale: the contraction's output
// rescaled by the activation scale.
bool isRescaledIntMm(Value value) {
  auto xRescaled = value.getDefiningOp<AtenMulTensorOp>();
  auto toFloat = xRescaled ? xRescaled.getSelf().getDefiningOp<Aten_ToCopyOp>() : nullptr;
  return toFloat && toFloat.getSelf().getDefiningOp<Aten_IntMmOp>();
}

std::optional<W8A8Conv> matchW8A8Conv(Value input, size_t channels) {
  // Walk up from the BatchNorm's input; every op on the way must feed only the next.
  W8A8Conv m;

  // permute(0, 3, 1, 2): NHWC back to NCHW.
  m.permute = input.getDefiningOp<AtenPermuteOp>();
  SmallVector<int64_t> dims;
  if (!feedsOnlyOneOp(m.permute) ||
      !matchPattern(m.permute.getDims(), m_TorchListOfConstantInts(dims)) ||
      dims != SmallVector<int64_t>{0, 3, 1, 2})
    return std::nullopt;

  // view: the contraction's [N*H*W, C] rows back to [N, H, W, C].
  m.view = m.permute.getSelf().getDefiningOp<AtenViewOp>();
  if (!feedsOnlyOneOp(m.view))
    return std::nullopt;

  // + bias, if the layer has one. alpha must be 1: a plain add.
  m.addBias = m.view.getSelf().getDefiningOp<AtenAddTensorOp>();
  if (m.addBias && (!feedsOnlyOneOp(m.addBias) || !isConstantInt(m.addBias.getAlpha(), 1)))
    return std::nullopt;

  // * w_scale, applied to the activation-rescaled contraction.
  Value rescaled = m.addBias ? m.addBias.getSelf() : m.view.getSelf();
  m.rescale = rescaled.getDefiningOp<AtenMulTensorOp>();
  if (!feedsOnlyOneOp(m.rescale) || !isRescaledIntMm(m.rescale.getSelf()))
    return std::nullopt;
  m.wScale = m.rescale.getOther().getDefiningOp<AtenUnsqueezeOp>();
  if (!feedsOnlyOneOp(m.wScale) || !isConstantInt(m.wScale.getDim(), 0))
    return std::nullopt;

  // The constants: one w_scale per output channel, and the bias if present.
  auto scale = middle_end::readFloats(m.wScale.getSelf());
  if (!scale || scale->size() != channels)
    return std::nullopt;
  m.scale = *scale;
  if (m.addBias) {
    m.bias = middle_end::readFloats(m.addBias.getOther());
    if (!m.bias || m.bias->size() != channels)
      return std::nullopt;
  }
  return m;
}

//===----------------------------------------------------------------------===//
// The pattern
//===----------------------------------------------------------------------===//

struct FuseConvBatchNorm : OpRewritePattern<OperatorOp> {
  FuseConvBatchNorm(MLIRContext *context, middle_end::FusedConstants &constants)
      : OpRewritePattern(context), constants(constants) {}

  LogicalResult matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override {
    // An inference BatchNorm. It also returns running_mean and running_var,
    // which nothing may use.
    if (op.getName() != kBatchNorm || op.getNumOperands() != 7 || op.getNumResults() != 3)
      return failure();
    if (!op.getResult(1).use_empty() || !op.getResult(2).use_empty())
      return failure();
    auto bn = BatchNorm::read(op);
    if (!bn)
      return failure();

    Value input = op.getOperand(BnInput);
    if (auto conv = matchFloatConv(input, bn->channels()))
      return fuseFloat(*conv, *bn, op, rewriter);
    if (auto conv = matchW8A8Conv(input, bn->channels()))
      return fuseW8A8(*conv, *bn, op, rewriter);
    return failure();
  }

private:
  // The convolution takes weight' and bias'.
  LogicalResult fuseFloat(FloatConv &m, const BatchNorm &bn, OperatorOp op,
                          PatternRewriter &rewriter) const {
    std::string key = middle_end::resourceName(m.conv.getWeight());
    Location loc = m.conv.getLoc();
    rewriter.setInsertionPoint(m.conv);
    Value weight = constants.literal(rewriter, loc, key + "_fused_weight",
                                     bn.foldIntoWeight(m.weight), m.conv.getWeight());
    // A convolution without a bias gains one, shaped like BatchNorm's mean.
    Value bias = constants.literal(rewriter, loc, key + "_fused_bias", bn.foldIntoBias(m.bias),
                                   op.getOperand(BnMean));
    rewriter.modifyOpInPlace(m.conv, [&] {
      m.conv.getWeightMutable().assign(weight);
      m.conv.getBiasMutable().assign(bias);
    });
    replaceBatchNorm(op, m.conv.getResult(), rewriter);
    return success();
  }

  // w_scale takes the multiplier and the bias becomes bias'; the INT8 weights stay.
  LogicalResult fuseW8A8(W8A8Conv &m, const BatchNorm &bn, OperatorOp op,
                         PatternRewriter &rewriter) const {
    std::string key = middle_end::resourceName(m.wScale.getSelf());
    Location loc = m.view.getLoc();

    rewriter.setInsertionPoint(m.wScale);
    Value scale = constants.literal(rewriter, m.wScale.getLoc(), key + "_fused_scale",
                                    bn.foldIntoWeight(m.scale), m.wScale.getSelf());
    rewriter.modifyOpInPlace(m.wScale, [&] { m.wScale.getSelfMutable().assign(scale); });

    Operation *beforeView = m.addBias ? m.addBias.getOperation() : m.view.getOperation();
    rewriter.setInsertionPoint(beforeView);
    Value bias = constants.literal(rewriter, loc, key + "_fused_bias", bn.foldIntoBias(m.bias),
                                   op.getOperand(BnMean));
    if (m.addBias) {
      rewriter.modifyOpInPlace(m.addBias, [&] { m.addBias.getOtherMutable().assign(bias); });
    } else {
      // A layer without a bias gains one, added before the view as W8A8Conv2d would.
      Value one = ConstantIntOp::create(rewriter, loc, rewriter.getI64IntegerAttr(1));
      Value added =
          AtenAddTensorOp::create(rewriter, loc, m.rescale.getType(), m.rescale.getResult(), bias, one);
      rewriter.modifyOpInPlace(m.view, [&] { m.view.getSelfMutable().assign(added); });
    }
    replaceBatchNorm(op, m.permute.getResult(), rewriter);
    return success();
  }

  // Every user of the BatchNorm's output now reads `result`; the BatchNorm goes.
  static void replaceBatchNorm(OperatorOp op, Value result, PatternRewriter &rewriter) {
    rewriter.replaceAllUsesWith(op.getResult(0), result);
    rewriter.eraseOp(op);
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
