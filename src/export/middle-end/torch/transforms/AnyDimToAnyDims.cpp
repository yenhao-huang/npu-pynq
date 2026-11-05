//===- AnyDimToAnyDims.cpp - aten.any.dim -> aten.any.dims ----------------===//
//
// This torch-mlir lowers any.dims but not any.dim, which transformers' attention
// mask produces; later versions lower both.
//
//===----------------------------------------------------------------------===//
#include "torch/transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace {

struct AnyDimToAnyDims : OpRewritePattern<AtenAnyDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtenAnyDimOp op, PatternRewriter &rewriter) const override {
    Value dims = PrimListConstructOp::create(
        rewriter, op.getLoc(), ListType::get(IntType::get(op.getContext())),
        ValueRange{op.getDim()});
    rewriter.replaceOpWithNewOp<AtenAnyDimsOp>(op, op.getType(), op.getSelf(), dims,
                                               op.getKeepdim());
    return success();
  }
};

struct AnyDimToAnyDimsPass : PassWrapper<AnyDimToAnyDimsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnyDimToAnyDimsPass)

  StringRef getArgument() const final { return "npu-any-dim-to-any-dims"; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AnyDimToAnyDims>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> middle_end::createAnyDimToAnyDimsPass() {
  return std::make_unique<AnyDimToAnyDimsPass>();
}
