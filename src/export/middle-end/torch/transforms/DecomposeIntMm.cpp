//===- DecomposeIntMm.cpp - aten._int_mm -> aten.mm -----------------------===//
//
// torch._int_mm is W8A8's INT8 x INT8 -> INT32 contraction. This torch-mlir has
// no lowering for it; upstream later added this same rewrite. Eager torch.mm
// returns the operand dtype while _int_mm returns int32, so the int32 result
// type is kept, and --npu-claim sees an i8 x i8 -> i32 linalg.matmul.
//
//===----------------------------------------------------------------------===//
#include "torch/transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace {

struct DecomposeIntMm : OpRewritePattern<Aten_IntMmOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Aten_IntMmOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AtenMmOp>(op, op.getType(), op.getSelf(), op.getMat2());
    return success();
  }
};

struct DecomposeIntMmPass : PassWrapper<DecomposeIntMmPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeIntMmPass)

  StringRef getArgument() const final { return "npu-decompose-int-mm"; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeIntMm>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> middle_end::createDecomposeIntMmPass() {
  return std::make_unique<DecomposeIntMmPass>();
}
