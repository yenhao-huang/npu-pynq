//===- Claim.cpp - rewrite a claimed contraction for the NPU --------------===//
//
// npu::claim rewrites one INT8 x INT8 -> INT32 contraction on a constant
// weight with a zero accumulator (npu::isClaimable, Match.cpp) into an NPU
// contraction with the same result:
//
//   linalg.matmul ins(%x, %w = arith.constant) outs(linalg.fill 0)
//       ->  npu.matmul %x, %w {tk = min(K, maxK)}
//
// The NPU runs K in chunks of tk; libnpu_rt sums the chunks, so nothing about
// them is left in the IR. The npu delegate calls claim on the matmul of each
// device function the middle end cut out; --npu-claim applies it to every
// claimable matmul of a module, standalone.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuDialect.h"
#include "Npu/NpuMatch.h"
#include "Npu/NpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::npu;

npu::MatmulOp mlir::npu::claim(linalg::MatmulOp op, int64_t maxK) {
  Value x = op.getInputs()[0], w = op.getInputs()[1];
  Operation *fill = op.getOutputs()[0].getDefiningOp();
  int64_t k = cast<RankedTensorType>(x.getType()).getDimSize(1);
  OpBuilder b(op);
  auto npu = MatmulOp::create(b, op.getLoc(), op.getResultTypes()[0], x, w,
                              b.getI64IntegerAttr(std::min<int64_t>(k, maxK)));
  op->getResult(0).replaceAllUsesWith(npu.getResult());
  op->erase();
  // The zero accumulator (fill of tensor.empty) is dead unless shared.
  if (fill && fill->use_empty()) {
    Operation *empty = fill->getOperand(1).getDefiningOp();
    fill->erase();
    if (empty && empty->use_empty())
      empty->erase();
  }
  return npu;
}

namespace {

struct ClaimPass : public PassWrapper<ClaimPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClaimPass)

  ClaimPass() = default;
  ClaimPass(int64_t k) { maxK = k; }
  ClaimPass(const ClaimPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "npu-claim"; }
  StringRef getDescription() const final {
    return "Move INT8 contractions on constant weights (zero accumulator) to the NPU";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NpuDialect, linalg::LinalgDialect, arith::ArithDialect>();
  }

  Option<int64_t> maxK{*this, "max-k", llvm::cl::desc("NPU MAX_K"),
                       llvm::cl::init(256)};

  void runOnOperation() override {
    SmallVector<linalg::MatmulOp> matmuls;
    getOperation().walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    for (linalg::MatmulOp op : matmuls)
      if (isClaimable(op))
        claim(op, maxK);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::npu::createClaimPass(int64_t maxK) {
  return std::make_unique<ClaimPass>(maxK);
}

void mlir::npu::registerClaimPass() { PassRegistration<ClaimPass>(); }
