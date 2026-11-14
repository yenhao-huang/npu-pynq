//===- Claim.cpp - rewrite a claimed contraction for risc_v_npu -----------===//
//
// Copied from src/vendors/npu_pynq/custom_dialect/lib/Claim.cpp and renamed to
// the rvnpu dialect (mlir::rvnpu, --rvnpu-claim), so both devices link into one
// pynqo-backend; keep the two in step (plan 2026-1112-003 section 6.2).
//
// rvnpu::claim rewrites one INT8 x INT8 -> INT32 contraction on a constant
// weight with a zero accumulator (rvnpu::isClaimable, Match.cpp) into an NPU
// contraction with the same result:
//
//   linalg.matmul ins(%x, %w = arith.constant) outs(linalg.fill 0)
//       ->  rvnpu.matmul %x, %w {tk = min(K, maxK)}
//
// The NPU runs K in chunks of tk; librv_npu_rt sums the chunks, so nothing
// about them is left in the IR. The risc_v_npu delegate calls claim on the
// matmul of each device function the middle end cut out; --rvnpu-claim
// applies it to every claimable matmul of a module, standalone.
//
//===----------------------------------------------------------------------===//

#include "RvNpu/RvNpuDialect.h"
#include "RvNpu/RvNpuMatch.h"
#include "RvNpu/RvNpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::rvnpu;

rvnpu::MatmulOp mlir::rvnpu::claim(linalg::MatmulOp op, int64_t maxK) {
  Value x = op.getInputs()[0], w = op.getInputs()[1];
  Operation *fill = op.getOutputs()[0].getDefiningOp();
  int64_t k = cast<RankedTensorType>(x.getType()).getDimSize(1);
  OpBuilder b(op);
  auto matmul = MatmulOp::create(b, op.getLoc(), op.getResultTypes()[0], x, w,
                                 b.getI64IntegerAttr(std::min<int64_t>(k, maxK)));
  op->getResult(0).replaceAllUsesWith(matmul.getResult());
  op->erase();
  // The zero accumulator (fill of tensor.empty) is dead unless shared.
  if (fill && fill->use_empty()) {
    Operation *empty = fill->getOperand(1).getDefiningOp();
    fill->erase();
    if (empty && empty->use_empty())
      empty->erase();
  }
  return matmul;
}

namespace {

struct RvNpuClaimPass : public PassWrapper<RvNpuClaimPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RvNpuClaimPass)

  RvNpuClaimPass() = default;
  RvNpuClaimPass(int64_t k) { maxK = k; }
  RvNpuClaimPass(const RvNpuClaimPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "rvnpu-claim"; }
  StringRef getDescription() const final {
    return "Move INT8 contractions on constant weights (zero accumulator) to the NPU "
           "(risc_v_npu, CVA6 scalar core)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RvNpuDialect, linalg::LinalgDialect, arith::ArithDialect>();
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

std::unique_ptr<Pass> mlir::rvnpu::createClaimPass(int64_t maxK) {
  return std::make_unique<RvNpuClaimPass>(maxK);
}

void mlir::rvnpu::registerClaimPass() { PassRegistration<RvNpuClaimPass>(); }
