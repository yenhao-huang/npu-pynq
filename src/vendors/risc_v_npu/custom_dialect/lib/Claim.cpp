//===- Claim.cpp - choose the contractions the NPU runs (risc_v_npu) -----===//
//
// Copied from src/vendors/npu_pynq/custom_dialect/lib/Claim.cpp and renamed to
// the rvnpu dialect (mlir::rvnpu, --rvnpu-claim), so both devices link into one
// pynqo-backend; keep the two in step (plan 2026-1112-003 section 6.2).
//
// --rvnpu-claim rewrites every INT8 x INT8 -> INT32 contraction on a constant
// weight into an NPU contraction:
//
//   linalg.matmul ins(%x, %w = arith.constant) outs(%acc)
//       ->  %p = rvnpu.matmul %x, %w {tk}  +  linalg.reduce (sum of K planes) into %acc
//
// Everything else stays in upstream dialects and becomes CPU code. The back
// end's partition then outlines each rvnpu.matmul as a delegate call.
//
//===----------------------------------------------------------------------===//

#include "RvNpu/RvNpuDialect.h"
#include "RvNpu/RvNpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::rvnpu;

namespace {

bool isIntMatrix(Value v, unsigned width) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.hasStaticShape() && t.getRank() == 2 &&
         t.getElementType().isInteger(width);
}

struct RvNpuClaimPass : public PassWrapper<RvNpuClaimPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RvNpuClaimPass)

  RvNpuClaimPass() = default;
  RvNpuClaimPass(int64_t k) { maxK = k; }
  RvNpuClaimPass(const RvNpuClaimPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "rvnpu-claim"; }
  StringRef getDescription() const final {
    return "Move INT8 contractions on constant weights to the NPU (risc_v_npu, CVA6 scalar core)";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RvNpuDialect, linalg::LinalgDialect, arith::ArithDialect>();
  }

  Option<int64_t> maxK{*this, "max-k", llvm::cl::desc("NPU MAX_K"),
                       llvm::cl::init(256)};

  void runOnOperation() override {
    SmallVector<linalg::MatmulOp> matmuls;
    getOperation().walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    for (linalg::MatmulOp op : matmuls) {
      if (op.hasUserDefinedMaps() || op.getInputs().size() != 2)
        continue;
      Value x = op.getInputs()[0], w = op.getInputs()[1];
      Value acc = op.getOutputs()[0];
      if (!isIntMatrix(x, 8) || !isIntMatrix(w, 8) || !isIntMatrix(acc, 32) ||
          !w.getDefiningOp<arith::ConstantOp>())
        continue;
      auto xt = cast<RankedTensorType>(x.getType());
      auto wt = cast<RankedTensorType>(w.getType());
      int64_t m = xt.getDimSize(0), k = xt.getDimSize(1), n = wt.getDimSize(1);
      int64_t tk = std::min<int64_t>(k, maxK);
      int64_t kt = (k + tk - 1) / tk;
      OpBuilder b(op);
      Location loc = op.getLoc();
      auto partials = RankedTensorType::get({kt, m, n}, b.getI32Type());
      auto matmul = MatmulOp::create(b, loc, partials, x, w, b.getI64IntegerAttr(tk));
      auto reduce = linalg::ReduceOp::create(
          b, loc, ValueRange{matmul.getResult()}, ValueRange{acc},
          ArrayRef<int64_t>{0}, [](OpBuilder &nb, Location nl, ValueRange args) {
            Value sum = arith::AddIOp::create(nb, nl, args[0], args[1]);
            linalg::YieldOp::create(nb, nl, sum);
          });
      op->getResult(0).replaceAllUsesWith(reduce->getResult(0));
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::rvnpu::createClaimPass(int64_t maxK) {
  return std::make_unique<RvNpuClaimPass>(maxK);
}

void mlir::rvnpu::registerClaimPass() { PassRegistration<RvNpuClaimPass>(); }
