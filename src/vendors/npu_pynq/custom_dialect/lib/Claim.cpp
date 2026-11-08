//===- Claim.cpp - choose the contractions the NPU runs -------------------===//
//
// --npu-claim rewrites every INT8 x INT8 -> INT32 contraction on a constant
// weight into an NPU contraction:
//
//   linalg.matmul ins(%x, %w = arith.constant) outs(%acc)
//       ->  %p = npu.matmul %x, %w {tk}  +  linalg.reduce (sum of K planes) into %acc
//
// Everything else stays in upstream dialects and becomes CPU code. The back
// end's partition then outlines each npu.matmul as a delegate call.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuDialect.h"
#include "Npu/NpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::npu;

namespace {

bool isIntMatrix(Value v, unsigned width) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.hasStaticShape() && t.getRank() == 2 &&
         t.getElementType().isInteger(width);
}

struct ClaimPass : public PassWrapper<ClaimPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClaimPass)

  ClaimPass() = default;
  ClaimPass(int64_t k) { maxK = k; }
  ClaimPass(const ClaimPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "npu-claim"; }
  StringRef getDescription() const final {
    return "Move INT8 contractions on constant weights to the NPU";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NpuDialect, linalg::LinalgDialect, arith::ArithDialect>();
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
      auto npu = MatmulOp::create(b, loc, partials, x, w, b.getI64IntegerAttr(tk));
      auto reduce = linalg::ReduceOp::create(
          b, loc, ValueRange{npu.getResult()}, ValueRange{acc},
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

std::unique_ptr<Pass> mlir::npu::createClaimPass(int64_t maxK) {
  return std::make_unique<ClaimPass>(maxK);
}

void mlir::npu::registerClaimPass() { PassRegistration<ClaimPass>(); }
