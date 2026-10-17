//===- Partition.cpp - split a model between the CPU and the NPU ----------===//
//
// --npu-partition rewrites every INT8 x INT8 -> INT32 contraction whose right
// operand is an npu.weight into an npu.matmul (an NPU task):
//
//   linalg.matmul ins(%x, %w) outs(%acc)       ->  npu.matmul + linalg.reduce
//   linalg.generic {npu.group_size = g} (C[c,m,n] += X[m,c*g+j] * W[c*g+j,n])
//                                              ->  npu.matmul {tk = g} + add
//
// Everything else stays in upstream dialects and becomes Cortex-A9 code. The
// report option writes which weights the NPU reads (so the exporter stores
// them as B tiles) and what was left on the CPU.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuDialect.h"
#include "Npu/NpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace mlir;
using namespace mlir::npu;

namespace {

bool isIntTensor(Value v, unsigned width) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.hasStaticShape() && t.getElementType().isInteger(width);
}

struct NpuTask {
  std::string weight;
  int64_t m, n, k, tk;
  std::string source;
};

struct PartitionPass
    : public PassWrapper<PartitionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionPass)

  PartitionPass() = default;
  PartitionPass(const PartitionPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "npu-partition"; }
  StringRef getDescription() const final {
    return "Move INT8 contractions on weights to the NPU";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NpuDialect, linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect>();
  }

  Option<std::string> report{*this, "report",
                             llvm::cl::desc("Write the partition as JSON")};
  Option<int64_t> maxK{*this, "max-k", llvm::cl::desc("NPU MAX_K"),
                       llvm::cl::init(256)};

  std::vector<NpuTask> tasks;

  // C[M,N] (+)= X[M,K] @ W[K,N] with W a weight.
  LogicalResult rewriteMatmul(linalg::MatmulOp op) {
    if (op.hasUserDefinedMaps() || op.getInputs().size() != 2)
      return failure();
    Value x = op.getInputs()[0], w = op.getInputs()[1];
    Value acc = op.getOutputs()[0];
    if (!isIntTensor(x, 8) || !isIntTensor(w, 8) || !isIntTensor(acc, 32))
      return failure();
    auto weight = w.getDefiningOp<WeightOp>();
    if (!weight)
      return failure();
    auto xt = cast<RankedTensorType>(x.getType());
    auto wt = cast<RankedTensorType>(w.getType());
    int64_t m = xt.getDimSize(0), k = xt.getDimSize(1), n = wt.getDimSize(1);
    int64_t tk = std::min<int64_t>(k, maxK);
    int64_t kt = (k + tk - 1) / tk;

    OpBuilder b(op);
    Location loc = op.getLoc();
    auto i32 = b.getI32Type();
    auto partialsType = RankedTensorType::get({kt, m, n}, i32);
    auto npu = MatmulOp::create(b, loc, partialsType, x, w,
                                b.getI64IntegerAttr(tk));
    auto reduce = linalg::ReduceOp::create(
        b, loc, ValueRange{npu.getResult()}, ValueRange{acc},
        ArrayRef<int64_t>{0},
        [](OpBuilder &nb, Location nl, ValueRange args) {
          Value sum = arith::AddIOp::create(nb, nl, args[0], args[1]);
          linalg::YieldOp::create(nb, nl, sum);
        });
    op->getResult(0).replaceAllUsesWith(reduce->getResult(0));
    op->erase();
    tasks.push_back({weight.getName().str(), m, n, k, tk, "linalg.matmul"});
    return success();
  }

  // C[c,m,n] (+)= sum_j X[m,c,j] * W[c,j,n]: group-wise products that the
  // CPU scales per group (INT4 weights with one scale per K group). X and W
  // are the [M,K] activations and [K,N] weight viewed as [M,G,g] and [G,g,N].
  LogicalResult rewriteGrouped(linalg::GenericOp op) {
    auto groupAttr = op->getAttrOfType<IntegerAttr>("npu.group_size");
    if (!groupAttr || op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return failure();
    Value x3 = op.getDpsInputs()[0], w3 = op.getDpsInputs()[1];
    Value acc = op.getDpsInits()[0];
    if (!isIntTensor(x3, 8) || !isIntTensor(w3, 8) || !isIntTensor(acc, 32))
      return op.emitError("npu.group_size contraction needs i8 x i8 -> i32");
    auto weight = w3.getDefiningOp<WeightOp>();
    if (!weight)
      return op.emitError("npu.group_size contraction needs a weight rhs");
    auto xt = cast<RankedTensorType>(x3.getType());
    auto wt = cast<RankedTensorType>(w3.getType());
    auto ct = cast<RankedTensorType>(acc.getType());
    int64_t g = groupAttr.getInt();
    if (xt.getRank() != 3 || wt.getRank() != 3 || ct.getRank() != 3)
      return op.emitError("npu.group_size contraction must be rank 3");
    int64_t m = xt.getDimSize(0), groups = xt.getDimSize(1), n = wt.getDimSize(2);
    if (xt.getDimSize(2) != g || wt.getDimSize(0) != groups ||
        wt.getDimSize(1) != g || ct.getDimSize(0) != groups ||
        ct.getDimSize(1) != m || ct.getDimSize(2) != n || g <= 0 || g > maxK)
      return op.emitError("npu.group_size contraction has an unexpected shape");

    MLIRContext *ctx = op.getContext();
    AffineExpr c, mm, nn, j;
    bindDims(ctx, c, mm, nn, j);
    SmallVector<AffineMap> expected = {AffineMap::get(4, 0, {mm, c, j}, ctx),
                                       AffineMap::get(4, 0, {c, j, nn}, ctx),
                                       AffineMap::get(4, 0, {c, mm, nn}, ctx)};
    if (op.getIndexingMapsArray() != ArrayRef<AffineMap>(expected))
      return op.emitError("npu.group_size contraction has unexpected maps");

    OpBuilder b(op);
    Location loc = op.getLoc();
    int64_t k = groups * g;
    auto i8 = b.getIntegerType(8);
    Value x = tensor::CollapseShapeOp::create(
        b, loc, RankedTensorType::get({m, k}, i8), x3,
        SmallVector<ReassociationIndices>{{0}, {1, 2}});
    Value w = WeightOp::create(b, loc, RankedTensorType::get({k, n}, i8),
                               weight.getArena(), weight.getNameAttr());
    auto npu = MatmulOp::create(b, loc, ct, x, w, b.getI64IntegerAttr(g));
    auto add = linalg::AddOp::create(b, loc, ValueRange{npu.getResult(), acc},
                                     ValueRange{acc});
    op->getResult(0).replaceAllUsesWith(add->getResult(0));
    op->erase();
    std::string name = weight.getName().str();
    if (weight->use_empty())
      weight->erase();
    tasks.push_back({name, m, n, k, g, "grouped"});
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    tasks.clear();
    SmallVector<linalg::MatmulOp> matmuls;
    SmallVector<linalg::GenericOp> grouped;
    module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    module.walk([&](linalg::GenericOp op) {
      if (op->hasAttr("npu.group_size"))
        grouped.push_back(op);
    });
    for (auto op : matmuls)
      (void)rewriteMatmul(op);
    for (auto op : grouped)
      if (failed(rewriteGrouped(op)))
        return signalPassFailure();

    if (report.empty())
      return;
    std::map<std::string, int64_t> cpuOps;
    module.walk([&](Operation *op) {
      if (isa<linalg::LinalgOp>(op))
        cpuOps[op->getName().getStringRef().str()] += 1;
    });
    std::error_code ec;
    llvm::raw_fd_ostream os(report, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError("cannot write ") << report << ": " << ec.message();
      return signalPassFailure();
    }
    llvm::json::OStream j(os, 2);
    j.object([&] {
      j.attributeArray("npu_tasks", [&] {
        for (const auto &t : tasks)
          j.object([&] {
            j.attribute("weight", t.weight);
            j.attribute("m", t.m);
            j.attribute("n", t.n);
            j.attribute("k", t.k);
            j.attribute("tk", t.tk);
            j.attribute("source", t.source);
          });
      });
      j.attributeObject("cpu_linalg_ops", [&] {
        for (const auto &[name, count] : cpuOps)
          j.attribute(name, count);
      });
    });
  }
};

} // namespace

void mlir::npu::registerPartitionPass() { PassRegistration<PartitionPass>(); }
