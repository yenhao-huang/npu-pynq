//===- lower.cpp - the npu delegate: the adapter to src/vendors/npu_pynq --===//
//
//   compile(ir/delegates/npu.mlir), one @npu_<id> function per delegate id,
//   in id order (the NPU's program index):
//     check   one argument (the activation), one claimable linalg.matmul on
//             it, its result returned
//     Claim   npu::claim (custom_dialect/lib/Claim.cpp): npu.matmul {tk},
//             the same M x N result (libnpu_rt sums the K chunks)
//     Pack    its weight as B tiles, into the arena
//     Encode  its ISA program, relocated against that block
//   then programs.bin / programs.txt / programs.json
//
// The vendor never includes the back end; this file is the only place the
// two meet.
//
//===----------------------------------------------------------------------===//

#include "lower.h"

#include "../../package.h"

#include "Npu/NpuDialect.h"
#include "Npu/NpuMatch.h"
#include "Npu/NpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace backend {
namespace {

// The arena layout id for tiled weights. Dense weights use 0.
// A weight that the CPU code also reads is therefore stored in both forms.
// The key does not need tk, because K and maxK fix it for a given constant.
constexpr int kLayoutBTiles = 1;
// Written by the middle end (middle-end/linalg/transforms/Passes.h).
constexpr llvm::StringLiteral kDelegateIdAttr = "pynqo.delegate_id";

// The subgraph the NPU runs: %y = linalg.matmul(%arg0, constant) into a zero
// accumulator, returned. The activation is the delegate's only input.
FailureOr<linalg::MatmulOp> subgraph(func::FuncOp fn) {
  SmallVector<linalg::MatmulOp> matmuls;
  fn.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  if (fn.isExternal() || fn.getNumArguments() != 1 || matmuls.size() != 1 ||
      !npu::isClaimable(matmuls.front())) {
    fn.emitError("an npu function must take one activation and hold one claimable linalg.matmul");
    return failure();
  }
  linalg::MatmulOp matmul = matmuls.front();
  Operation *ret = fn.front().getTerminator();
  if (matmul.getInputs()[0] != fn.getArgument(0) || ret->getNumOperands() != 1 ||
      ret->getOperand(0) != matmul->getResult(0)) {
    fn.emitError("an npu function must return the matmul of its argument");
    return failure();
  }
  return matmul;
}

class NpuDelegate : public Delegate {
public:
  explicit NpuDelegate(const NpuOptions &options) : options(options) {}

  std::string name() const override { return "npu"; }

  FailureOr<std::vector<int64_t>> compile(ModuleOp module, WeightArena &arena,
                                          const std::string &out) override {
    std::vector<std::pair<int64_t, func::FuncOp>> functions;
    for (auto fn : module.getOps<func::FuncOp>())
      if (auto id = fn->getAttrOfType<IntegerAttr>(kDelegateIdAttr))
        functions.push_back({id.getInt(), fn});
    llvm::sort(functions, [](const auto &a, const auto &b) { return a.first < b.first; });

    npu::Limits limits{options.rows, options.columns, options.maxK};
    std::vector<uint64_t> words;
    std::vector<npu::Program> programs;
    std::vector<int64_t> ids;
    for (auto [id, fn] : functions) {
      auto matmul = subgraph(fn);
      if (failed(matmul))
        return failure();
      npu::MatmulOp op = npu::claim(*matmul, options.maxK);
      auto constant = op.getRhs().getDefiningOp<arith::ConstantOp>();
      auto data = constantBytes(constant.getValue());
      if (failed(data)) {
        op.emitError("npu weight is not a dense constant");
        return failure();
      }
      auto lhs = op.getLhs().getType();
      auto rhs = op.getRhs().getType();
      int64_t m = lhs.getDimSize(0), k = lhs.getDimSize(1), n = rhs.getDimSize(1);
      // The B layout depends only on (k, n, tk): pack with m = 1 once.
      npu::GemmPlan layout = npu::GemmPlan::make(1, n, k, limits, op.getTk());
      std::vector<int8_t> tiles =
          npu::packB(layout, reinterpret_cast<const int8_t *>(data->data()));
      int64_t offset = arena.add(
          constant.getValue().getAsOpaquePointer(), kLayoutBTiles,
          llvm::ArrayRef<char>(reinterpret_cast<const char *>(tiles.data()), tiles.size()));
      npu::GemmPlan plan = npu::GemmPlan::make(m, n, k, limits, op.getTk());
      int64_t index = static_cast<int64_t>(programs.size());
      programs.push_back(npu::encode(index, plan, offset, options.segmentBytes, words));
      ids.push_back(id);
    }
    std::string error;
    if (!npu::writePrograms(out, limits, options.segmentBytes, words, programs, error)) {
      emitError(module.getLoc(), error);
      return failure();
    }
    return ids;
  }

private:
  NpuOptions options;
};

} // namespace

std::unique_ptr<Delegate> createNpuDelegate(const NpuOptions &options) {
  return std::make_unique<NpuDelegate>(options);
}

} // namespace backend
