//===- lower.cpp - the npu delegate: npu entry point ----------===//
//
//   compile(ir/delegates/npu.mlir), one @npu_<id> function per delegate id,
//   in id order (the NPU's program index):
//     check   one argument (the activation), one claimable linalg.matmul on
//             it, its result returned; M, K, N, the constant weight and
//             tk = min(K, MAX_K) are read off that matmul (the NPU runs K in
//             chunks of tk; libnpu_rt sums them into the M x N result)
//     Pack    its weight as B tiles, into the arena
//     Encode  its ISA program, relocated against that block
//   then programs.bin / programs.txt / programs.json
//
// The device code (this folder) never includes the back end; this file is
// the only place the two meet.
//
//===----------------------------------------------------------------------===//

#include "lower.h"

#include "../../package.h"

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

// One claimed matmul, as the NPU runs it: lhs [m, k] @ weight [k, n] ->
// [m, n], K in chunks of tk.
struct Gemm {
  linalg::MatmulOp matmul;
  arith::ConstantOp weight;
  int64_t m, k, n, tk;
};

// The subgraph the NPU runs: %y = linalg.matmul(%arg0, constant) into a zero
// accumulator, returned. The activation is the delegate's only input.
// npu::isClaimable checks static 2-D INT8 operands and a static 2-D INT32
// accumulator; the shapes must also agree, and tk be positive.
FailureOr<Gemm> subgraph(func::FuncOp fn, int64_t maxK) {
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
  auto lhs = cast<RankedTensorType>(matmul.getInputs()[0].getType());
  auto rhs = cast<RankedTensorType>(matmul.getInputs()[1].getType());
  auto out = cast<RankedTensorType>(matmul->getResult(0).getType());
  int64_t m = lhs.getDimSize(0), k = lhs.getDimSize(1), n = rhs.getDimSize(1);
  int64_t tk = std::min<int64_t>(k, maxK);
  if (rhs.getDimSize(0) != k || out.getDimSize(0) != m || out.getDimSize(1) != n ||
      tk <= 0) {
    matmul.emitError("an npu matmul must be [M, K] x [K, N] -> [M, N] with K and MAX_K "
                     "positive");
    return failure();
  }
  return Gemm{matmul, matmul.getInputs()[1].getDefiningOp<arith::ConstantOp>(), m, k, n, tk};
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
      auto gemm = subgraph(fn, options.maxK);
      if (failed(gemm))
        return failure();
      auto data = constantBytes(gemm->weight.getValue());
      if (failed(data)) {
        gemm->matmul.emitError("npu weight is not a dense constant");
        return failure();
      }
      // The B layout depends only on (k, n, tk): pack with m = 1 once.
      npu::GemmPlan layout = npu::GemmPlan::make(1, gemm->n, gemm->k, limits, gemm->tk);
      std::vector<int8_t> tiles =
          npu::packB(layout, reinterpret_cast<const int8_t *>(data->data()));
      int64_t offset = arena.add(
          gemm->weight.getValue().getAsOpaquePointer(), kLayoutBTiles,
          llvm::ArrayRef<char>(reinterpret_cast<const char *>(tiles.data()), tiles.size()));
      npu::GemmPlan plan =
          npu::GemmPlan::make(gemm->m, gemm->n, gemm->k, limits, gemm->tk);
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
