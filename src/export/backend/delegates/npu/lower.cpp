//===- lower.cpp - the npu delegate: the adapter to src/vendors/npu_pynq --===//
//
//   lower()    npu::claim (custom_dialect/lib/Claim.cpp) on the device
//              function's matmul: npu.matmul {tk} + linalg.reduce of its
//              K planes, which stays CPU code
//   compile()  per npu.matmul, in delegate-id order:
//                Pack   its weight as B tiles, into the arena
//                Encode its ISA program, relocated against that block
//              then programs.bin / programs.txt / programs.json
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace backend {
namespace {

// The arena layout id for tiled weights. Dense weights use 0.
// A weight that the CPU code also reads is therefore stored in both forms.
// The key does not need tk, because K and maxK fix it for a given constant.
constexpr int kLayoutBTiles = 1;

class NpuDelegate : public Delegate {
public:
  explicit NpuDelegate(const NpuOptions &options) : options(options) {}

  std::string name() const override { return "npu"; }

  FailureOr<Operation *> lower(func::FuncOp fn) override {
    SmallVector<linalg::MatmulOp> matmuls;
    fn.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    if (matmuls.size() != 1 || !npu::isClaimable(matmuls.front())) {
      fn.emitError("an npu device function must hold one claimable linalg.matmul");
      return failure();
    }
    return npu::claim(matmuls.front(), options.maxK).getOperation();
  }

  LogicalResult compile(const std::vector<std::pair<int64_t, Operation *>> &ops,
                        WeightArena &arena, const std::string &out) override {
    npu::Limits limits{options.rows, options.columns, options.maxK};
    std::vector<uint64_t> words;
    std::vector<npu::Program> programs;
    for (auto [id, op] : ops) {
      auto matmul = cast<npu::MatmulOp>(op);
      auto constant = matmul.getRhs().getDefiningOp<arith::ConstantOp>();
      auto data = constantBytes(constant.getValue());
      if (failed(data))
        return matmul.emitError("npu weight is not a dense constant");
      auto lhs = matmul.getLhs().getType();
      auto rhs = matmul.getRhs().getType();
      int64_t m = lhs.getDimSize(0), k = lhs.getDimSize(1), n = rhs.getDimSize(1);
      // The B layout depends only on (k, n, tk): pack with m = 1 once.
      npu::GemmPlan layout = npu::GemmPlan::make(1, n, k, limits, matmul.getTk());
      std::vector<int8_t> tiles =
          npu::packB(layout, reinterpret_cast<const int8_t *>(data->data()));
      int64_t offset = arena.add(
          constant.getValue().getAsOpaquePointer(), kLayoutBTiles,
          llvm::ArrayRef<char>(reinterpret_cast<const char *>(tiles.data()), tiles.size()));
      npu::GemmPlan plan = npu::GemmPlan::make(m, n, k, limits, matmul.getTk());
      if (static_cast<int64_t>(programs.size()) != id)
        return matmul.emitError("npu delegate ids must be 0..n-1 in order");
      programs.push_back(npu::encode(id, plan, offset, options.segmentBytes, words));
    }
    std::string error;
    if (!npu::writePrograms(out, limits, options.segmentBytes, words, programs, error))
      return emitError(UnknownLoc::get(ops.empty() ? nullptr : ops.front().second->getContext()), error);
    return success();
  }

private:
  NpuOptions options;
};

} // namespace

std::unique_ptr<Delegate> createNpuDelegate(const NpuOptions &options) {
  return std::make_unique<NpuDelegate>(options);
}

} // namespace backend
