//===- lower.cpp - the risc_v_npu delegate: entry point -----===//
//
//   compile(ir/delegates/risc_v_npu.mlir), one @risc_v_npu_<id> function per
//   delegate id, in id order (the device's program index):
//     check         one argument (the activation), one claimable
//                   linalg.matmul on it, its result returned; M, K, N, the
//                   constant weight and tk = min(K, MAX_K) are read off that
//                   matmul (the NPU runs K in chunks of tk; librv_npu_rt sums
//                   them into the M x N result)
//     Pack          its weight as B tiles, into the arena
//     planProgram   its segments and data block (spec/layout.py)
//   then, once per package (issue #206):
//     LowerToRiscv  the one parameterized kernel: the job loops (scf +
//                   volatile MMIO stores), every bound read from the
//                   parameter block the driver writes at DATA_BASE
//     convertToLLVMDialect, then rv_codegen: RV32IMC, linked with crt0.S and
//     link.ld, flattened
//   and rv_programs.bin / rv_programs.txt / rv_programs.json
//
// Modelled on delegates/npu/lower.cpp. The device code (this folder)
// never includes the back end; this file is the only place the two meet.
//
//===----------------------------------------------------------------------===//

#include "lower.h"

#include "rv_codegen.h"

#include "../../package.h"

#include "RvNpu/RvNpuMatch.h"
#include "RvNpu/RvNpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/Support/FileSystem.h"

using namespace mlir;

namespace backend {
namespace {

// The arena layout id for tiled weights, as the npu delegate's: the B tile
// layout is the same matrix-core layout, so the two could share the bytes.
constexpr int kLayoutBTiles = 1;
// Written by the middle end (middle-end/linalg/transforms/Passes.h).
constexpr llvm::StringLiteral kDelegateIdAttr = "pynqo.delegate_id";

// One claimed matmul, as the device runs it: lhs [m, k] @ weight [k, n] ->
// [m, n], K in chunks of tk.
struct Gemm {
  linalg::MatmulOp matmul;
  arith::ConstantOp weight;
  int64_t m, k, n, tk;
};

// The subgraph the device runs: %y = linalg.matmul(%arg0, constant) into a
// zero accumulator, returned. The activation is the delegate's only input.
// rvnpu::isClaimable checks static 2-D INT8 operands and a static 2-D INT32
// accumulator; the shapes must also agree, and tk be positive.
FailureOr<Gemm> subgraph(func::FuncOp fn, int64_t maxK) {
  SmallVector<linalg::MatmulOp> matmuls;
  fn.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  if (fn.isExternal() || fn.getNumArguments() != 1 || matmuls.size() != 1 ||
      !rvnpu::isClaimable(matmuls.front())) {
    fn.emitError("a risc_v_npu function must take one activation and hold one claimable "
                 "linalg.matmul");
    return failure();
  }
  linalg::MatmulOp matmul = matmuls.front();
  Operation *ret = fn.front().getTerminator();
  if (matmul.getInputs()[0] != fn.getArgument(0) || ret->getNumOperands() != 1 ||
      ret->getOperand(0) != matmul->getResult(0)) {
    fn.emitError("a risc_v_npu function must return the matmul of its argument");
    return failure();
  }
  auto lhs = cast<RankedTensorType>(matmul.getInputs()[0].getType());
  auto rhs = cast<RankedTensorType>(matmul.getInputs()[1].getType());
  auto out = cast<RankedTensorType>(matmul->getResult(0).getType());
  int64_t m = lhs.getDimSize(0), k = lhs.getDimSize(1), n = rhs.getDimSize(1);
  int64_t tk = std::min<int64_t>(k, maxK);
  if (rhs.getDimSize(0) != k || out.getDimSize(0) != m || out.getDimSize(1) != n ||
      tk <= 0) {
    matmul.emitError("a risc_v_npu matmul must be [M, K] x [K, N] -> [M, N] with K and MAX_K "
                     "positive");
    return failure();
  }
  return Gemm{matmul, matmul.getInputs()[1].getDefiningOp<arith::ConstantOp>(), m, k, n, tk};
}

class RiscvNpuDelegate : public Delegate {
public:
  explicit RiscvNpuDelegate(const RiscvNpuOptions &options) : options(options) {}

  std::string name() const override { return "risc_v_npu"; }

  FailureOr<std::vector<int64_t>> compile(ModuleOp module, WeightArena &arena,
                                          const std::string &out) override {
    std::vector<std::pair<int64_t, func::FuncOp>> functions;
    for (auto fn : module.getOps<func::FuncOp>())
      if (auto id = fn->getAttrOfType<IntegerAttr>(kDelegateIdAttr))
        functions.push_back({id.getInt(), fn});
    llvm::sort(functions, [](const auto &a, const auto &b) { return a.first < b.first; });
    if (!functions.empty() && !rvCodegenAvailable()) {
      emitError(module.getLoc(), "risc_v_npu: pynqo-backend was built without LLVM's RISCV target");
      return failure();
    }

    rvnpu::Limits limits{options.rows, options.columns, options.maxK};
    std::vector<rvnpu::Program> programs;
    std::vector<int64_t> ids;
    std::string runtimeAsm = rvnpu::mmioEquates() + rvnpu::crt0Source().str();
    std::string work = out + "/risc_v_npu";
    for (auto [id, fn] : functions) {
      auto gemm = subgraph(fn, options.maxK);
      if (failed(gemm))
        return failure();
      auto data = constantBytes(gemm->weight.getValue());
      if (failed(data)) {
        gemm->matmul.emitError("risc_v_npu weight is not a dense constant");
        return failure();
      }
      // The B layout depends only on (k, n, tk): pack with m = 1 once.
      rvnpu::GemmPlan layout = rvnpu::GemmPlan::make(1, gemm->n, gemm->k, limits, gemm->tk);
      std::vector<int8_t> tiles =
          rvnpu::packB(layout, reinterpret_cast<const int8_t *>(data->data()));
      int64_t offset = arena.add(
          gemm->weight.getValue().getAsOpaquePointer(), kLayoutBTiles,
          llvm::ArrayRef<char>(reinterpret_cast<const char *>(tiles.data()), tiles.size()));
      rvnpu::GemmPlan plan =
          rvnpu::GemmPlan::make(gemm->m, gemm->n, gemm->k, limits, gemm->tk);
      int64_t index = static_cast<int64_t>(programs.size());
      programs.push_back(rvnpu::planProgram(index, plan, offset, options.segmentBytes));
      ids.push_back(id);
    }

    // The kernel does not depend on any task: compile it once, and only when
    // there is something to run (a package without risc_v_npu tasks needs no
    // RISCV target).
    RvImage kernel;
    if (!programs.empty()) {
      OwningOpRef<ModuleOp> code = rvnpu::lowerToRiscv(module.getContext());
      std::string name = "kernel";
      if (options.keepIr) {
        llvm::sys::fs::create_directories(work);
        std::error_code ec;
        llvm::raw_fd_ostream os(work + "/" + name + ".mlir", ec, llvm::sys::fs::OF_Text);
        if (!ec)
          code->print(os);
      }
      if (failed(rvnpu::convertToLLVMDialect(*code))) {
        emitError(module.getLoc(), "risc_v_npu: lowering the kernel to the LLVM dialect failed");
        return failure();
      }
      std::string error;
      if (failed(rvCodegen(*code, RvTarget{}, runtimeAsm, rvnpu::linkScript(), work, name,
                           options.llvmBin, options.keepIr, kernel, error))) {
        emitError(module.getLoc(), error);
        return failure();
      }
    }
    if (!options.keepIr)
      llvm::sys::fs::remove(work);  // only succeeds when empty
    std::string error;
    if (!rvnpu::writePrograms(out, limits, options.segmentBytes, programs, kernel.bytes,
                              kernel.memBytes, error)) {
      emitError(module.getLoc(), error);
      return failure();
    }
    return ids;
  }

private:
  RiscvNpuOptions options;
};

} // namespace

std::unique_ptr<Delegate> createRiscvNpuDelegate(const RiscvNpuOptions &options) {
  return std::make_unique<RiscvNpuDelegate>(options);
}

} // namespace backend
