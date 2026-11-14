//===- lower.cpp - the risc_v_npu delegate: the adapter to src/vendors/risc_v_npu ===//
//
//   compile(ir/delegates/risc_v_npu.mlir), one @risc_v_npu_<id> function per
//   delegate id, in id order (the device's program index):
//     check         one argument (the activation), one claimable
//                   linalg.matmul on it, its result returned
//     Claim         rvnpu::claim (custom_dialect/lib/Claim.cpp): rvnpu.matmul
//                   {tk}, the same M x N result (librv_npu_rt sums the K
//                   chunks)
//     Pack          its weight as B tiles, into the arena
//     planProgram   its segments and data block (spec/layout.py)
//     per segment:
//       LowerToRiscv  the job loops (scf + volatile MMIO stores)
//       convertToLLVMDialect, then rv_codegen: RV32IMC, linked with crt0.S
//       and link.ld, flattened
//   then rv_programs.bin / rv_programs.txt / rv_programs.json
//
// Modelled on delegates/npu/lower.cpp. The vendor never includes the back
// end; this file is the only place the two meet.
//
//===----------------------------------------------------------------------===//

#include "lower.h"

#include "rv_codegen.h"

#include "../../package.h"

#include "RvNpu/RvNpuDialect.h"
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

// The subgraph the device runs: %y = linalg.matmul(%arg0, constant) into a
// zero accumulator, returned. The activation is the delegate's only input.
FailureOr<linalg::MatmulOp> subgraph(func::FuncOp fn) {
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
  return matmul;
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
    std::vector<std::vector<std::vector<uint8_t>>> binaries;
    std::vector<std::vector<int64_t>> memBytes;
    std::vector<int64_t> ids;
    std::string runtimeAsm = rvnpu::mmioEquates() + rvnpu::crt0Source().str();
    std::string work = out + "/risc_v_npu";
    for (auto [id, fn] : functions) {
      auto found = subgraph(fn);
      if (failed(found))
        return failure();
      rvnpu::MatmulOp op = rvnpu::claim(*found, options.maxK);
      auto constant = op.getRhs().getDefiningOp<arith::ConstantOp>();
      auto data = constantBytes(constant.getValue());
      if (failed(data)) {
        op.emitError("risc_v_npu weight is not a dense constant");
        return failure();
      }
      auto lhs = op.getLhs().getType();
      auto rhs = op.getRhs().getType();
      int64_t m = lhs.getDimSize(0), k = lhs.getDimSize(1), n = rhs.getDimSize(1);
      // The B layout depends only on (k, n, tk): pack with m = 1 once.
      rvnpu::GemmPlan layout = rvnpu::GemmPlan::make(1, n, k, limits, op.getTk());
      std::vector<int8_t> tiles =
          rvnpu::packB(layout, reinterpret_cast<const int8_t *>(data->data()));
      int64_t offset = arena.add(
          constant.getValue().getAsOpaquePointer(), kLayoutBTiles,
          llvm::ArrayRef<char>(reinterpret_cast<const char *>(tiles.data()), tiles.size()));
      rvnpu::GemmPlan plan = rvnpu::GemmPlan::make(m, n, k, limits, op.getTk());
      int64_t index = static_cast<int64_t>(programs.size());
      rvnpu::Program program = rvnpu::planProgram(index, plan, offset, options.segmentBytes);

      std::vector<std::vector<uint8_t>> images;
      std::vector<int64_t> sizes;
      for (int64_t s = 0; s < static_cast<int64_t>(program.segments.size()); ++s) {
        OwningOpRef<ModuleOp> code = rvnpu::lowerToRiscv(module.getContext(), program, s);
        std::string name = "program_" + std::to_string(index) + "_" + std::to_string(s);
        if (options.keepIr) {
          llvm::sys::fs::create_directories(work);
          std::error_code ec;
          llvm::raw_fd_ostream os(work + "/" + name + ".mlir", ec, llvm::sys::fs::OF_Text);
          if (!ec)
            code->print(os);
        }
        if (failed(rvnpu::convertToLLVMDialect(*code))) {
          op.emitError("risc_v_npu: lowering the program to the LLVM dialect failed");
          return failure();
        }
        RvImage image;
        std::string error;
        if (failed(rvCodegen(*code, RvTarget{}, runtimeAsm, rvnpu::linkScript(), work, name,
                             options.llvmBin, options.keepIr, image, error))) {
          op.emitError(error);
          return failure();
        }
        images.push_back(std::move(image.bytes));
        sizes.push_back(image.memBytes);
      }
      programs.push_back(std::move(program));
      binaries.push_back(std::move(images));
      memBytes.push_back(std::move(sizes));
      ids.push_back(id);
    }
    if (!options.keepIr)
      llvm::sys::fs::remove(work);  // only succeeds when empty
    std::string error;
    if (!rvnpu::writePrograms(out, limits, options.segmentBytes, programs, binaries, memBytes,
                              error)) {
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
