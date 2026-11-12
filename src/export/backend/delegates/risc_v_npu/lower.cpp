//===- lower.cpp - the risc_v_npu delegate: the adapter to src/vendors/risc_v_npu ===//
//
//   claims()   --rvnpu-claim (custom_dialect/lib/Claim.cpp): INT8
//              contractions on constant weights become rvnpu.matmul
//   compile()  per rvnpu.matmul, in delegate-id order:
//                Pack          its weight as B tiles, into the arena
//                planProgram   its segments and data block (spec/layout.py)
//                per segment:
//                  LowerToRiscv  the job loops (scf + volatile MMIO stores)
//                  convertToLLVMDialect, then rv_codegen: RV32IMC, linked
//                  with crt0.S and link.ld, flattened
//              then rv_programs.bin / rv_programs.txt / rv_programs.json
//
// Modelled on delegates/npu/lower.cpp. The vendor never includes the back
// end; this file is the only place the two meet.
//
//===----------------------------------------------------------------------===//

#include "lower.h"

#include "rv_codegen.h"

#include "../../package.h"

#include "RvNpu/RvNpuDialect.h"
#include "RvNpu/RvNpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/FileSystem.h"

using namespace mlir;

namespace backend {
namespace {

// The arena layout id for tiled weights, as the npu delegate's: the B tile
// layout is the same matrix-core layout, so the two could share the bytes.
constexpr int kLayoutBTiles = 1;

class RiscvNpuDelegate : public Delegate {
public:
  explicit RiscvNpuDelegate(const RiscvNpuOptions &options) : options(options) {}

  std::string name() const override { return "risc_v_npu"; }

  FailureOr<std::vector<Operation *>> claims(ModuleOp module) override {
    PassManager pm(module.getContext());
    pm.addPass(rvnpu::createClaimPass(options.maxK));
    if (failed(pm.run(module)))
      return failure();
    std::vector<Operation *> ops;
    module.walk([&](rvnpu::MatmulOp op) { ops.push_back(op); });
    return ops;
  }

  LogicalResult compile(const std::vector<std::pair<int64_t, Operation *>> &ops,
                        WeightArena &arena, const std::string &out) override {
    rvnpu::Limits limits{options.rows, options.columns, options.maxK};
    std::vector<rvnpu::Program> programs;
    std::vector<std::vector<std::vector<uint8_t>>> binaries;
    std::vector<std::vector<int64_t>> memBytes;
    std::string runtimeAsm = rvnpu::mmioEquates() + rvnpu::crt0Source().str();
    std::string work = out + "/risc_v_npu";
    if (!ops.empty() && !rvCodegenAvailable())
      return ops.front().second->emitError(
          "risc_v_npu: pynqo-backend was built without LLVM's RISCV target");
    for (auto [id, op] : ops) {
      auto matmul = cast<rvnpu::MatmulOp>(op);
      auto constant = matmul.getRhs().getDefiningOp<arith::ConstantOp>();
      auto data = constantBytes(constant.getValue());
      if (failed(data))
        return matmul.emitError("risc_v_npu weight is not a dense constant");
      auto lhs = matmul.getLhs().getType();
      auto rhs = matmul.getRhs().getType();
      int64_t m = lhs.getDimSize(0), k = lhs.getDimSize(1), n = rhs.getDimSize(1);
      // The B layout depends only on (k, n, tk): pack with m = 1 once.
      rvnpu::GemmPlan layout = rvnpu::GemmPlan::make(1, n, k, limits, matmul.getTk());
      std::vector<int8_t> tiles =
          rvnpu::packB(layout, reinterpret_cast<const int8_t *>(data->data()));
      int64_t offset = arena.add(
          constant.getValue().getAsOpaquePointer(), kLayoutBTiles,
          llvm::ArrayRef<char>(reinterpret_cast<const char *>(tiles.data()), tiles.size()));
      rvnpu::GemmPlan plan = rvnpu::GemmPlan::make(m, n, k, limits, matmul.getTk());
      if (static_cast<int64_t>(programs.size()) != id)
        return matmul.emitError("risc_v_npu delegate ids must be 0..n-1 in order");
      rvnpu::Program program =
          rvnpu::planProgram(id, plan, offset, options.segmentBytes);

      std::vector<std::vector<uint8_t>> images;
      std::vector<int64_t> sizes;
      for (int64_t s = 0; s < static_cast<int64_t>(program.segments.size()); ++s) {
        OwningOpRef<ModuleOp> code = rvnpu::lowerToRiscv(op->getContext(), program, s);
        std::string name = "program_" + std::to_string(id) + "_" + std::to_string(s);
        if (options.keepIr) {
          llvm::sys::fs::create_directories(work);
          std::error_code ec;
          llvm::raw_fd_ostream os(work + "/" + name + ".mlir", ec, llvm::sys::fs::OF_Text);
          if (!ec)
            code->print(os);
        }
        if (failed(rvnpu::convertToLLVMDialect(*code)))
          return matmul.emitError("risc_v_npu: lowering the program to the LLVM dialect failed");
        RvImage image;
        std::string error;
        if (failed(rvCodegen(*code, RvTarget{}, runtimeAsm, rvnpu::linkScript(), work, name,
                             options.llvmBin, options.keepIr, image, error)))
          return matmul.emitError(error);
        images.push_back(std::move(image.bytes));
        sizes.push_back(image.memBytes);
      }
      programs.push_back(std::move(program));
      binaries.push_back(std::move(images));
      memBytes.push_back(std::move(sizes));
    }
    if (!options.keepIr)
      llvm::sys::fs::remove(work);  // only succeeds when empty
    std::string error;
    if (!rvnpu::writePrograms(out, limits, options.segmentBytes, programs, binaries,
                              memBytes, error))
      return emitError(UnknownLoc::get(ops.empty() ? nullptr : ops.front().second->getContext()), error);
    return success();
  }

private:
  RiscvNpuOptions options;
};

} // namespace

std::unique_ptr<Delegate> createRiscvNpuDelegate(const RiscvNpuOptions &options) {
  return std::make_unique<RiscvNpuDelegate>(options);
}

} // namespace backend
