//===- Lower.cpp - NPU tasks to ISA programs and runtime calls ------------===//
//
// --npu-lower finishes the NPU side of compilation:
//
//   * npu.weight becomes a view of the weight arena at the offset the
//     exporter chose (weights=<table.json>, {"name": byte offset}).
//   * every npu.matmul is planned into tiles and encoded into an ISA program
//     (the "NPU instruction encoder"), written to programs=<prefix>.bin with
//     its plan and relocations in <prefix>.json, and replaced by
//
//         %a = memref.alloc   <- the task's INT8 left operand
//         call @npu_rt_gemm(program id, %a, %partials)
//         %partials as a tensor
//
// The runtime packs %a into A tiles, starts the program, and unpacks the
// INT32 partial planes. ADDR_B words hold offsets within the task's weight
// block; the loader relocates them to wherever the block sits in DDR.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuDialect.h"
#include "Npu/NpuEncoder.h"
#include "Npu/NpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace mlir;
using namespace mlir::npu;

namespace {

constexpr StringLiteral kGemmFn = "npu_rt_gemm";

struct Program {
  int64_t id, wordOffset, length;
  GemmPlan plan;
  std::string weight;
  int64_t weightOffset;
  std::vector<int64_t> relocations;
};

struct LowerPass : public PassWrapper<LowerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPass)

  LowerPass() = default;
  LowerPass(const LowerPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "npu-lower"; }
  StringRef getDescription() const final {
    return "Encode NPU tasks as ISA programs and bind weights to the arena";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  Option<std::string> weights{*this, "weights",
                              llvm::cl::desc("Weight offset table (JSON)")};
  Option<std::string> programs{*this, "programs",
                               llvm::cl::desc("Output prefix for programs")};
  Option<int64_t> rows{*this, "rows", llvm::cl::init(16)};
  Option<int64_t> columns{*this, "columns", llvm::cl::init(16)};
  Option<int64_t> maxK{*this, "max-k", llvm::cl::init(256)};

  std::map<std::string, int64_t> offsets;

  LogicalResult loadWeights(ModuleOp module) {
    auto buffer = llvm::MemoryBuffer::getFile(weights);
    if (!buffer)
      return module.emitError("cannot read weight table ") << weights;
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed)
      return module.emitError("weight table is not JSON: ")
             << llvm::toString(parsed.takeError());
    auto *object = parsed->getAsObject();
    if (!object)
      return module.emitError("weight table must be an object");
    for (auto &entry : *object) {
      auto value = entry.second.getAsInteger();
      if (!value)
        return module.emitError("weight offset is not an integer: ")
               << entry.first.str();
      offsets[entry.first.str()] = *value;
    }
    return success();
  }

  func::FuncOp declareRuntime(ModuleOp module) {
    if (auto fn = module.lookupSymbol<func::FuncOp>(kGemmFn))
      return fn;
    OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
    MLIRContext *ctx = module.getContext();
    auto i8buf = MemRefType::get({ShapedType::kDynamic}, IntegerType::get(ctx, 8));
    auto i32buf = MemRefType::get({ShapedType::kDynamic}, IntegerType::get(ctx, 32));
    auto type = b.getFunctionType({b.getI32Type(), i8buf, i32buf}, {});
    auto fn = func::FuncOp::create(b, module.getLoc(), kGemmFn, type);
    fn.setPrivate();
    fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    return fn;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(loadWeights(module)))
      return signalPassFailure();
    Limits limits{rows, columns, maxK};
    std::vector<uint64_t> words;
    std::vector<Program> table;

    SmallVector<MatmulOp> tasks;
    module.walk([&](MatmulOp op) { tasks.push_back(op); });
    func::FuncOp gemmFn;
    if (!tasks.empty())
      gemmFn = declareRuntime(module);

    for (MatmulOp op : tasks) {
      auto weight = op.getRhs().getDefiningOp<WeightOp>();
      std::string name = weight.getName().str();
      auto found = offsets.find(name);
      if (found == offsets.end()) {
        op.emitError("weight '") << name << "' has no offset";
        return signalPassFailure();
      }
      auto lhsType = op.getLhs().getType();
      auto partialsType = op.getPartials().getType();
      int64_t m = lhsType.getDimSize(0), k = lhsType.getDimSize(1);
      int64_t n = op.getRhs().getType().getDimSize(1);
      GemmPlan plan = GemmPlan::make(m, n, k, limits, op.getTk());
      if (plan.tk != op.getTk()) {
        op.emitError("tk exceeds the NPU's MAX_K");
        return signalPassFailure();
      }
      int64_t cBase = align(plan.aBytes(), 64);
      Program program{static_cast<int64_t>(table.size()),
                      static_cast<int64_t>(words.size()),
                      0,
                      plan,
                      name,
                      found->second,
                      {}};
      std::vector<int64_t> bWords;
      encodeGemm(plan, 0, 0, cBase, words, bWords);
      words.push_back(encodeEnd());
      program.length = static_cast<int64_t>(words.size()) - program.wordOffset;
      for (int64_t w : bWords)
        program.relocations.push_back(w - program.wordOffset);
      table.push_back(program);

      OpBuilder b(op);
      Location loc = op.getLoc();
      auto aType = MemRefType::get(lhsType.getShape(), lhsType.getElementType());
      auto cType = MemRefType::get(partialsType.getShape(),
                                   partialsType.getElementType());
      Value a = memref::AllocOp::create(b, loc, aType);
      bufferization::MaterializeInDestinationOp::create(b, loc, op.getLhs(), a)
          .setWritable(true);
      Value c = memref::AllocOp::create(b, loc, cType);
      Value aFlat = memref::CollapseShapeOp::create(
          b, loc, a, SmallVector<ReassociationIndices>{{0, 1}});
      Value cFlat = memref::CollapseShapeOp::create(
          b, loc, c, SmallVector<ReassociationIndices>{{0, 1, 2}});
      auto fnType = gemmFn.getFunctionType();
      Value aDyn = memref::CastOp::create(b, loc, fnType.getInput(1), aFlat);
      Value cDyn = memref::CastOp::create(b, loc, fnType.getInput(2), cFlat);
      Value id = arith::ConstantIntOp::create(b, loc, program.id, 32);
      func::CallOp::create(b, loc, gemmFn, ValueRange{id, aDyn, cDyn});
      auto result = bufferization::ToTensorOp::create(b, loc, partialsType, c);
      result.setRestrict(true);
      result.setWritable(true);
      op.getPartials().replaceAllUsesWith(result.getResult());
      op.erase();
    }

    SmallVector<WeightOp> weightOps;
    module.walk([&](WeightOp op) { weightOps.push_back(op); });
    for (WeightOp op : weightOps) {
      if (op->use_empty()) {
        op.erase();
        continue;
      }
      std::string name = op.getName().str();
      auto found = offsets.find(name);
      if (found == offsets.end()) {
        op.emitError("weight '") << name << "' has no offset";
        return signalPassFailure();
      }
      auto arena = dyn_cast<MemRefType>(op.getArena().getType());
      if (!arena || arena.getRank() != 1 ||
          !arena.getElementType().isInteger(8) || !arena.getLayout().isIdentity()) {
        op.emitError("the weight arena must be a 1-D i8 memref");
        return signalPassFailure();
      }
      auto tensorType = cast<RankedTensorType>(op.getResult().getType());
      OpBuilder b(op);
      Location loc = op.getLoc();
      Value offset = arith::ConstantIndexOp::create(b, loc, found->second);
      auto viewType = MemRefType::get(tensorType.getShape(),
                                      tensorType.getElementType());
      Value view = memref::ViewOp::create(b, loc, viewType, op.getArena(),
                                          offset, ValueRange{});
      auto tensor = bufferization::ToTensorOp::create(b, loc, tensorType, view);
      tensor.setRestrict(true);
      op.getResult().replaceAllUsesWith(tensor.getResult());
      op.erase();
    }

    if (programs.empty())
      return;
    std::error_code ec;
    llvm::raw_fd_ostream bin(programs + ".bin", ec, llvm::sys::fs::OF_None);
    if (ec) {
      module.emitError("cannot write programs: ") << ec.message();
      return signalPassFailure();
    }
    for (uint64_t w : words)
      for (int byte = 0; byte < 8; ++byte)
        bin << static_cast<char>((w >> (8 * byte)) & 0xff);
    llvm::raw_fd_ostream js(programs + ".json", ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError("cannot write program table: ") << ec.message();
      return signalPassFailure();
    }
    llvm::json::OStream j(js, 2);
    j.object([&] {
      j.attribute("isa_version", 1);
      j.attribute("rows", static_cast<int64_t>(rows));
      j.attribute("columns", static_cast<int64_t>(columns));
      j.attribute("max_k", static_cast<int64_t>(maxK));
      j.attribute("words", static_cast<int64_t>(words.size()));
      j.attributeArray("programs", [&] {
        for (const Program &p : table)
          j.object([&] {
            j.attribute("id", p.id);
            j.attribute("word_offset", p.wordOffset);
            j.attribute("length", p.length);
            j.attribute("m", p.plan.m);
            j.attribute("n", p.plan.n);
            j.attribute("k", p.plan.k);
            j.attribute("tm", p.plan.tm);
            j.attribute("tn", p.plan.tn);
            j.attribute("tk", p.plan.tk);
            j.attribute("jobs", p.plan.jobs());
            j.attribute("a_bytes", p.plan.aBytes());
            j.attribute("b_bytes", p.plan.bBytes());
            j.attribute("c_base", align(p.plan.aBytes(), 64));
            j.attribute("c_bytes", p.plan.cBytes());
            j.attribute("weight", p.weight);
            j.attribute("weight_offset", p.weightOffset);
            j.attributeArray("relocations", [&] {
              for (int64_t r : p.relocations)
                j.value(r);
            });
          });
      });
    });
  }
};

} // namespace

void mlir::npu::registerLowerPass() { PassRegistration<LowerPass>(); }
