//===- LowerToRiscv.cpp - rvnpu.matmul as one CVA6 kernel -----------------===//
//
// One parameterized RV32IMC kernel per package (issue #206; plan
// 2026-1112-003 section 6.2). Where npu_pynq's Encode.cpp writes ISA words
// (ADDR_A, ADDR_B, ADDR_C, INCR, SHAPE, REPEAT, GEMM), this builds the loops
// that issue the same jobs, in the same order, through the MMIO window
// (RvNpuMmio.h). Before every segment run the driver writes that task's and
// segment's numbers to the parameter block at DATA_BASE (RVNPU_PARAM_*,
// spec/layout.py), so one image runs every (task, segment):
//
//   func.func @rvnpu_main() -> i32 {
//     %db = load volatile DATA_BASE
//     %M, %N, ... = load %db + RVNPU_PARAM_*   the 16 parameter words
//     scf.for %mi = 0 to MT {             M tiles
//       scf.for %kc = 0 to KT {           K chunks
//         store volatile SHAPE_K, ADDR_A
//         scf.for %ni = NI_LO to NI_HI {  the segment's column tiles
//           store volatile SHAPE_MN, ADDR_B, ADDR_C, DOORBELL
//     }}}
//     %f = load volatile FENCE            returns once every job is written back
//     return error bit ? error code : 0   crt0.S stores it to DONE
//   }
//
// Addresses are DATA_BASE plus the offsets of spec/layout.py (DataLayout and
// GemmPlan): parameters at 0, A at A_BASE, C at C_BASE, the segment's staged
// B block at B_BASE. The bounds and strides the per-task programs used to
// carry as constants are computed from the parameters once, before the loops
// (RV32IMC has mul); the loop bodies do the same arithmetic as before, so a
// (task, segment) issues the same jobs as its old program did. Everything is
// i32: RV32 has no wider registers, and the MMIO registers are 32 bits wide.
//
// The parameter words are read with plain (non-volatile) loads: they are
// ordinary DDR memory, not device registers, and nothing writes them while
// the kernel runs. CVA6 is held in reset between runs, so its caches start
// cold and a load never sees a stale line from the previous segment's block.
// Plain loads leave LLVM free to schedule them; each is read once, into an SSA
// value, at entry.
//
//===----------------------------------------------------------------------===//

#include "RvNpu/RvNpuMmio.h"
#include "RvNpu/RvNpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::rvnpu;

Program mlir::rvnpu::planProgram(int64_t id, const GemmPlan &plan,
                                 int64_t weightOffset, int64_t segmentBytes) {
  // spec/layout.py data_layout: the parameter block at 0, A right after it,
  // C and the B stage window each on the next 64-byte boundary.
  constexpr int64_t kBlockAlign = 64;
  constexpr int64_t aBase = RVNPU_PARAM_BYTES;
  Program program{id, plan, weightOffset, {}, {}};
  int64_t cBase = align(aBase + plan.aBytes(), kBlockAlign);
  int64_t bBase = align(cBase + plan.cBytes(), kBlockAlign);
  int64_t perSegment = std::max<int64_t>(1, segmentBytes / plan.bPanel());
  int64_t largest = 0;
  for (int64_t lo = 0; lo < plan.nt(); lo += perSegment) {
    int64_t hi = std::min(plan.nt(), lo + perSegment);
    int64_t bytes = (hi - lo) * plan.bPanel();
    program.segments.push_back({lo, hi, weightOffset + lo * plan.bPanel(), bytes,
                                plan.mt() * plan.kt() * (hi - lo)});
    largest = std::max(largest, bytes);
  }
  program.data = {aBase, plan.aBytes(), cBase, plan.cBytes(), bBase, largest};
  return program;
}

namespace {

class Emitter {
public:
  Emitter(OpBuilder &b, Location loc) : b(b), loc(loc) {}

  Value c(int64_t v) {
    return arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(static_cast<int32_t>(v)));
  }
  Value add(Value x, Value y) { return arith::AddIOp::create(b, loc, x, y); }
  Value sub(Value x, Value y) { return arith::SubIOp::create(b, loc, x, y); }
  Value mul(Value x, Value y) { return arith::MulIOp::create(b, loc, x, y); }
  Value min(Value x, Value y) { return arith::MinSIOp::create(b, loc, x, y); }
  Value andi(Value x, Value y) { return arith::AndIOp::create(b, loc, x, y); }
  Value shl(Value x, int64_t s) { return arith::ShLIOp::create(b, loc, x, c(s)); }
  Value shr(Value x, int64_t s) { return arith::ShRUIOp::create(b, loc, x, c(s)); }
  Value align8(Value x) { return andi(add(x, c(7)), c(~int64_t{7})); }

  Value pointer(Value address) {
    return LLVM::IntToPtrOp::create(b, loc, LLVM::LLVMPointerType::get(b.getContext()),
                                    address);
  }
  Value reg(int64_t offset) {
    return pointer(c(static_cast<int64_t>(RVNPU_MMIO_BASE) + offset));
  }
  void store(int64_t offset, Value v) {
    LLVM::StoreOp::create(b, loc, v, reg(offset), /*alignment=*/4, /*isVolatile=*/true);
  }
  Value load(int64_t offset) {
    return LLVM::LoadOp::create(b, loc, b.getI32Type(), reg(offset), /*alignment=*/4,
                                /*isVolatile=*/true);
  }
  // The parameter word at byte `offset` of the block at `base`: plain DDR
  // memory, so a plain load (see the file header).
  Value param(Value base, int64_t offset) {
    return LLVM::LoadOp::create(b, loc, b.getI32Type(), pointer(add(base, c(offset))),
                                /*alignment=*/4);
  }

  // scf.for %iv = lo to hi, with the builder left inside its body.
  Value loop(Value lo, Value hi) {
    auto op = scf::ForOp::create(b, loc, lo, hi, c(1));
    b.setInsertionPoint(op.getBody()->getTerminator());
    return op.getInductionVar();
  }

private:
  OpBuilder &b;
  Location loc;
};

} // namespace

OwningOpRef<ModuleOp> mlir::rvnpu::lowerToRiscv(MLIRContext *context) {
  context->loadDialect<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                       scf::SCFDialect>();
  Location loc = NameLoc::get(StringAttr::get(context, "rvnpu.kernel"));
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder b(context);
  b.setInsertionPointToEnd(module->getBody());
  auto fn = func::FuncOp::create(b, loc, kEntryName, b.getFunctionType({}, {b.getI32Type()}));
  b.setInsertionPointToStart(fn.addEntryBlock());
  Emitter e(b, loc);

  Value db = e.load(RVNPU_MMIO_DATA_BASE);
  Value m = e.param(db, RVNPU_PARAM_M);
  Value n = e.param(db, RVNPU_PARAM_N);
  Value k = e.param(db, RVNPU_PARAM_K);
  Value tm = e.param(db, RVNPU_PARAM_TM);
  Value tn = e.param(db, RVNPU_PARAM_TN);
  Value tk = e.param(db, RVNPU_PARAM_TK);
  Value mt = e.param(db, RVNPU_PARAM_MT);
  Value kt = e.param(db, RVNPU_PARAM_KT);
  Value nt = e.param(db, RVNPU_PARAM_NT);
  Value niLo = e.param(db, RVNPU_PARAM_NI_LO);
  Value niHi = e.param(db, RVNPU_PARAM_NI_HI);
  Value aOff = e.param(db, RVNPU_PARAM_A_BASE);
  Value cOff = e.param(db, RVNPU_PARAM_C_BASE);
  Value bOff = e.param(db, RVNPU_PARAM_B_BASE);
  Value bPanel = e.param(db, RVNPU_PARAM_B_PANEL);
  Value cPlane = e.param(db, RVNPU_PARAM_C_PLANE);

  // Loop invariants: what the per-task programs carried as constants.
  // ADDR_B is the segment's block: tile ni sits at (ni - niLo) * bPanel
  // inside it, so fold -niLo * bPanel into the base.
  Value aBase = e.add(db, aOff);
  Value cBase = e.add(db, cOff);
  Value bBase = e.sub(e.add(db, bOff), e.mul(niLo, bPanel));
  Value aRowStride = e.mul(kt, e.align8(e.mul(tm, tk)));          // kt * align(tm * tk)
  Value cRowStride = e.mul(nt, e.align8(e.shl(e.mul(tm, tn), 2))); // nt * align(4 * tm * tn)
  Value bChunk = e.align8(e.mul(tk, tn));                         // align(tk * tn)
  Value tn4 = e.shl(tn, 2);                                       // 4 * tn

  Value mi = e.loop(e.c(0), mt);
  {
    Value mI = e.min(tm, e.sub(m, e.mul(mi, tm)));
    // aOffset(mi, kc) = mi * kt * align(tm * tk) + kc * align(mI * tk)
    Value aRow = e.add(aBase, e.mul(mi, aRowStride));
    Value aStep = e.align8(e.mul(mI, tk));
    // cOffset(kc, mi, ni) = kc * cPlane + mi * nt * align(4 * tm * tn)
    //                       + ni * align(4 * mI * tn)   (every row before mi is full)
    Value cRow = e.add(cBase, e.mul(mi, cRowStride));
    Value cStride = e.align8(e.mul(mI, tn4));
    Value shapeM = mI;

    Value kc = e.loop(e.c(0), kt);
    {
      Value kC = e.min(tk, e.sub(k, e.mul(kc, tk)));
      e.store(RVNPU_MMIO_SHAPE_K, kC);
      e.store(RVNPU_MMIO_ADDR_A, e.add(aRow, e.mul(kc, aStep)));
      // bOffset(kc, ni) = ni * bPanel + kc * align(tk * tn)
      Value bK = e.add(bBase, e.mul(kc, bChunk));
      Value cK = e.add(cRow, e.mul(kc, cPlane));

      Value ni = e.loop(niLo, niHi);
      {
        Value nI = e.min(tn, e.sub(n, e.mul(ni, tn)));
        e.store(RVNPU_MMIO_SHAPE_MN,
                arith::OrIOp::create(b, loc, shapeM, e.shl(nI, RVNPU_SHAPE_N_SHIFT)));
        e.store(RVNPU_MMIO_ADDR_B, e.add(bK, e.mul(ni, bPanel)));
        e.store(RVNPU_MMIO_ADDR_C, e.add(cK, e.mul(ni, cStride)));
        e.store(RVNPU_MMIO_DOORBELL, e.c(1));
      }
    }
  }
  b.setInsertionPointAfter(mi.getParentRegion()->getParentOp());

  // FENCE returns once every issued job is written back, or a fault is
  // latched; a latched fault also wins over the DONE code in hardware.
  Value fence = e.load(RVNPU_MMIO_FENCE);
  Value failed = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                       e.andi(fence, e.c(RVNPU_STATUS_ERROR)), e.c(0));
  Value code = e.andi(e.shr(fence, RVNPU_STATUS_ERROR_CODE_SHIFT),
                      e.c(RVNPU_STATUS_ERROR_CODE_MASK));
  Value result = arith::SelectOp::create(b, loc, failed, code, e.c(RVNPU_DONE_SUCCESS));
  func::ReturnOp::create(b, loc, result);
  return module;
}

LogicalResult mlir::rvnpu::convertToLLVMDialect(ModuleOp module) {
  // Needs the upstream passes registered (pynqo-backend calls registerAllPasses).
  PassManager pm(module.getContext());
  if (failed(parsePassPipeline(
          "canonicalize,cse,convert-scf-to-cf,"
          "convert-arith-to-llvm{index-bitwidth=32},convert-cf-to-llvm{index-bitwidth=32},"
          "convert-func-to-llvm{index-bitwidth=32},reconcile-unrealized-casts",
          pm)))
    return module.emitError("bad risc_v_npu lowering pipeline");
  return pm.run(module);
}
