//===- LowerToRiscv.cpp - rvnpu.matmul as a CVA6 program ------------------===//
//
// One RV32IMC program per segment of an rvnpu.matmul (plan 2026-1112-003
// section 6.2). Where npu_pynq's Encode.cpp writes ISA words (ADDR_A, ADDR_B,
// ADDR_C, INCR, SHAPE, REPEAT, GEMM), this builds the loops that issue the
// same jobs, in the same order, through the MMIO window (RvNpuMmio.h):
//
//   func.func @rvnpu_main() -> i32 {
//     %db = load volatile DATA_BASE
//     scf.for %mi = 0 to mt {             M tiles
//       scf.for %kc = 0 to kt {           K chunks
//         store volatile SHAPE_K, ADDR_A
//         scf.for %ni = niLo to niHi {    the segment's column tiles
//           store volatile SHAPE_MN, ADDR_B, ADDR_C, DOORBELL
//     }}}
//     %f = load volatile FENCE            returns once every job is written back
//     return error bit ? error code : 0   crt0.S stores it to DONE
//   }
//
// Addresses are DATA_BASE plus the offsets of spec/layout.py (DataLayout and
// GemmPlan): A at 0, C at cBase, the segment's staged B block at bBase. The
// shapes are static, so every bound and stride is a constant; only the loop
// counters are computed at run time. Everything is i32: RV32 has no wider
// registers, and the MMIO registers are 32 bits wide.
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
  // spec/layout.py data_layout: A at 0, C and the B stage window each on the
  // next 64-byte boundary.
  constexpr int64_t kBlockAlign = 64;
  Program program{id, plan, weightOffset, {}, {}};
  int64_t cBase = align(plan.aBytes(), kBlockAlign);
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
  program.data = {plan.aBytes(), cBase, plan.cBytes(), bBase, largest};
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

  Value reg(int64_t offset) {
    Value address = c(static_cast<int64_t>(RVNPU_MMIO_BASE) + offset);
    return LLVM::IntToPtrOp::create(b, loc, LLVM::LLVMPointerType::get(b.getContext()),
                                    address);
  }
  void store(int64_t offset, Value v) {
    LLVM::StoreOp::create(b, loc, v, reg(offset), /*alignment=*/4, /*isVolatile=*/true);
  }
  Value load(int64_t offset) {
    return LLVM::LoadOp::create(b, loc, b.getI32Type(), reg(offset), /*alignment=*/4,
                                /*isVolatile=*/true);
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

OwningOpRef<ModuleOp> mlir::rvnpu::lowerToRiscv(MLIRContext *context,
                                                const Program &program,
                                                int64_t index) {
  context->loadDialect<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                       scf::SCFDialect>();
  const GemmPlan &p = program.plan;
  const Segment &seg = program.segments.at(index);
  Location loc = NameLoc::get(StringAttr::get(
      context, "rvnpu.program." + std::to_string(program.id) + "." + std::to_string(index)));
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder b(context);
  b.setInsertionPointToEnd(module->getBody());
  auto fn = func::FuncOp::create(b, loc, kEntryName, b.getFunctionType({}, {b.getI32Type()}));
  b.setInsertionPointToStart(fn.addEntryBlock());
  Emitter e(b, loc);

  Value db = e.load(RVNPU_MMIO_DATA_BASE);
  // Loop-invariant bases. ADDR_B is the segment's block: tile ni sits at
  // (ni - niLo) * bPanel inside it, so fold -niLo * bPanel into the base.
  Value cBase = e.add(db, e.c(program.data.cBase));
  Value bBase = e.add(db, e.c(program.data.bBase - seg.niLo * p.bPanel()));

  Value mi = e.loop(e.c(0), e.c(p.mt()));
  {
    Value mI = e.min(e.c(p.tm), e.sub(e.c(p.m), e.mul(mi, e.c(p.tm))));
    // aOffset(mi, kc) = mi * kt * align(tm * tk) + kc * align(mI * tk)
    Value aRow = e.add(db, e.mul(mi, e.c(p.kt() * align(p.tm * p.tk))));
    Value aStep = e.align8(e.mul(mI, e.c(p.tk)));
    // cOffset(kc, mi, ni) = kc * cPlane + mi * nt * align(4 * tm * tn)
    //                       + ni * align(4 * mI * tn)   (every row before mi is full)
    Value cRow = e.add(cBase, e.mul(mi, e.c(p.nt() * align(4 * p.tm * p.tn))));
    Value cStride = e.align8(e.mul(mI, e.c(4 * p.tn)));
    Value shapeM = mI;

    Value kc = e.loop(e.c(0), e.c(p.kt()));
    {
      Value kC = e.min(e.c(p.tk), e.sub(e.c(p.k), e.mul(kc, e.c(p.tk))));
      e.store(RVNPU_MMIO_SHAPE_K, kC);
      e.store(RVNPU_MMIO_ADDR_A, e.add(aRow, e.mul(kc, aStep)));
      // bOffset(kc, ni) = ni * bPanel + kc * align(tk * tn)
      Value bK = e.add(bBase, e.mul(kc, e.c(align(p.tk * p.tn))));
      Value cK = e.add(cRow, e.mul(kc, e.c(p.cPlaneBytes())));

      Value ni = e.loop(e.c(seg.niLo), e.c(seg.niHi));
      {
        Value nI = e.min(e.c(p.tn), e.sub(e.c(p.n), e.mul(ni, e.c(p.tn))));
        e.store(RVNPU_MMIO_SHAPE_MN,
                arith::OrIOp::create(b, loc, shapeM, e.shl(nI, RVNPU_SHAPE_N_SHIFT)));
        e.store(RVNPU_MMIO_ADDR_B, e.add(bK, e.mul(ni, e.c(p.bPanel()))));
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
