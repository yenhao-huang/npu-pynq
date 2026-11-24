//===- RvNpuPasses.h - risc_v_npu passes and compile steps ------*- C++ -*-===//
//
// The risc_v_npu delegate's compile-time steps (plan 2026-1112-003 section
// 6.2): Claim and Pack are copies of npu_pynq's; LowerToRiscv replaces its
// ISA encoder with one parameterized RV32IMC kernel per package (issue #206)
// that issues every task's jobs through the CVA6 MMIO window (RvNpuMmio.h),
// reading the task and segment from a parameter block at DATA_BASE. The back
// end's delegate (src/export/backend/delegates/risc_v_npu/) drives them and
// compiles the kernel; nothing here includes the back end.
//
//===----------------------------------------------------------------------===//
#ifndef RVNPU_RVNPUPASSES_H
#define RVNPU_RVNPUPASSES_H

#include "RvNpu/RvNpuDialect.h"
#include "RvNpu/RvNpuLayout.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::rvnpu {

// Rewrite one claimable matmul (rvnpu::isClaimable, RvNpu/RvNpuMatch.h) into
// rvnpu.matmul {tk = min(K, maxK)} with the same M x N result; return it. The
// matmul and its zero accumulator, if nothing else reads it, are erased.
MatmulOp claim(linalg::MatmulOp op, int64_t maxK);

// --rvnpu-claim: every INT8 x INT8 -> INT32 linalg.matmul whose right operand
// is a constant and whose accumulator is zero becomes rvnpu.matmul
// (isClaimable + claim).
std::unique_ptr<Pass> createClaimPass(int64_t maxK = 256);
void registerClaimPass();

// Pack: a [K, N] INT8 weight as the B tiles the NPU reads (spec/layout.py).
std::vector<int8_t> packB(const GemmPlan &plan, const int8_t *b);

// One delegate's data block, as offsets from DATA_BASE (spec/layout.py
// DataLayout): the kernel's parameter block at 0 (RVNPU_PARAM_BYTES), A at
// aBase (= RVNPU_PARAM_BYTES), C at cBase, the staged B segment at bBase.
struct DataLayout {
  int64_t aBase, aBytes, cBase, cBytes, bBase, bBytes;
};

// One segment: the column tiles [niLo, niHi) whose B panels form one block
// of at most segmentBytes (the same split as npu_pynq's Encode.cpp and
// spec/layout.py GemmPlan.segments). The driver stages that block at
// DATA_BASE + bBase, writes the segment's parameter block at DATA_BASE and
// runs the kernel.
struct Segment {
  int64_t niLo, niHi, weightOffset, bBytes, jobs;
};

struct Program {
  int64_t id;
  GemmPlan plan;
  int64_t weightOffset;
  DataLayout data;
  std::vector<Segment> segments;
};

// Split one rvnpu.matmul (its plan, and where Pack put its B tiles) into
// segments and fix its data block.
Program planProgram(int64_t id, const GemmPlan &plan, int64_t weightOffset,
                    int64_t segmentBytes);

// The name of the function the kernel module defines; crt0.S calls it and
// stores its i32 result to DONE.
constexpr llvm::StringLiteral kEntryName = "rvnpu_main";

// LowerToRiscv: the kernel, one module per package holding
//
//   func.func @rvnpu_main() -> i32
//
// in scf / arith / func / llvm. It reads DATA_BASE (a volatile MMIO load),
// the RVNPU_PARAM_* words of the parameter block there (plain loads), issues
// the segment's jobs in the order npu_pynq's encodeGemm does (mi, kc, ni)
// with volatile MMIO stores, then reads FENCE and returns the MMIO error code
// (0 = success). It runs any Program and segment whose parameter block the
// driver wrote.
OwningOpRef<ModuleOp> lowerToRiscv(MLIRContext *context);

// Upstream conversions from a LowerToRiscv module to the LLVM dialect
// (32-bit index, as RV32).
LogicalResult convertToLLVMDialect(ModuleOp module);

// The CVA6 runtime linked into every program (devices/risc_v_npu/runtime/),
// embedded at build time. crt0Source() is preceded by mmioEquates(): `.equ`
// lines for the RvNpuMmio.h constants crt0.S uses, so the assembly needs no
// preprocessor.
llvm::StringRef crt0Source();
llvm::StringRef linkScript();
std::string mmioEquates();

// rv_programs.bin (the kernel's flat binary, zero-padded to a multiple of 8
// bytes: bin_bytes), rv_programs.txt (format 2, the table librv_npu_rt reads)
// and rv_programs.json (the same, for people and tools). `memBytes` is what the kernel uses when it runs (image, .bss and
// stack). A package without programs may pass an empty binary and 0.
bool writePrograms(const std::string &dir, const Limits &limits,
                   int64_t segmentBytes, const std::vector<Program> &programs,
                   const std::vector<uint8_t> &binary, int64_t memBytes,
                   std::string &error);

} // namespace mlir::rvnpu

#endif // RVNPU_RVNPUPASSES_H
