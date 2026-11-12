//===- RvNpuPasses.h - risc_v_npu passes and compile steps ------*- C++ -*-===//
//
// The risc_v_npu delegate's compile-time steps (plan 2026-1112-003 section
// 6.2): Claim and Pack are copies of npu_pynq's; LowerToRiscv replaces its
// ISA encoder with an RV32IMC program per segment, issued through the CVA6
// MMIO window (RvNpuMmio.h). The back end's delegate
// (src/export/backend/delegates/risc_v_npu/) drives them and compiles the
// programs; nothing here includes the back end.
//
//===----------------------------------------------------------------------===//
#ifndef RVNPU_RVNPUPASSES_H
#define RVNPU_RVNPUPASSES_H

#include "RvNpu/RvNpuLayout.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::rvnpu {

// --rvnpu-claim: every INT8 x INT8 -> INT32 linalg.matmul whose right operand
// is a constant becomes rvnpu.matmul + a sum of its K planes.
std::unique_ptr<Pass> createClaimPass(int64_t maxK = 256);
void registerClaimPass();

// Pack: a [K, N] INT8 weight as the B tiles the NPU reads (spec/layout.py).
std::vector<int8_t> packB(const GemmPlan &plan, const int8_t *b);

// One delegate's data block, as offsets from DATA_BASE (spec/layout.py
// DataLayout): A at 0, C at cBase, the staged B segment at bBase.
struct DataLayout {
  int64_t aBytes, cBase, cBytes, bBase, bBytes;
};

// One segment: the column tiles [niLo, niHi) whose B panels form one block
// of at most segmentBytes (the same split as npu_pynq's Encode.cpp and
// spec/layout.py GemmPlan.segments). The driver stages that block at
// DATA_BASE + bBase and runs the segment's own binary.
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

// The name of the function a program module defines; crt0.S calls it and
// stores its i32 result to DONE.
constexpr llvm::StringLiteral kEntryName = "rvnpu_main";

// LowerToRiscv: segment `index` of `program` as a module holding
//
//   func.func @rvnpu_main() -> i32
//
// in scf / arith / func, with every MMIO access a volatile llvm.load /
// llvm.store at an absolute address. It reads DATA_BASE, issues the
// segment's jobs in the order npu_pynq's encodeGemm does (mi, kc, ni), then
// reads FENCE and returns the MMIO error code (0 = success).
OwningOpRef<ModuleOp> lowerToRiscv(MLIRContext *context, const Program &program,
                                   int64_t index);

// Upstream conversions from a LowerToRiscv module to the LLVM dialect
// (32-bit index, as RV32).
LogicalResult convertToLLVMDialect(ModuleOp module);

// The CVA6 runtime linked into every program (custom_dialect/runtime/),
// embedded at build time. crt0Source() is preceded by mmioEquates(): `.equ`
// lines for the RvNpuMmio.h constants crt0.S uses, so the assembly needs no
// preprocessor.
llvm::StringRef crt0Source();
llvm::StringRef linkScript();
std::string mmioEquates();

// rv_programs.bin (every segment's flat binary, each at an 8-byte aligned
// offset), rv_programs.txt (the table librv_npu_rt reads) and
// rv_programs.json (the same, for people and tools). binaries[p][s] is the
// image of program p's segment s; memBytes[p][s] what it uses when it runs
// (image, .bss and stack).
bool writePrograms(const std::string &dir, const Limits &limits,
                   int64_t segmentBytes, const std::vector<Program> &programs,
                   const std::vector<std::vector<std::vector<uint8_t>>> &binaries,
                   const std::vector<std::vector<int64_t>> &memBytes,
                   std::string &error);

} // namespace mlir::rvnpu

#endif // RVNPU_RVNPUPASSES_H
