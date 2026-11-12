//===- rv_codegen.h - LLVM dialect -> flat RV32IMC binary --------*- C++ -*-===//
#ifndef BACKEND_DELEGATES_RISC_V_NPU_RV_CODEGEN_H
#define BACKEND_DELEGATES_RISC_V_NPU_RV_CODEGEN_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace backend {

// What CVA6 runs: RV32IMC, ilp32, no other extensions (the cv32a60x-derived
// configuration may lose RVB / RVZCB at the synthesis gate, plan
// 2026-1112-003 section 6.2).
struct RvTarget {
  std::string triple = "riscv32-unknown-elf";
  std::string cpu = "generic-rv32";
  std::string features = "+m,+c,-relax";
  std::string abi = "ilp32";
};

struct RvImage {
  std::vector<uint8_t> bytes;  // the loadable bytes, from offset 0 (bin_bytes)
  int64_t memBytes = 0;        // what the program uses from its load address
};

// Whether this pynqo-backend was built with LLVM's RISCV target.
bool rvCodegenAvailable();

// `module` (LLVM dialect, defining the program's functions) -> LLVM IR,
// optimized at O2 and compiled by LLVM's RISC-V back end in process, with
// `runtimeAsm` (crt0) assembled into the same object by LLVM MC; then linked
// by `<llvmBin>/ld.lld` with `linkScript` and flattened. Intermediate files
// go to `workDir/<name>.*` and are removed unless `keep`.
//
// The object is refused if it holds an absolute relocation in a loaded
// section: the image is linked at 0 but loaded anywhere.
mlir::LogicalResult rvCodegen(mlir::ModuleOp module, const RvTarget &target,
                              llvm::StringRef runtimeAsm, llvm::StringRef linkScript,
                              const std::string &workDir, const std::string &name,
                              const std::string &llvmBin, bool keep, RvImage &image,
                              std::string &error);

} // namespace backend

#endif // BACKEND_DELEGATES_RISC_V_NPU_RV_CODEGEN_H
