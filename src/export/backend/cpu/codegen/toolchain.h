//===- toolchain.h - CPU code for one target --------------------*- C++ -*-===//
#ifndef BACKEND_CPU_CODEGEN_TOOLCHAIN_H
#define BACKEND_CPU_CODEGEN_TOOLCHAIN_H

#include "device_register.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace backend {

// Translate `module` (the LLVM dialect, from cpu/lowering/pipeline.h) to LLVM
// IR, run opt and llc for `target`, and link it with the runtime support the
// generated code may call into `<dir>/model.so`. `llvmBin` holds opt, llc,
// clang and ld.lld (LLVM 22).
mlir::LogicalResult codegen(mlir::ModuleOp module, const CpuTarget &target,
                            const std::string &dir, const std::string &llvmBin,
                            bool keepIr, std::string &error);

} // namespace backend

#endif // BACKEND_CPU_CODEGEN_TOOLCHAIN_H
