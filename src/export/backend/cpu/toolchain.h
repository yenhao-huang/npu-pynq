//===- toolchain.h - CPU code for one target --------------------*- C++ -*-===//
#ifndef BACKEND_CPU_TOOLCHAIN_H
#define BACKEND_CPU_TOOLCHAIN_H

#include "../device_register.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace backend {

// Lower `module` (linalg on tensors, with __delegate__ calls) to LLVM IR in
// process, then llc and link it with MLIR's runtime support into
// `<dir>/model.so`. `llvmBin` holds llc, clang and ld.lld (LLVM 22).
mlir::LogicalResult codegen(mlir::ModuleOp module, const CpuTarget &target,
                            const std::string &dir, const std::string &llvmBin,
                            bool keepIr, std::string &error);

} // namespace backend

#endif // BACKEND_CPU_TOOLCHAIN_H
