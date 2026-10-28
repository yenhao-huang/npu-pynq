//===- NpuDialect.h - NPU dialect -------------------------------*- C++ -*-===//
#ifndef NPU_NPUDIALECT_H
#define NPU_NPUDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Npu/NpuOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Npu/NpuOps.h.inc"

namespace mlir::npu {
// The npu.module --npu-partition creates, and the attribute on each task
// function that gives the program id the C runtime uses.
inline constexpr llvm::StringLiteral kNpuModuleName = "npu";
inline constexpr llvm::StringLiteral kTaskIdAttr = "npu.task_id";
} // namespace mlir::npu

#endif // NPU_NPUDIALECT_H
