//===- NpuDialect.h - NPU dialect -------------------------------*- C++ -*-===//
#ifndef NPU_NPUDIALECT_H
#define NPU_NPUDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Npu/NpuOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Npu/NpuOps.h.inc"

#endif // NPU_NPUDIALECT_H
