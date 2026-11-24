//===- RvNpuDialect.h - risc_v_npu dialect ----------------------*- C++ -*-===//
#ifndef RVNPU_RVNPUDIALECT_H
#define RVNPU_RVNPUDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "RvNpu/RvNpuOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "RvNpu/RvNpuOps.h.inc"

#endif // RVNPU_RVNPUDIALECT_H
