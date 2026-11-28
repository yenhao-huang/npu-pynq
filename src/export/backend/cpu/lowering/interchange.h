//===- interchange.h - unit-stride innermost loops --------------*- C++ -*-===//
#ifndef BACKEND_CPU_LOWERING_INTERCHANGE_H
#define BACKEND_CPU_LOWERING_INTERCHANGE_H

#include "mlir/IR/BuiltinOps.h"

namespace backend {

// On linalg on tensors, before bufferization: make every linalg op's
// innermost loop the dimension with the fewest strided operands, so that
// LLVM's loop vectorizer sees unit-stride or invariant accesses. Named ops are
// generalized first. Every output element still sums its terms in the same
// order, so the result is bit-identical.
void interchangeForUnitStride(mlir::ModuleOp module);

} // namespace backend

#endif // BACKEND_CPU_LOWERING_INTERCHANGE_H
