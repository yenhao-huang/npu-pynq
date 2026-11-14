//===- delegate_calls.h - delegate calls to the runtime ABI -----*- C++ -*-===//
#ifndef BACKEND_CPU_DELEGATE_CALLS_H
#define BACKEND_CPU_DELEGATE_CALLS_H

#include "mlir/IR/BuiltinOps.h"

namespace backend {

// Lower every call to a delegate declaration (@__delegate_<id>, carrying
// pynqo.delegate_id) to `func.call @__delegate__(id, in, out)` on byte
// buffers, then erase the declarations. Reads only the id and the tensor
// types; which device an id belongs to is the runtime's business.
mlir::LogicalResult lowerDelegateCalls(mlir::ModuleOp module);

} // namespace backend

#endif // BACKEND_CPU_DELEGATE_CALLS_H
