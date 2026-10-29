//===- delegate.h - a device the back end can hand regions to ---*- C++ -*-===//
//
// A delegate claims the operations its device runs, then compiles each one
// it was given an id for. Partition outlines every claimed operation as
//
//   func.call @__delegate__(id, in, out)      (bytes in, bytes out)
//
// and the runtime dispatches the call by id to the device's backend
// (src/runtime). The generic passes, CPU codegen and model.so never know
// which device an id belongs to.
//
//===----------------------------------------------------------------------===//
#ifndef BACKEND_DELEGATES_DELEGATE_H
#define BACKEND_DELEGATES_DELEGATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace backend {

class WeightArena;

class Delegate {
public:
  virtual ~Delegate() = default;
  virtual std::string name() const = 0;
  // Rewrite the module so the operations this device runs are recognisable,
  // and return them in module order.
  virtual mlir::FailureOr<std::vector<mlir::Operation *>>
  claims(mlir::ModuleOp module) = 0;
  // Compile the claimed operations, each with this device's own index for it
  // (0..n-1 in module order; the manifest maps global ids to it). Constants the
  // operations read (weights) belong to the delegate: it stores them in the
  // arena in its own layout. Device artefacts go under `out`.
  virtual mlir::LogicalResult
  compile(const std::vector<std::pair<int64_t, mlir::Operation *>> &ops,
          WeightArena &arena, const std::string &out) = 0;
};

} // namespace backend

#endif // BACKEND_DELEGATES_DELEGATE_H
