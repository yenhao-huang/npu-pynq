//===- delegate.h - a device the back end can hand regions to ---*- C++ -*-===//
//
// The middle end decides which operations a device runs and outlines each
// into a device function (@<device>_<id>, pynqo.device, pynqo.delegate_id).
// A delegate lowers each of its device functions to the operation its device
// runs, then compiles those operations. Partition outlines every one as
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  // Rewrite one device function's body into the form compile() reads and
  // return the operation the device runs. Everything else in the body stays
  // CPU code: partition inlines it at the call site.
  virtual mlir::FailureOr<mlir::Operation *> lower(mlir::func::FuncOp fn) = 0;
  // Compile the lowered operations, each with this device's own index for it
  // (0..n-1 in delegate-id order; the manifest maps global ids to it). Constants the
  // operations read (weights) belong to the delegate: it stores them in the
  // arena in its own layout. Device artefacts go under `out`.
  virtual mlir::LogicalResult
  compile(const std::vector<std::pair<int64_t, mlir::Operation *>> &ops,
          WeightArena &arena, const std::string &out) = 0;
};

} // namespace backend

#endif // BACKEND_DELEGATES_DELEGATE_H
