//===- delegate.h - a device the back end compiles subgraphs for -*- C++ -*-===//
//
// The middle end decided which operations a device runs, cut each out into
// that device's module (ir/delegates/<device>.mlir: one function per
// delegate id, carrying pynqo.delegate_id) and left a delegate call in the
// CPU graph:
//
//   %y = func.call @__delegate_<id>(%x)      (a bodiless declaration)
//
// A delegate compiles its device's module, and only that: it never sees the
// CPU graph and contributes no CPU code. The CPU code generator lowers every
// delegate call to the runtime ABI
//
//   func.call @__delegate__(id, in, out)     (bytes in, bytes out)
//
// (cpu/delegate_calls.cpp), and the runtime dispatches it by id to the
// device's runtime library (src/runtime). `in` holds the subgraph's argument
// and `out` its result, so a device's own way of computing the result (for
// the NPU, K chunks summed in libnpu_rt) stays inside the device.
//
//===----------------------------------------------------------------------===//
#ifndef BACKEND_DELEGATES_DELEGATE_H
#define BACKEND_DELEGATES_DELEGATE_H

#include "mlir/IR/BuiltinOps.h"

#include <memory>
#include <string>
#include <vector>

namespace backend {

class WeightArena;

class Delegate {
public:
  virtual ~Delegate() = default;
  virtual std::string name() const = 0;
  // Compile `module`, this device's subgraphs, into device artefacts under
  // `out`. Constants the subgraphs read (weights) belong to the delegate: it
  // stores them in the arena in its own layout. Returns the delegate ids in
  // the device's own index order (index i runs ids[i]); pynqo-backend writes
  // that index into the delegate table.
  virtual mlir::FailureOr<std::vector<int64_t>>
  compile(mlir::ModuleOp module, WeightArena &arena, const std::string &out) = 0;
};

} // namespace backend

#endif // BACKEND_DELEGATES_DELEGATE_H
