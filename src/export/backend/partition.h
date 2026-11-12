//===- partition.h - compile device functions, apply the ABI ----*- C++ -*-===//
#ifndef BACKEND_PARTITION_H
#define BACKEND_PARTITION_H

#include "delegates/delegate.h"

#include <memory>
#include <string>
#include <vector>

namespace backend {

// One __delegate__ id: the device it belongs to and that device's own index
// for it (its program number).
struct DelegateEntry {
  int64_t id;
  std::string device;
  int64_t index;
  int64_t inputBytes, outputBytes;
};

// Compile the device functions the middle end outlined: check their
// pynqo.delegate_id values are 0..n-1, let each function's delegate lower and
// compile it, inline the lowered body at its call site, and replace the
// device's operation with `func.call @__delegate__(id, in, out)`. The ids are
// kept as given; a function for a device with no delegate is an error.
mlir::FailureOr<std::vector<DelegateEntry>>
partition(mlir::ModuleOp module, const std::vector<std::unique_ptr<Delegate>> &delegates,
          WeightArena &arena, const std::string &out);

} // namespace backend

#endif // BACKEND_PARTITION_H
