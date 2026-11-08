//===- partition.h - hand claimed regions to delegates ----------*- C++ -*-===//
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

// Let every delegate claim its operations, compile them, and replace each
// with `func.call @__delegate__(id, in, out)`. Delegates run in the order
// given; an operation belongs to the first that claims it.
mlir::FailureOr<std::vector<DelegateEntry>>
partition(mlir::ModuleOp module, const std::vector<std::unique_ptr<Delegate>> &delegates,
          WeightArena &arena, const std::string &out);

} // namespace backend

#endif // BACKEND_PARTITION_H
