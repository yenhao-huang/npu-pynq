//===- package.h - weights.bin and the package manifest ---------*- C++ -*-===//
#ifndef BACKEND_PACKAGE_H
#define BACKEND_PACKAGE_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>
#include <string>
#include <vector>

namespace backend {

// The weight arena: one byte blob the model reads through its `ws` argument
// and the delegates read through the runtime. Every block is 64-byte aligned.
class WeightArena {
public:
  // Append `data` once per (`key`, `layout`): the constant's attribute and
  // the layout it is stored in (0 dense; a delegate picks its own); return
  // its offset.
  int64_t add(const void *key, int layout, llvm::ArrayRef<char> data);
  int64_t bytes() const { return static_cast<int64_t>(blob.size()); }
  int64_t count() const { return static_cast<int64_t>(offsets.size()); }
  bool write(const std::string &path, std::string &error) const;

private:
  std::vector<char> blob;
  llvm::DenseMap<std::pair<const void *, int>, int64_t> offsets;
};

// The raw bytes of a non-splat constant (dense or dense_resource), or
// failure for anything else.
mlir::FailureOr<llvm::ArrayRef<char>> constantBytes(mlir::Attribute value);

// Generic weight externalization: every public function gets a leading
// `ws: memref<?xi8>` argument, and every large constant tensor becomes a view
// of it at the offset the arena gave the constant.
mlir::LogicalResult externalizeWeights(mlir::ModuleOp module, WeightArena &arena,
                                       int64_t minBytes = 256);

} // namespace backend

#endif // BACKEND_PACKAGE_H
