//===- Constants.h - float32 tensors of torch.vtensor.literal ----*- C++ -*-===//
#ifndef MIDDLE_END_TORCH_TRANSFORMS_CONSTANTS_H
#define MIDDLE_END_TORCH_TRANSFORMS_CONSTANTS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringMap.h"

#include <optional>
#include <string>
#include <vector>

namespace middle_end {

// The values of a float32 torch.vtensor.literal, or nullopt for anything else.
std::optional<std::vector<float>> readFloats(mlir::Value value);

// The name of a literal's dense_resource, which stays stable across entry points.
std::string resourceName(mlir::Value literal);

// Makes the float32 literals of fused layers. A layer shared by entry points
// appears once per entry point; its constants are made once, so the back end's
// weight arena still stores them once.
class FusedConstants {
public:
  // A literal of `data` before the rewriter's insertion point, typed as `like`.
  mlir::Value literal(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      llvm::StringRef name, llvm::ArrayRef<float> data,
                      mlir::Value like);

private:
  llvm::StringMap<mlir::ElementsAttr> made;
};

} // namespace middle_end

#endif // MIDDLE_END_TORCH_TRANSFORMS_CONSTANTS_H
