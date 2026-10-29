//===- package.cpp - weights.bin and weight externalization ---------------===//
//
// The front end leaves weights as constants (dense or dense_resource). The
// delegates take the constants their operations read; this pass moves every
// other large constant into the same arena and reads it back through a
// `ws` argument:
//
//   %w = arith.constant dense_resource<blob> : tensor<KxNxf32>
//     ->  %v = memref.view %ws[%offset][] : memref<?xi8> to memref<KxNxf32>
//         %w = bufferization.to_tensor %v restrict
//
//===----------------------------------------------------------------------===//

#include "package.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace backend {

int64_t WeightArena::add(const void *key, int layout, llvm::ArrayRef<char> data) {
  auto found = offsets.find({key, layout});
  if (found != offsets.end())
    return found->second;
  blob.resize((blob.size() + 63) / 64 * 64, 0);
  int64_t offset = static_cast<int64_t>(blob.size());
  blob.insert(blob.end(), data.begin(), data.end());
  offsets[{key, layout}] = offset;
  return offset;
}

bool WeightArena::write(const std::string &path, std::string &error) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    error = "cannot write " + path + ": " + ec.message();
    return false;
  }
  os.write(blob.data(), blob.size());
  return true;
}

FailureOr<llvm::ArrayRef<char>> constantBytes(Attribute value) {
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(value)) {
    AsmResourceBlob *blob = resource.getRawHandle().getBlob();
    if (!blob)
      return failure();
    return blob->getData();
  }
  if (auto dense = dyn_cast<DenseIntOrFPElementsAttr>(value)) {
    if (dense.isSplat() || dense.getElementType().isInteger(1))
      return failure();
    return dense.getRawData();
  }
  return failure();
}

LogicalResult externalizeWeights(ModuleOp module, WeightArena &arena,
                                 int64_t minBytes) {
  MLIRContext *ctx = module.getContext();
  auto arenaType = MemRefType::get({ShapedType::kDynamic}, IntegerType::get(ctx, 8));
  for (auto fn : module.getOps<func::FuncOp>()) {
    if (fn.isPrivate() || fn.isExternal())
      continue;
    if (failed(fn.insertArgument(0, arenaType, DictionaryAttr::get(ctx), fn.getLoc())))
      return fn.emitError("cannot add the weight arena argument");
    Value ws = fn.getArgument(0);
    SmallVector<arith::ConstantOp> constants;
    fn.walk([&](arith::ConstantOp op) {
      auto type = dyn_cast<RankedTensorType>(op.getType());
      if (type && type.hasStaticShape() &&
          type.getNumElements() * ((type.getElementTypeBitWidth() + 7) / 8) >= minBytes)
        constants.push_back(op);
    });
    for (arith::ConstantOp op : constants) {
      auto bytes = constantBytes(op.getValue());
      if (failed(bytes))
        continue;   // splats and booleans stay inline
      auto type = cast<RankedTensorType>(op.getType());
      int64_t offset = arena.add(op.getValue().getAsOpaquePointer(), 0, *bytes);
      OpBuilder b(op);
      Location loc = op.getLoc();
      Value at = arith::ConstantIndexOp::create(b, loc, offset);
      auto viewType = MemRefType::get(type.getShape(), type.getElementType());
      Value view = memref::ViewOp::create(b, loc, viewType, ws, at, ValueRange{});
      auto tensor = bufferization::ToTensorOp::create(b, loc, type, view);
      tensor.setRestrict(true);
      op.getResult().replaceAllUsesWith(tensor.getResult());
      op.erase();
    }
  }
  return success();
}

} // namespace backend
