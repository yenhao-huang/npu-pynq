//===- partition.cpp - hand claimed regions to delegates ------------------===//
//
// Every delegate claims the operations its device runs. Each claimed
// operation gets a global delegate id; its device compiles it (with its own
// index for it), and the operation is replaced by a call the runtime
// dispatches by id:
//
//   %r = <claimed op>(%x, %w = arith.constant)
//     ->  %in  = memref.alloc() : memref<Bxi8>    viewed as %x's type, %x stored into it
//         %out = memref.alloc() : memref<Cxi8>    viewed as %r's type
//         func.call @__delegate__(%id, %in, %out) : (i64, memref<?xi8>, memref<?xi8>) -> ()
//         %r = bufferization.to_tensor %out-view
//
// Operands defined by constants (weights) belong to the delegate, which
// stores them itself; a claimed operation has one other operand and one
// result. The C interface of __delegate__ is
//
//   void _mlir_ciface___delegate__(int64_t id, memref_1d_i8 *in, memref_1d_i8 *out)
//
//===----------------------------------------------------------------------===//

#include "partition.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace backend {
namespace {

constexpr llvm::StringLiteral kDelegateFn = "__delegate__";

int64_t byteSize(RankedTensorType type) {
  return type.getNumElements() * ((type.getElementTypeBitWidth() + 7) / 8);
}

func::FuncOp declareDelegate(ModuleOp module) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(kDelegateFn))
    return fn;
  OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
  auto bytes = MemRefType::get({ShapedType::kDynamic}, b.getI8Type());
  auto type = b.getFunctionType({b.getI64Type(), bytes, bytes}, {});
  auto fn = func::FuncOp::create(b, module.getLoc(), kDelegateFn, type);
  fn.setPrivate();
  // This makes the call target _mlir_ciface___delegate__, which the runtime defines.
  fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  return fn;
}

// A byte buffer of `type`'s size and a typed view of it.
std::pair<Value, Value> allocBytes(OpBuilder &b, Location loc, RankedTensorType type,
                                   MemRefType dynBytes) {
  auto raw = MemRefType::get({byteSize(type)}, b.getI8Type());
  Value buffer = memref::AllocOp::create(b, loc, raw, b.getI64IntegerAttr(64));
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  auto viewType = MemRefType::get(type.getShape(), type.getElementType());
  Value view = memref::ViewOp::create(b, loc, viewType, buffer, zero, ValueRange{});
  Value flat = memref::CastOp::create(b, loc, dynBytes, buffer);
  return {flat, view};
}

LogicalResult outline(Operation *op, int64_t id, func::FuncOp callee,
                      DelegateEntry &entry) {
  SmallVector<Value> inputs;
  for (Value v : op->getOperands())
    if (!v.getDefiningOp<arith::ConstantOp>())
      inputs.push_back(v);
  if (inputs.size() != 1 || op->getNumResults() != 1)
    return op->emitError("a delegated operation needs one non-constant operand and one result");
  auto inType = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inType || !outType || !inType.hasStaticShape() || !outType.hasStaticShape())
    return op->emitError("delegated tensors must have static shapes");

  OpBuilder b(op);
  Location loc = op->getLoc();
  auto dynBytes = cast<MemRefType>(callee.getFunctionType().getInput(1));
  auto [inFlat, inView] = allocBytes(b, loc, inType, dynBytes);
  // The delegate reads this buffer as raw bytes, so %x must really be stored in it.
  bufferization::MaterializeInDestinationOp::create(b, loc, inputs[0], inView)
      .setWritable(true);
  auto [outFlat, outView] = allocBytes(b, loc, outType, dynBytes);
  Value idValue = arith::ConstantIntOp::create(b, loc, id, 64);
  func::CallOp::create(b, loc, callee, ValueRange{idValue, inFlat, outFlat});
  auto result = bufferization::ToTensorOp::create(b, loc, outType, outView);
  // The delegate filled a fresh buffer that nothing else refers to.
  // Later operations may therefore reuse it in place.
  result.setRestrict(true);
  result.setWritable(true);
  op->getResult(0).replaceAllUsesWith(result.getResult());

  SmallVector<Operation *> constants;
  for (Value v : op->getOperands())
    if (auto c = v.getDefiningOp<arith::ConstantOp>())
      constants.push_back(c);
  op->erase();
  for (Operation *c : constants)
    // Another operation may still read the same constant.
    if (c->use_empty())
      c->erase();
  entry.inputBytes = byteSize(inType);
  entry.outputBytes = byteSize(outType);
  return success();
}

} // namespace

FailureOr<std::vector<DelegateEntry>>
partition(ModuleOp module, const std::vector<std::unique_ptr<Delegate>> &delegates,
          WeightArena &arena, const std::string &out) {
  std::vector<DelegateEntry> table;
  llvm::DenseSet<Operation *> taken;
  std::vector<std::pair<int64_t, Operation *>> toOutline;
  for (const auto &delegate : delegates) {
    auto claimed = delegate->claims(module);
    if (failed(claimed))
      return failure();
    std::vector<std::pair<int64_t, Operation *>> mine;
    for (Operation *op : *claimed) {
      if (!taken.insert(op).second)
        continue;
      int64_t id = static_cast<int64_t>(table.size());
      int64_t index = static_cast<int64_t>(mine.size());
      table.push_back({id, delegate->name(), index, 0, 0});
      mine.push_back({index, op});
      toOutline.push_back({id, op});
    }
    // Compile before anything is outlined.
    // compile reads the operations and their weights, and outline erases them.
    if (!mine.empty() && failed(delegate->compile(mine, arena, out)))
      return failure();
  }
  if (toOutline.empty())
    return table;
  func::FuncOp callee = declareDelegate(module);
  for (auto [id, op] : toOutline)
    if (failed(outline(op, id, callee, table[id])))
      return failure();
  return table;
}

} // namespace backend
