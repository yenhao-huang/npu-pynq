//===- partition.cpp - compile device functions, apply the ABI ------------===//
//
// The middle end outlined every operation a device runs into a device
// function and fixed its global delegate id:
//
//   %r = func.call @npu_<id>(%x)
//   func.func private @npu_<id>(%x) attributes {pynqo.device = "npu",
//                                               pynqo.delegate_id = <id>}
//
// Partition checks the ids are 0..n-1, lets each function's delegate lower its
// body (the NPU: npu.matmul {tk} + a linalg.reduce of the K planes) and
// compile the operation the device runs (with its own index for it), inlines
// the lowered body at the call site, and replaces that operation with a call
// the runtime dispatches by id:
//
//   %r = <device op>(%x, %w = arith.constant)
//     ->  %in  = memref.alloc() : memref<Bxi8>    viewed as %x's type, %x stored into it
//         %out = memref.alloc() : memref<Cxi8>    viewed as %r's type
//         func.call @__delegate__(%id, %in, %out) : (i64, memref<?xi8>, memref<?xi8>) -> ()
//         %r = bufferization.to_tensor %out-view
//
// What else the body held (the NPU's reduce and accumulator fill) stays CPU
// code next to the call. Operands defined by constants (weights) belong to the
// delegate, which stores them itself; a device operation has one other
// operand and one result. The C interface of __delegate__ is
//
//   void _mlir_ciface___delegate__(int64_t id, memref_1d_i8 *in, memref_1d_i8 *out)
//
//===----------------------------------------------------------------------===//

#include "partition.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace backend {
namespace {

constexpr llvm::StringLiteral kDelegateFn = "__delegate__";
// Written by the middle end (middle-end/linalg/transforms/Passes.h); the
// spellings must agree (src/test/tests/test_export_layering.py).
constexpr llvm::StringLiteral kDeviceAttr = "pynqo.device";
constexpr llvm::StringLiteral kDelegateIdAttr = "pynqo.delegate_id";

// One device function and the call that uses it.
struct DeviceFunction {
  func::FuncOp fn;
  func::CallOp call;
  std::string device;
  int64_t id;
};

// Every device function, in delegate-id order, after checking the contract:
// ids 0..n-1 without gaps or repeats, tensor arguments and one tensor result
// of static shape, and exactly one call.
FailureOr<std::vector<DeviceFunction>> deviceFunctions(ModuleOp module) {
  std::vector<DeviceFunction> found;
  for (auto fn : module.getOps<func::FuncOp>()) {
    if (!fn->hasAttr(kDeviceAttr))
      continue;
    auto device = fn->getAttrOfType<StringAttr>(kDeviceAttr);
    auto id = fn->getAttrOfType<IntegerAttr>(kDelegateIdAttr);
    if (!device || !id)
      return fn.emitError() << "a device function needs a string " << kDeviceAttr
                            << " and an integer " << kDelegateIdAttr;
    FunctionType type = fn.getFunctionType();
    auto isStatic = [](Type t) {
      auto tensor = dyn_cast<RankedTensorType>(t);
      return tensor && tensor.hasStaticShape();
    };
    if (fn.isExternal() || type.getNumResults() != 1 || !llvm::all_of(type.getInputs(), isStatic) ||
        !isStatic(type.getResult(0)))
      return fn.emitError("a device function takes static tensors and returns one static tensor");
    auto uses = SymbolTable::getSymbolUses(fn, module);
    func::CallOp call;
    if (uses && llvm::hasSingleElement(*uses))
      call = dyn_cast<func::CallOp>(uses->begin()->getUser());
    if (!call)
      return fn.emitError("a device function needs exactly one func.call");
    found.push_back({fn, call, device.getValue().str(), id.getInt()});
  }
  llvm::sort(found, [](const DeviceFunction &a, const DeviceFunction &b) { return a.id < b.id; });
  for (size_t i = 0; i < found.size(); ++i)
    if (found[i].id != static_cast<int64_t>(i))
      return found[i].fn.emitError() << "delegate ids must be 0.." << found.size() - 1
                                     << " without gaps or repeats; found " << found[i].id
                                     << " at position " << i;
  return found;
}

// Move the body of `f.fn` to its call site, then erase the call and the
// function. Operations keep their identity, so pointers into the body stay valid.
void inlineAtCall(DeviceFunction &f) {
  Block &body = f.fn.front();
  for (auto [argument, operand] : llvm::zip(body.getArguments(), f.call.getOperands()))
    argument.replaceAllUsesWith(operand);
  Operation *ret = body.getTerminator();
  f.call->replaceAllUsesWith(ret->getOperands());
  f.call->getBlock()->getOperations().splice(Block::iterator(f.call), body.getOperations(),
                                             body.begin(), Block::iterator(ret));
  f.call.erase();
  f.fn.erase();
}

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
  auto functions = deviceFunctions(module);
  if (failed(functions))
    return failure();
  std::vector<DelegateEntry> table;
  // The operation each id's device runs, once lowered.
  std::vector<Operation *> lowered(functions->size(), nullptr);
  for (DeviceFunction &f : *functions) {
    if (!llvm::any_of(delegates, [&](const auto &d) { return d->name() == f.device; }))
      return f.fn.emitError() << "device '" << f.device << "' has no delegate in --devices";
    table.push_back({f.id, f.device, 0, 0, 0});
  }
  for (const auto &delegate : delegates) {
    std::vector<std::pair<int64_t, Operation *>> mine;
    for (DeviceFunction &f : *functions) {
      if (f.device != delegate->name())
        continue;
      auto op = delegate->lower(f.fn);
      if (failed(op))
        return failure();
      int64_t index = static_cast<int64_t>(mine.size());
      table[f.id].index = index;
      lowered[f.id] = *op;
      mine.push_back({index, *op});
    }
    // Compile before anything is outlined.
    // compile reads the operations and their weights, and outline erases them.
    if (!mine.empty() && failed(delegate->compile(mine, arena, out)))
      return failure();
  }
  if (functions->empty())
    return table;
  for (DeviceFunction &f : *functions)
    inlineAtCall(f);
  func::FuncOp callee = declareDelegate(module);
  for (size_t id = 0; id < lowered.size(); ++id)
    if (failed(outline(lowered[id], static_cast<int64_t>(id), callee, table[id])))
      return failure();
  return table;
}

} // namespace backend
