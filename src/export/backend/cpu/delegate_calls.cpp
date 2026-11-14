//===- delegate_calls.cpp - delegate calls to the runtime ABI -------------===//
//
// The middle end left every delegated subgraph as a call to a bodiless
// declaration. The CPU code calls the runtime's dispatcher instead, with the
// argument and result as raw bytes:
//
//   %r = func.call @__delegate_<id>(%x)
//     ->  %in  = memref.alloc() : memref<Bxi8>    viewed as %x's type, %x stored into it
//         %out = memref.alloc() : memref<Cxi8>    viewed as %r's type
//         func.call @__delegate__(%id, %in, %out) : (i64, memref<?xi8>, memref<?xi8>) -> ()
//         %r = bufferization.to_tensor %out-view
//
// This runs on tensors, once, before the per-target pipeline in
// toolchain.cpp bufferizes. The C interface of __delegate__, which
// src/runtime/delegate/dispatch.c defines, is
//
//   void _mlir_ciface___delegate__(int64_t id, memref_1d_i8 *in, memref_1d_i8 *out)
//
//===----------------------------------------------------------------------===//

#include "delegate_calls.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace backend {
namespace {

constexpr llvm::StringLiteral kDelegateFn = "__delegate__";
// Written by the middle end (middle-end/linalg/transforms/Passes.h); the
// spellings must agree (src/test/tests/test_export_layering.py).
constexpr llvm::StringLiteral kDelegateIdAttr = "pynqo.delegate_id";

int64_t byteSize(RankedTensorType type) {
  return type.getNumElements() * ((type.getElementTypeBitWidth() + 7) / 8);
}

func::FuncOp declareDispatcher(ModuleOp module) {
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

LogicalResult lower(func::CallOp call, int64_t id, func::FuncOp dispatcher) {
  if (call.getNumOperands() != 1 || call.getNumResults() != 1)
    return call.emitError("a delegate call takes one tensor and returns one");
  auto inType = dyn_cast<RankedTensorType>(call.getOperand(0).getType());
  auto outType = dyn_cast<RankedTensorType>(call.getResult(0).getType());
  if (!inType || !outType || !inType.hasStaticShape() || !outType.hasStaticShape())
    return call.emitError("delegated tensors must have static shapes");

  OpBuilder b(call);
  Location loc = call.getLoc();
  auto dynBytes = cast<MemRefType>(dispatcher.getFunctionType().getInput(1));
  auto [inFlat, inView] = allocBytes(b, loc, inType, dynBytes);
  // The delegate reads this buffer as raw bytes, so %x must really be stored in it.
  bufferization::MaterializeInDestinationOp::create(b, loc, call.getOperand(0), inView)
      .setWritable(true);
  auto [outFlat, outView] = allocBytes(b, loc, outType, dynBytes);
  Value idValue = arith::ConstantIntOp::create(b, loc, id, 64);
  func::CallOp::create(b, loc, dispatcher, ValueRange{idValue, inFlat, outFlat});
  auto result = bufferization::ToTensorOp::create(b, loc, outType, outView);
  // The delegate filled a fresh buffer that nothing else refers to.
  // Later operations may therefore reuse it in place.
  result.setRestrict(true);
  result.setWritable(true);
  call.getResult(0).replaceAllUsesWith(result.getResult());
  call.erase();
  return success();
}

} // namespace

LogicalResult lowerDelegateCalls(ModuleOp module) {
  SmallVector<func::FuncOp> declarations;
  for (auto fn : module.getOps<func::FuncOp>())
    if (fn->hasAttr(kDelegateIdAttr))
      declarations.push_back(fn);
  if (declarations.empty())
    return success();
  func::FuncOp dispatcher = declareDispatcher(module);
  for (func::FuncOp fn : declarations) {
    int64_t id = fn->getAttrOfType<IntegerAttr>(kDelegateIdAttr).getInt();
    SmallVector<func::CallOp> calls;
    if (auto uses = SymbolTable::getSymbolUses(fn, module))
      for (const SymbolTable::SymbolUse &use : *uses) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call)
          return use.getUser()->emitError("a delegate is only ever called");
        calls.push_back(call);
      }
    for (func::CallOp call : calls)
      if (failed(lower(call, id, dispatcher)))
        return failure();
    fn.erase();
  }
  return success();
}

} // namespace backend
