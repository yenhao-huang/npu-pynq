//===- PartitionDevices.cpp - one device function per assigned operation --===//
//
// pynqo-partition-devices: every operation pynqo-assign-devices annotated
// moves into a private device function, and a call takes its place:
//
//   %acc = linalg.fill ins(%c0) outs(tensor.empty())
//   %y = linalg.matmul {pynqo.device = "npu"} ins(%x, %w = arith.constant) outs(%acc)
//     ->  %y = func.call @npu_<id>(%x)
//
//   func.func private @npu_<id>(%x) -> tensor<...>
//       attributes {pynqo.device = "npu", pynqo.delegate_id = <id> : i64} {
//     %w, %c0, tensor.empty, linalg.fill, linalg.matmul cloned here; return %y
//   }
//
// The function's arguments are the operation's other operands (for the NPU,
// the activation). Constants, operand-free tensor.empty and linalg.fill of
// those (the weight and the accumulator initialization) are cloned into the
// body; the originals stay only where something else still reads them.
//
// Ids are global, by device in pynqo.devices order, then in module order: the
// order the back end used to claim in. The back end keeps them.
//
// A func.call is opaque to the passes that run later (canonicalize, cse,
// elementwise fusion), so nothing fuses into a device function.
//
//===----------------------------------------------------------------------===//

#include "linalg/transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {

// Whether `v` comes from constants alone, through operations cheap to clone
// into a device function: the weight and the accumulator initialization.
bool isCaptured(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<arith::ConstantOp>(def))
    return true;
  if (isa<tensor::EmptyOp>(def))
    return def->getNumOperands() == 0;
  if (isa<linalg::FillOp>(def))
    return llvm::all_of(def->getOperands(), isCaptured);
  return false;
}

// The operations that compute captured operands of `op`, producers first.
void capturedProducers(Value v, llvm::SetVector<Operation *> &ops) {
  Operation *def = v.getDefiningOp();
  if (ops.contains(def))
    return;
  for (Value operand : def->getOperands())
    capturedProducers(operand, ops);
  ops.insert(def);
}

LogicalResult outline(Operation *op, StringRef device, int64_t id, ModuleOp module) {
  llvm::SetVector<Value> arguments;
  llvm::SetVector<Operation *> producers;
  for (Value v : op->getOperands()) {
    if (isCaptured(v))
      capturedProducers(v, producers);
    else
      arguments.insert(v);
  }

  std::string name = (device + "_" + llvm::Twine(id)).str();
  if (module.lookupSymbol(name))
    return op->emitError() << "cannot outline into @" << name << ": the symbol exists";

  MLIRContext *ctx = op->getContext();
  SmallVector<Type> inputs;
  for (Value v : arguments)
    inputs.push_back(v.getType());
  auto type = FunctionType::get(ctx, inputs, op->getResultTypes());
  OpBuilder b = OpBuilder::atBlockEnd(module.getBody());
  auto fn = func::FuncOp::create(b, op->getLoc(), name, type);
  fn.setPrivate();
  fn->setAttr(middle_end::kDeviceAttr, b.getStringAttr(device));
  fn->setAttr(middle_end::kDelegateIdAttr, b.getI64IntegerAttr(id));

  Block *body = fn.addEntryBlock();
  b.setInsertionPointToStart(body);
  IRMapping map;
  for (auto [value, argument] : llvm::zip(arguments, body->getArguments()))
    map.map(value, argument);
  for (Operation *producer : producers)
    b.clone(*producer, map);
  Operation *inner = b.clone(*op, map);
  inner->removeAttr(middle_end::kDeviceAttr);
  func::ReturnOp::create(b, op->getLoc(), inner->getResults());

  b.setInsertionPoint(op);
  auto call = func::CallOp::create(b, op->getLoc(), fn, arguments.getArrayRef());
  op->replaceAllUsesWith(call.getResults());
  op->erase();
  // Consumers before producers; another operation may still read a producer.
  for (Operation *producer : llvm::reverse(producers))
    if (producer->use_empty())
      producer->erase();
  return success();
}

struct PartitionDevicesPass : PassWrapper<PartitionDevicesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionDevicesPass)

  StringRef getArgument() const final { return "pynqo-partition-devices"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto devices = module->getAttrOfType<ArrayAttr>(middle_end::kDevicesAttr);
    if (!devices) {
      module.emitError() << "no " << middle_end::kDevicesAttr << "; run pynqo-assign-devices first";
      return signalPassFailure();
    }
    // Collect first: outlining adds functions to the module being walked.
    llvm::StringMap<SmallVector<Operation *>> assigned;
    WalkResult walk = module.walk([&](Operation *op) {
      auto device = op->getAttrOfType<StringAttr>(middle_end::kDeviceAttr);
      if (!device)
        return WalkResult::advance();
      if (!llvm::is_contained(devices, device)) {
        op->emitError() << "assigned to '" << device.getValue() << "', which is not in "
                        << middle_end::kDevicesAttr;
        return WalkResult::interrupt();
      }
      assigned[device.getValue()].push_back(op);
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return signalPassFailure();

    int64_t id = 0;
    for (Attribute attr : devices) {
      StringRef device = cast<StringAttr>(attr).getValue();
      for (Operation *op : assigned.lookup(device))
        if (failed(outline(op, device, id++, module)))
          return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> middle_end::createPartitionDevicesPass() {
  return std::make_unique<PartitionDevicesPass>();
}
