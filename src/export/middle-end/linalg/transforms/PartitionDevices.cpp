//===- PartitionDevices.cpp - cut out device subgraphs, call the delegate -===//
//
// pynqo-partition-devices: every operation pynqo-assign-devices annotated is
// cut out of the graph into its device's module, and a delegate call takes
// its place. The CPU graph keeps only a declaration, so it cannot see how the
// device computes the result:
//
//   %acc = linalg.fill ins(%c0) outs(tensor.empty())
//   %y = linalg.matmul {pynqo.device = "npu"} ins(%x, %w = arith.constant) outs(%acc)
//     ->  %y = func.call @__delegate_<id>(%x)
//         func.func private @__delegate_<id>(tensor<...>) -> tensor<...>
//             attributes {pynqo.device = "npu", pynqo.delegate_id = <id> : i64}
//
//   module @npu attributes {pynqo.device = "npu"} {      (the device module)
//     func.func @npu_<id>(%x) -> tensor<...> attributes {pynqo.delegate_id = <id> : i64} {
//       %w, %c0, tensor.empty, linalg.fill, linalg.matmul cloned here; return %y
//     }
//   }
//
// The function's arguments are the operation's other operands (for the NPU,
// the activation). Constants, operand-free tensor.empty and linalg.fill of
// those (the weight and the accumulator initialization) are cloned into the
// body; the originals stay only where something else still reads them.
//
// Ids are global, by device in pynqo.devices order, then in module order. The
// back end keeps them. The device modules are nested in the module this pass
// runs on; pynqo-middle-end writes each to its own file (ir/delegates/) and
// removes it before the graph passes, which therefore only see the CPU graph
// and the opaque declarations.
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

// The module holding `device`'s subgraphs, nested at the end of `module`.
ModuleOp deviceModule(ModuleOp module, StringRef device) {
  for (auto nested : module.getOps<ModuleOp>())
    if (auto d = nested->getAttrOfType<StringAttr>(middle_end::kDeviceAttr); d && d.getValue() == device)
      return nested;
  OpBuilder b = OpBuilder::atBlockEnd(module.getBody());
  auto nested = ModuleOp::create(b, module.getLoc(), std::optional<StringRef>(device));
  nested->setAttr(middle_end::kDeviceAttr, b.getStringAttr(device));
  return nested;
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

  std::string declared = (llvm::Twine(middle_end::kDelegatePrefix) + llvm::Twine(id)).str();
  std::string name = (device + "_" + llvm::Twine(id)).str();
  if (module.lookupSymbol(declared))
    return op->emitError() << "cannot declare @" << declared << ": the symbol exists";

  MLIRContext *ctx = op->getContext();
  SmallVector<Type> inputs;
  for (Value v : arguments)
    inputs.push_back(v.getType());
  auto type = FunctionType::get(ctx, inputs, op->getResultTypes());

  // The declaration in the CPU graph: no body, so later passes see a call
  // with unknown effects and nothing to fuse with.
  OpBuilder b(ctx);
  ModuleOp nested = deviceModule(module, device);
  b.setInsertionPoint(nested);
  auto decl = func::FuncOp::create(b, op->getLoc(), declared, type);
  decl.setPrivate();
  decl->setAttr(middle_end::kDeviceAttr, b.getStringAttr(device));
  decl->setAttr(middle_end::kDelegateIdAttr, b.getI64IntegerAttr(id));

  // The subgraph, in the device's module.
  b.setInsertionPointToEnd(nested.getBody());
  auto fn = func::FuncOp::create(b, op->getLoc(), name, type);
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
  auto call = func::CallOp::create(b, op->getLoc(), decl, arguments.getArrayRef());
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
