//===- AssignDevices.cpp - which device runs each operation ---------------===//
//
// pynqo-assign-devices{devices=cpu,npu}: for each device in order except the
// CPU, every operation its matcher accepts and no earlier device took gets
//
//   pynqo.device = "<device>"
//
// The CPU runs everything left. A matcher is a device's claim rule from its
// vendor folder: a pattern match, with no rewrite and no code generation.
//
//===----------------------------------------------------------------------===//

#include "linalg/transforms/Passes.h"

#include "Npu/NpuMatch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

// One device's claim rule.
struct DeviceMatcher {
  llvm::StringLiteral name;
  bool (*matches)(Operation *);
};

// Every device the middle end can assign to. A new device adds its matcher
// here and a delegate in the back end (backend/device_register.cpp).
const DeviceMatcher kMatchers[] = {
    {"npu", npu::isClaimable},
};

const DeviceMatcher *matcher(StringRef device) {
  for (const DeviceMatcher &m : kMatchers)
    if (m.name == device)
      return &m;
  return nullptr;
}

struct AssignDevicesPass : PassWrapper<AssignDevicesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignDevicesPass)

  explicit AssignDevicesPass(std::vector<std::string> devices) : devices(std::move(devices)) {}

  StringRef getArgument() const final { return "pynqo-assign-devices"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringSet<> seen;
    for (const std::string &device : devices) {
      if (!seen.insert(device).second) {
        module.emitError() << "device '" << device << "' is listed twice";
        return signalPassFailure();
      }
      if (device != "cpu" && !matcher(device)) {
        module.emitError() << "unknown device '" << device << "'; known: cpu, npu";
        return signalPassFailure();
      }
    }
    if (!seen.contains("cpu")) {
      module.emitError() << "the devices must include cpu, which runs what nobody claims";
      return signalPassFailure();
    }

    Builder b(module.getContext());
    SmallVector<Attribute> names;
    for (const std::string &device : devices)
      names.push_back(b.getStringAttr(device));
    module->setAttr(middle_end::kDevicesAttr, b.getArrayAttr(names));

    for (const std::string &device : devices) {
      if (device == "cpu")
        continue;
      const DeviceMatcher *m = matcher(device);
      StringAttr name = b.getStringAttr(device);
      module.walk([&](Operation *op) {
        // First claim wins: an earlier device keeps what it took.
        if (!op->hasAttr(middle_end::kDeviceAttr) && m->matches(op))
          op->setAttr(middle_end::kDeviceAttr, name);
      });
    }
  }

  std::vector<std::string> devices;
};

} // namespace

std::unique_ptr<Pass> middle_end::createAssignDevicesPass(std::vector<std::string> devices) {
  return std::make_unique<AssignDevicesPass>(std::move(devices));
}
