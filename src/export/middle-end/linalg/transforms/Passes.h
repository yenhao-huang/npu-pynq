//===- Passes.h - device assignment and partitioning on linalg -*- C++ -*-===//
//
// Graph-level passes on linalg on tensors, after torch-mlir's lowering. They
// decide which device runs each operation, cut every device-bound operation
// out into its device's module and leave a delegate call in its place; they
// never generate device code.
//
// The contract with pynqo-backend is the delegate declarations, the device
// modules and these attributes. The back end spells them too
// (backend/pynqo_backend.cpp); src/test/tests/test_export_layering.py checks
// the spellings agree.
//
//===----------------------------------------------------------------------===//
#ifndef MIDDLE_END_LINALG_TRANSFORMS_PASSES_H
#define MIDDLE_END_LINALG_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>
#include <vector>

namespace middle_end {

// On the module: the device list the assignment was made for, ["cpu", "npu"].
constexpr const char *kDevicesAttr = "pynqo.devices";
// On an operation (before partitioning), on its delegate declaration and on
// its device's module: "npu".
constexpr const char *kDeviceAttr = "pynqo.device";
// On a delegate declaration and its device function: the __delegate__ id,
// 0..n-1 over the whole model.
constexpr const char *kDelegateIdAttr = "pynqo.delegate_id";
// The delegate declarations in the CPU graph: @__delegate_<id>.
constexpr const char *kDelegatePrefix = "__delegate_";

// Annotate every operation a device in `devices` (in order, except "cpu",
// which runs whatever nobody claims) accepts with pynqo.device; the first
// device that accepts an operation takes it. Sets pynqo.devices on the module.
std::unique_ptr<mlir::Pass> createAssignDevicesPass(std::vector<std::string> devices);
// Cut every annotated operation, with the constants and accumulator
// initialization only it reads, into func.func @<device>_<id> in a nested
// module @<device> {pynqo.device}, and call a bodiless declaration
// @__delegate_<id> in its place. Ids follow the device order, then module
// order.
std::unique_ptr<mlir::Pass> createPartitionDevicesPass();

} // namespace middle_end

#endif // MIDDLE_END_LINALG_TRANSFORMS_PASSES_H
