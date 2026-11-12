//===- Passes.h - device assignment and partitioning on linalg -*- C++ -*-===//
//
// Graph-level passes on linalg on tensors, after torch-mlir's lowering. They
// decide which device runs each operation and move every device-bound
// operation into its own device function; they never generate device code.
//
// The contract with pynqo-backend is the device functions and these
// attributes. The back end spells them too (backend/partition.cpp,
// backend/pynqo_backend.cpp); src/test/tests/test_export_layering.py checks
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
// On an operation (before partitioning) and on its device function: "npu".
constexpr const char *kDeviceAttr = "pynqo.device";
// On a device function: its __delegate__ id, 0..n-1 over the whole module.
constexpr const char *kDelegateIdAttr = "pynqo.delegate_id";

// Annotate every operation a device in `devices` (in order, except "cpu",
// which runs whatever nobody claims) accepts with pynqo.device; the first
// device that accepts an operation takes it. Sets pynqo.devices on the module.
std::unique_ptr<mlir::Pass> createAssignDevicesPass(std::vector<std::string> devices);
// Outline every annotated operation, with the constants and accumulator
// initialization only it reads, into func.func private @<device>_<id> and
// call it in its place. Ids follow the device order, then module order.
std::unique_ptr<mlir::Pass> createPartitionDevicesPass();

} // namespace middle_end

#endif // MIDDLE_END_LINALG_TRANSFORMS_PASSES_H
