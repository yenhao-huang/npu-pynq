//===- device_register.h - the devices a package targets --------*- C++ -*-===//
#ifndef BACKEND_DEVICE_REGISTER_H
#define BACKEND_DEVICE_REGISTER_H

#include "delegates/npu/lower.h"

#include "mlir/Support/LLVM.h"

#include <memory>
#include <string>
#include <vector>

namespace backend {

// Where the CPU part of a model runs. Models are named <arch>-<model>.
struct CpuTarget {
  std::string model;     // arm-cortex-a9, x86_64-<host cpu>
  std::string triple;
  std::string cpu;       // passed to llc; empty compiles generic code
  std::string features;
  std::string floatAbi;
  unsigned indexBits;
  bool native;           // runs on the machine that compiles it
};

// "arm-cortex-a9" (the PYNQ-Z1, facts from src/vendors/npu_pynq/spec/board.py)
// or "native" (this machine, compiled generic).
mlir::FailureOr<CpuTarget> cpuTarget(const std::string &name, std::string &error);

// Accepted device lists: [cpu] or [cpu, npu]. Returns the delegates in
// claim order (the CPU is never a delegate: it runs what nobody claims).
mlir::FailureOr<std::vector<std::unique_ptr<Delegate>>>
registerDevices(const std::vector<std::string> &devices, const NpuOptions &npu,
                std::string &error);

} // namespace backend

#endif // BACKEND_DEVICE_REGISTER_H
