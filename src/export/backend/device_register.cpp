//===- device_register.cpp - the devices a package targets ----------------===//

#include "device_register.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

namespace backend {

mlir::FailureOr<CpuTarget> cpuTarget(const std::string &name, std::string &error) {
  if (name == "arm-cortex-a9")
    return CpuTarget{name, "armv7a-unknown-linux-gnueabihf", "cortex-a9",
                     "+neon,+vfp3", "hard", 32, false};
  if (name == "native") {
    llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
    std::string model = triple.getArchName().str() + "-" +
                        llvm::sys::getHostCPUName().str();
    unsigned bits = triple.isArch64Bit() ? 64 : 32;
    return CpuTarget{model, triple.str(), "", "", "", bits, true};
  }
  error = "unknown CPU model '" + name + "'; known: arm-cortex-a9, native";
  return mlir::failure();
}

mlir::FailureOr<std::vector<std::unique_ptr<Delegate>>>
registerDevices(const std::vector<std::string> &devices, const NpuOptions &npu,
                std::string &error) {
  std::vector<std::unique_ptr<Delegate>> delegates;
  if (devices == std::vector<std::string>{"cpu"})
    return delegates;
  if (devices == std::vector<std::string>{"cpu", "npu"}) {
    delegates.push_back(createNpuDelegate(npu));
    return delegates;
  }
  error = "devices must be [cpu] or [cpu, npu]";
  return mlir::failure();
}

} // namespace backend
