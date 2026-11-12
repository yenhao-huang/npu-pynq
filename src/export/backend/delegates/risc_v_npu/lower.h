//===- lower.h - the risc_v_npu delegate ------------------------*- C++ -*-===//
#ifndef BACKEND_DELEGATES_RISC_V_NPU_LOWER_H
#define BACKEND_DELEGATES_RISC_V_NPU_LOWER_H

#include "../delegate.h"

#include <cstdint>
#include <memory>
#include <string>

namespace backend {

struct RiscvNpuOptions {
  // The matrix core: the same as npu's (one array, two scalar cores).
  int64_t rows = 16, columns = 16, maxK = 256;
  int64_t segmentBytes = 4 << 20;
  std::string llvmBin = "/usr/lib/llvm-22/bin";  // ld.lld
  bool keepIr = false;  // keep <out>/risc_v_npu/*.ll, .o, .ld, .elf
};

std::unique_ptr<Delegate> createRiscvNpuDelegate(const RiscvNpuOptions &options);

} // namespace backend

#endif // BACKEND_DELEGATES_RISC_V_NPU_LOWER_H
