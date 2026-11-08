//===- lower.h - the npu delegate -------------------------------*- C++ -*-===//
#ifndef BACKEND_DELEGATES_NPU_LOWER_H
#define BACKEND_DELEGATES_NPU_LOWER_H

#include "../delegate.h"

#include <cstdint>
#include <memory>

namespace backend {

struct NpuOptions {
  int64_t rows = 16, columns = 16, maxK = 256;
  int64_t segmentBytes = 4 << 20;
};

std::unique_ptr<Delegate> createNpuDelegate(const NpuOptions &options);

} // namespace backend

#endif // BACKEND_DELEGATES_NPU_LOWER_H
