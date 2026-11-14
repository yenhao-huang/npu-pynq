//===- Layout.cpp - tiled-GEMM layout in DDR ------------------------------===//
//
// Copied from the GemmPlan part of src/vendors/npu_pynq/custom_dialect/lib/
// NpuEncoder.cpp, renamed to mlir::rvnpu. Keep the two copies in step.
//
//===----------------------------------------------------------------------===//
#include "RvNpu/RvNpuLayout.h"

#include <algorithm>

namespace mlir::rvnpu {

int64_t align(int64_t value, int64_t to) { return (value + to - 1) / to * to; }

GemmPlan GemmPlan::make(int64_t m, int64_t n, int64_t k, const Limits &limits,
                        int64_t tk) {
  if (tk <= 0)
    tk = limits.maxK;
  return {m, n, k, std::min(m, limits.rows), std::min(n, limits.columns),
          std::min(k, tk)};
}

int64_t GemmPlan::mI(int64_t mi) const { return std::min(tm, m - mi * tm); }
int64_t GemmPlan::nI(int64_t ni) const { return std::min(tn, n - ni * tn); }
int64_t GemmPlan::kC(int64_t kc) const { return std::min(tk, k - kc * tk); }

int64_t GemmPlan::aOffset(int64_t mi, int64_t kc) const {
  return mi * kt() * align(tm * tk) + kc * align(mI(mi) * tk);
}
int64_t GemmPlan::aBytes() const {
  int64_t last = mt() - 1;
  return aOffset(last, kt() - 1) + align(mI(last) * kC(kt() - 1));
}
int64_t GemmPlan::bTileStride(int64_t kc) const { return align(kC(kc) * tn); }
int64_t GemmPlan::bPanel() const {
  return (kt() - 1) * align(tk * tn) + bTileStride(kt() - 1);
}
int64_t GemmPlan::bOffset(int64_t kc, int64_t ni) const {
  return ni * bPanel() + kc * align(tk * tn);
}
int64_t GemmPlan::bBytes() const { return nt() * bPanel(); }
int64_t GemmPlan::cTileStride(int64_t mi) const { return align(4 * mI(mi) * tn); }
int64_t GemmPlan::cPlaneBytes() const {
  int64_t total = 0;
  for (int64_t mi = 0; mi < mt(); ++mi)
    total += nt() * cTileStride(mi);
  return total;
}
int64_t GemmPlan::cOffset(int64_t kc, int64_t mi, int64_t ni) const {
  int64_t row = 0;
  for (int64_t r = 0; r < mi; ++r)
    row += nt() * cTileStride(r);
  return kc * cPlaneBytes() + row + ni * cTileStride(mi);
}
int64_t GemmPlan::cBytes() const { return kt() * cPlaneBytes(); }

} // namespace mlir::rvnpu
