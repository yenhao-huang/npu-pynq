//===- RvNpuLayout.h - tiled-GEMM layout in DDR -----------------*- C++ -*-===//
//
// Copied from the GemmPlan part of src/vendors/npu_pynq/custom_dialect/
// include/Npu/NpuEncoder.h (a C++ port of spec/layout.py), renamed to
// mlir::rvnpu. The matrix core is the same in both builds, so the A, B and C
// layouts are too; keep the two copies in step. There is no ISA encoder here:
// on the CVA6 build the jobs are issued by a RISC-V program (LowerToRiscv).
//
//===----------------------------------------------------------------------===//
#ifndef RVNPU_RVNPULAYOUT_H
#define RVNPU_RVNPULAYOUT_H

#include <cstdint>

namespace mlir::rvnpu {

struct Limits {
  int64_t rows = 16, columns = 16, maxK = 256;
};

struct GemmPlan {
  int64_t m, n, k, tm, tn, tk;
  static GemmPlan make(int64_t m, int64_t n, int64_t k, const Limits &limits,
                       int64_t tk);
  int64_t mt() const { return (m + tm - 1) / tm; }
  int64_t nt() const { return (n + tn - 1) / tn; }
  int64_t kt() const { return (k + tk - 1) / tk; }
  int64_t mI(int64_t mi) const;
  int64_t nI(int64_t ni) const;
  int64_t kC(int64_t kc) const;
  int64_t aOffset(int64_t mi, int64_t kc) const;
  int64_t aBytes() const;
  int64_t bTileStride(int64_t kc) const;
  int64_t bPanel() const;
  int64_t bOffset(int64_t kc, int64_t ni) const;
  int64_t bBytes() const;
  int64_t cTileStride(int64_t mi) const;
  int64_t cPlaneBytes() const;
  int64_t cOffset(int64_t kc, int64_t mi, int64_t ni) const;
  int64_t cBytes() const;
  int64_t jobs() const { return mt() * nt() * kt(); }
};

int64_t align(int64_t value, int64_t to = 8);

} // namespace mlir::rvnpu

#endif // RVNPU_RVNPULAYOUT_H
