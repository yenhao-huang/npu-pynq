//===- NpuEncoder.h - NPU ISA encoder ---------------------------*- C++ -*-===//
//
// C++ port of src/isa/isa.py and src/isa/layout.py. The two must produce
// identical words; src/compiler/tests checks that.
//
//===----------------------------------------------------------------------===//
#ifndef NPU_NPUENCODER_H
#define NPU_NPUENCODER_H

#include <cstdint>
#include <vector>

namespace mlir::npu {

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

// Appends the program for column tiles [niLo, niHi) of `plan` (without END).
// bBase is where tile niLo's panel sits; ADDR_B word indices are recorded in
// `bWords` so a loader can relocate them.
void encodeGemm(const GemmPlan &plan, int64_t aBase, int64_t bBase,
                int64_t cBase, int64_t niLo, int64_t niHi,
                std::vector<uint64_t> &words, std::vector<int64_t> &bWords);

uint64_t encodeEnd();

} // namespace mlir::npu

#endif // NPU_NPUENCODER_H
