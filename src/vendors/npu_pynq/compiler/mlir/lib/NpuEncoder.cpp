//===- NpuEncoder.cpp - NPU ISA encoder -----------------------------------===//
#include "Npu/NpuEncoder.h"

#include <algorithm>

namespace mlir::npu {

namespace {
constexpr uint64_t OP_END = 0x01, OP_SHAPE = 0x10, OP_ADDR_A = 0x11,
                   OP_ADDR_B = 0x12, OP_ADDR_C = 0x13, OP_INCR = 0x14,
                   OP_REPEAT = 0x15, OP_GEMM = 0x20;
constexpr uint64_t INC_B = 2, INC_C = 4;

uint64_t word(uint64_t op, uint64_t payload) { return (op << 56) | payload; }
} // namespace

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

void encodeGemm(const GemmPlan &p, int64_t aBase, int64_t bBase,
                int64_t cBase, int64_t niLo, int64_t niHi,
                std::vector<uint64_t> &words, std::vector<int64_t> &bWords) {
  auto u32 = [](int64_t v) { return static_cast<uint64_t>(v) & 0xffffffffu; };
  auto shape = [](int64_t m, int64_t n, int64_t k) {
    return word(OP_SHAPE, (static_cast<uint64_t>(m) << 24) |
                              (static_cast<uint64_t>(n) << 16) |
                              static_cast<uint64_t>(k));
  };
  bool narrowLast = niHi == p.nt() && p.nI(p.nt() - 1) < p.tn;
  int64_t full = niHi - niLo - (narrowLast ? 1 : 0);
  for (int64_t mi = 0; mi < p.mt(); ++mi) {
    for (int64_t kc = 0; kc < p.kt(); ++kc) {
      words.push_back(word(OP_ADDR_A, u32(aBase + p.aOffset(mi, kc))));
      bWords.push_back(static_cast<int64_t>(words.size()));
      words.push_back(word(OP_ADDR_B, u32(bBase + p.bOffset(kc, 0))));
      words.push_back(word(OP_ADDR_C, u32(cBase + p.cOffset(kc, mi, niLo))));
      uint64_t bInc = static_cast<uint64_t>(p.bPanel() / 8);
      uint64_t cInc = static_cast<uint64_t>(p.cTileStride(mi) / 8);
      words.push_back(word(OP_INCR, (bInc << 16) | (cInc << 32)));
      if (full) {
        words.push_back(shape(p.mI(mi), p.tn, p.kC(kc)));
        if (full > 1)
          words.push_back(word(OP_REPEAT, static_cast<uint64_t>(full)));
        words.push_back(word(OP_GEMM, INC_B | INC_C));
      }
      if (narrowLast) {
        words.push_back(shape(p.mI(mi), p.nI(p.nt() - 1), p.kC(kc)));
        words.push_back(word(OP_GEMM, INC_B | INC_C));
      }
    }
  }
}

uint64_t encodeEnd() { return word(OP_END, 0); }

} // namespace mlir::npu
