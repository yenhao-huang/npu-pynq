//===- Pack.cpp - weights as the B tiles the NPU reads --------------------===//
//
// Port of pack_b in src/vendors/npu_pynq/isa/layout.py: for every K chunk and
// column tile, the rows x cols block of B, row-major, at GemmPlan::bOffset.
// The layout depends only on (k, n, tk), so a weight shared by contractions
// with different M (prefill and decode) is packed once.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuPasses.h"

#include <cstring>

using namespace mlir::npu;

std::vector<int8_t> mlir::npu::packB(const GemmPlan &p, const int8_t *b) {
  std::vector<int8_t> out(p.bBytes(), 0);
  for (int64_t kc = 0; kc < p.kt(); ++kc) {
    int64_t r0 = kc * p.tk, rows = p.kC(kc);
    for (int64_t ni = 0; ni < p.nt(); ++ni) {
      int64_t c0 = ni * p.tn, cols = p.nI(ni);
      int8_t *dst = out.data() + p.bOffset(kc, ni);
      for (int64_t r = 0; r < rows; ++r)
        std::memcpy(dst + r * cols, b + (r0 + r) * p.n + c0, cols);
    }
  }
  return out;
}
