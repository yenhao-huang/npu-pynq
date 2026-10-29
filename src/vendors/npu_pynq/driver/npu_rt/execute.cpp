/* One delegate call: pack A, run the program's segments, unpack C.
 *
 * A and C tile layouts are those of src/vendors/npu_pynq/isa/layout.py:
 * A tiles per (row tile, K chunk), 8-byte aligned; C as K planes of INT32
 * tiles. The caller's output is [Kt, M, N] INT32 partial planes.
 */
#include "npu_rt.h"

#include <algorithm>
#include <cstring>

using namespace npu_rt;

namespace {

int32_t align8(int32_t v) { return (v + 7) & ~7; }
int32_t cdiv(int32_t a, int32_t b) { return (a + b - 1) / b; }
int32_t rowsOf(const Program &p, int32_t mi) { return std::min(p.tm, p.m - mi * p.tm); }
int32_t colsOf(const Program &p, int32_t ni) { return std::min(p.tn, p.n - ni * p.tn); }
int32_t depthOf(const Program &p, int32_t kc) { return std::min(p.tk, p.k - kc * p.tk); }

int32_t aOffset(const Program &p, int32_t mi, int32_t kc) {
  return mi * cdiv(p.k, p.tk) * align8(p.tm * p.tk) + kc * align8(rowsOf(p, mi) * p.tk);
}
int32_t cTileStride(const Program &p, int32_t mi) { return align8(4 * rowsOf(p, mi) * p.tn); }

void packA(const Program &p, const int8_t *a, uint8_t *dst) {
  int32_t mt = cdiv(p.m, p.tm), kt = cdiv(p.k, p.tk);
  for (int32_t mi = 0; mi < mt; ++mi) {
    int32_t rows = rowsOf(p, mi);
    for (int32_t kc = 0; kc < kt; ++kc) {
      int32_t cols = depthOf(p, kc);
      uint8_t *tile = dst + aOffset(p, mi, kc);
      const int8_t *src = a + static_cast<int64_t>(mi) * p.tm * p.k + kc * p.tk;
      for (int32_t r = 0; r < rows; ++r)
        std::memcpy(tile + r * cols, src + static_cast<int64_t>(r) * p.k, cols);
    }
  }
}

void unpackC(const Program &p, const uint8_t *src, int32_t *c) {
  int32_t mt = cdiv(p.m, p.tm), nt = cdiv(p.n, p.tn), kt = cdiv(p.k, p.tk);
  int32_t plane = 0;
  for (int32_t mi = 0; mi < mt; ++mi)
    plane += nt * cTileStride(p, mi);
  for (int32_t kc = 0; kc < kt; ++kc) {
    int32_t rowBase = 0;
    for (int32_t mi = 0; mi < mt; ++mi) {
      int32_t rows = rowsOf(p, mi);
      for (int32_t ni = 0; ni < nt; ++ni) {
        int32_t cols = colsOf(p, ni);
        const int32_t *tile = reinterpret_cast<const int32_t *>(
            src + kc * plane + rowBase + ni * cTileStride(p, mi));
        int32_t *dst = c + (static_cast<int64_t>(kc) * p.m + mi * p.tm) * p.n + ni * p.tn;
        for (int32_t r = 0; r < rows; ++r)
          std::memcpy(dst + static_cast<int64_t>(r) * p.n, tile + r * cols, cols * 4);
      }
      rowBase += nt * cTileStride(p, mi);
    }
  }
}

} // namespace

extern "C" int npu_rt_execute(int64_t index, void **buffers, int count) {
  if (rt.error)
    return rt.error;
  if (index < 0 || index >= static_cast<int64_t>(rt.programs.size()) || count != 2) {
    rt.error = 0xfe;
    rt.errorProgram = static_cast<int32_t>(index);
    return rt.error;
  }
  const Program &p = rt.programs[index];
  uint8_t *io = static_cast<uint8_t *>(rt.io.virt);
  packA(p, static_cast<const int8_t *>(buffers[0]), io);
  const uint64_t *words = static_cast<const uint64_t *>(rt.code.virt);
  for (const Segment &g : p.segments) {
    std::memcpy(rt.stage.virt, rt.weights + g.weightOffset, g.bBytes);
    int32_t status = rt.pynq ? pynqRun(g) : simRun(words + g.wordOffset, g.length, rt.io.phys);
    if (status) {
      rt.error = status;
      rt.errorProgram = static_cast<int32_t>(index);
      return status;
    }
  }
  rt.calls += 1;
  if (rt.pynq)
    rt.macs += static_cast<int64_t>(p.m) * p.n * p.k;
  unpackC(p, io + p.cBase, static_cast<int32_t *>(buffers[1]));
  return 0;
}
