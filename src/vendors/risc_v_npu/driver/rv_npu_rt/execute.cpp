/* One delegate call: pack A, run the program's segments, unpack C.
 *
 * A and C tile layouts are those of src/vendors/risc_v_npu/spec/layout.py
 * (the same as npu_pynq's): A tiles per (row tile, K chunk), 8-byte aligned;
 * C as one plane of INT32 tiles per K chunk. Unpacking sums the planes, so
 * the caller's output is the M x N INT32 product, the delegated subgraph's
 * result. The sum wraps modulo 2^32, exactly as libnpu_rt's unpackC
 * (src/vendors/npu_pynq/driver/npu_rt/execute.cpp). Each segment's B block is
 * copied to io + b_base before its binary runs.
 */
#include "rv_npu_rt.h"

#include <algorithm>
#include <cstring>

using namespace rv_npu_rt;

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
        int32_t *dst = c + static_cast<int64_t>(mi) * p.tm * p.n + ni * p.tn;
        for (int32_t r = 0; r < rows; ++r) {
          int32_t *row = dst + static_cast<int64_t>(r) * p.n;
          const int32_t *from = tile + r * cols;
          if (kc == 0) {
            std::memcpy(row, from, cols * 4);
            continue;
          }
          // Unsigned, so the sum wraps instead of overflowing.
          for (int32_t j = 0; j < cols; ++j)
            row[j] = static_cast<int32_t>(static_cast<uint32_t>(row[j]) +
                                          static_cast<uint32_t>(from[j]));
        }
      }
      rowBase += nt * cTileStride(p, mi);
    }
  }
}

void stageB(const Program &p, const Segment &g) {
  std::memcpy(static_cast<uint8_t *>(rt.io.virt) + p.bBase, rt.weights + g.weightOffset, g.bBytes);
}

int fail(int64_t index, int32_t status) {
  rt.error = status;
  rt.errorProgram = static_cast<int32_t>(index);
  return status;
}

void finish(const Program &p) {
  rt.calls += 1;
  rt.macs += static_cast<int64_t>(p.m) * p.n * p.k;
}

/* The same call as rv_npu_rt_execute's, with each phase timed (profile.cpp). */
int executeProfiled(int64_t index, const Program &p, void **buffers) {
  enum { PACK_A, STAGE_B, PL, UNPACK_C };
  PhaseStats &s = rt.phases[index];
  const int64_t cycles = rt.cycles, polls = rt.polls, jobs = rt.jobs;
  int64_t staged = 0, pl = 0, stageBBytes = 0;
  uint8_t *io = static_cast<uint8_t *>(rt.io.virt);

  const int64_t begin = nowNs();
  packA(p, static_cast<const int8_t *>(buffers[0]), io);
  int64_t t = nowNs();
  recordEvent(index, PACK_A, begin, t);
  const int64_t packed = t;
  for (const Segment &g : p.segments) {
    const int64_t t0 = t;
    stageB(p, g);
    const int64_t t1 = nowNs();
    int32_t status = pynqRun(g);
    t = nowNs();
    recordEvent(index, STAGE_B, t0, t1);
    recordEvent(index, PL, t1, t);
    staged += t1 - t0;
    pl += t - t1;
    stageBBytes += g.bBytes;
    if (status)
      return fail(index, status);
  }
  finish(p);
  const int64_t ran = t;
  unpackC(p, io + p.cBase, static_cast<int32_t *>(buffers[1]));
  const int64_t end = nowNs();
  recordEvent(index, UNPACK_C, ran, end);

  const int64_t call = end - begin;
  s.callMin = s.calls ? std::min(s.callMin, call) : call;
  s.callMax = std::max(s.callMax, call);
  s.calls += 1;
  s.packA += packed - begin;
  s.stageB += staged;
  s.pl += pl;
  s.unpackC += end - ran;
  s.plCycles += rt.cycles - cycles;
  s.polls += rt.polls - polls;
  s.jobs += rt.jobs - jobs;
  s.packABytes += static_cast<int64_t>(p.m) * p.k;
  s.stageBBytes += stageBBytes;
  s.unpackCBytes += static_cast<int64_t>(cdiv(p.k, p.tk)) * p.m * p.n * 4;
  return 0;
}

} // namespace

extern "C" int rv_npu_rt_execute(int64_t index, void **buffers, int count) {
  if (rt.error)
    return rt.error;
  if (!rt.mmio || index < 0 || index >= static_cast<int64_t>(rt.programs.size()) || count != 2)
    return fail(index, 0xfe);
  const Program &p = rt.programs[index];
  if (rt.profile)
    return executeProfiled(index, p, buffers);
  uint8_t *io = static_cast<uint8_t *>(rt.io.virt);
  packA(p, static_cast<const int8_t *>(buffers[0]), io);
  for (const Segment &g : p.segments) {
    stageB(p, g);
    int32_t status = pynqRun(g);
    if (status)
      return fail(index, status);
  }
  finish(p);
  unpackC(p, io + p.cBase, static_cast<int32_t *>(buffers[1]));
  return 0;
}
