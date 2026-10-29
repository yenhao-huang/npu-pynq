/* The export machine's backend: a C++ port of the ISA golden model
 * (src/vendors/npu_pynq/isa/sim.py) over host memory. Addresses are DATA_BASE
 * relative physical addresses, translated through the io and stage buffers.
 */
#include "npu_rt.h"

using namespace npu_rt;

namespace {

uint8_t *host(uint32_t phys, uint32_t bytes) {
  for (const npu_rt_buffer *b : {&rt.io, &rt.stage})
    if (phys >= b->phys && phys - b->phys + bytes <= b->size)
      return static_cast<uint8_t *>(b->virt) + (phys - b->phys);
  return nullptr;
}

} // namespace

int32_t npu_rt::simRun(const uint64_t *words, int32_t length, uint32_t dataBase) {
  uint32_t m = 0, n = 0, k = 0, a = 0, b = 0, c = 0, ai = 0, bi = 0, ci = 0;
  for (int32_t pc = 0; pc < length; ++pc) {
    uint64_t w = words[pc];
    uint32_t op = static_cast<uint32_t>(w >> 56), lo = static_cast<uint32_t>(w),
             hi = static_cast<uint32_t>(w >> 32);
    uint32_t count = 1;
    if (op == 0x15) { /* REPEAT */
      count = lo ? lo : 1;
      if (++pc >= length)
        return 0x15;
      w = words[pc];
      op = static_cast<uint32_t>(w >> 56);
      lo = static_cast<uint32_t>(w);
      hi = static_cast<uint32_t>(w >> 32);
    }
    for (uint32_t rep = 0; rep < count; ++rep) {
      switch (op) {
      case 0x00:
      case 0x02:
        break;
      case 0x01:
        return 0;
      case 0x10:
        m = lo >> 24;
        n = (lo >> 16) & 0xff;
        k = lo & 0xffff;
        break;
      case 0x11: a = lo; break;
      case 0x12: b = lo; break;
      case 0x13: c = lo; break;
      case 0x14:
        ai = (lo & 0xffff) * 8;
        bi = (lo >> 16) * 8;
        ci = (hi & 0xffff) * 8;
        break;
      case 0x20: {
        if (!m || !n || !k || m > 16 || n > 16 || k > 256)
          return 0x11;
        if ((a | b | c) & 7)
          return 0x12;
        const int8_t *A = reinterpret_cast<const int8_t *>(host(dataBase + a, m * k));
        const int8_t *B = reinterpret_cast<const int8_t *>(host(dataBase + b, k * n));
        int32_t *C = reinterpret_cast<int32_t *>(host(dataBase + c, 4 * m * n));
        if (!A || !B || !C)
          return 0x13;
        for (uint32_t i = 0; i < m; ++i)
          for (uint32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (uint32_t q = 0; q < k; ++q)
              acc += static_cast<int32_t>(A[i * k + q]) * static_cast<int32_t>(B[q * n + j]);
            C[i * n + j] = acc;
          }
        rt.jobs += 1;
        rt.macs += static_cast<int64_t>(m) * n * k;
        if (lo & 1) a += ai;
        if (lo & 2) b += bi;
        if (lo & 4) c += ci;
        break;
      }
      default:
        return 0x10;
      }
    }
  }
  return 0x15; /* ran past the end */
}
