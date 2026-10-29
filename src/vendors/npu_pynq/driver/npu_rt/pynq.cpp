/* The board backend: start one ISA segment and wait for it.
 *
 * Registers of the npu_accelerator (ABI 1.1): PROG_ADDR / PROG_LEN /
 * DATA_BASE point the front end at a program and its data, ISA_CONTROL
 * starts it, ISA_STATUS reports RUNNING / DONE / ERROR / BUSY.
 */
#include "npu_rt.h"

using namespace npu_rt;

namespace {

constexpr uint32_t REG_ISA_CONTROL = 0x40, REG_ISA_STATUS = 0x44, REG_PROG_ADDR = 0x48,
                   REG_PROG_LEN = 0x4C, REG_DATA_BASE = 0x50, REG_ISA_ERROR = 0x54,
                   REG_ISA_JOBS = 0x5C, REG_ISA_CYCLES = 0x60;
constexpr uint32_t STATUS_RUNNING = 1, STATUS_DONE = 2, STATUS_ERROR = 4, STATUS_BUSY = 8;

inline void barrier() {
#if defined(__arm__)
  __asm__ volatile("dsb sy" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif
}

} // namespace

int32_t npu_rt::pynqRun(const Segment &g) {
  volatile uint32_t *r = rt.mmio;
  barrier();
  r[REG_PROG_ADDR / 4] = rt.code.phys + 8u * static_cast<uint32_t>(g.wordOffset);
  r[REG_PROG_LEN / 4] = static_cast<uint32_t>(g.length);
  r[REG_DATA_BASE / 4] = rt.io.phys;
  barrier();
  r[REG_ISA_CONTROL / 4] = 1;
  uint32_t status;
  do {
    status = r[REG_ISA_STATUS / 4];
    rt.polls += 1;
  } while (status & (STATUS_RUNNING | STATUS_BUSY));
  barrier();
  rt.cycles += r[REG_ISA_CYCLES / 4];
  rt.jobs += r[REG_ISA_JOBS / 4];
  if ((status & STATUS_ERROR) || !(status & STATUS_DONE)) {
    uint32_t code = r[REG_ISA_ERROR / 4];
    return static_cast<int32_t>(code ? code : 0xff);
  }
  return 0;
}
