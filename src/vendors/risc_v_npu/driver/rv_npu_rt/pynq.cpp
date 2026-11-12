/* The board backend: start one RISC-V binary on CVA6 and wait for it.
 *
 * CSR window of the npu_accelerator (spec/abi.py, the npu_pynq ABI 2.0 plus
 * RISCV_SCALAR_CORE): PROG_ADDR is CVA6's boot address (the binary in the code
 * buffer), PROG_LEN its size in 64-bit words rounded up, DATA_BASE the io
 * buffer; ISA_CONTROL starts CVA6, ISA_STATUS reports RUNNING / DONE / ERROR /
 * BUSY. The run ends when the program writes DONE (0 = success, otherwise
 * ISA_ERROR = the code) or the MMIO block latches a fault.
 */
#include "rv_npu_rt.h"

#include <cstdio>

using namespace rv_npu_rt;

namespace {

constexpr uint32_t REG_MAGIC = 0x00, REG_VERSION = 0x04, REG_CAPABILITIES = 0x08;
constexpr uint32_t REG_ISA_CONTROL = 0x40, REG_ISA_STATUS = 0x44, REG_PROG_ADDR = 0x48,
                   REG_PROG_LEN = 0x4C, REG_DATA_BASE = 0x50, REG_ISA_ERROR = 0x54,
                   REG_ISA_JOBS = 0x5C, REG_ISA_CYCLES = 0x60;
constexpr uint32_t STATUS_RUNNING = 1, STATUS_DONE = 2, STATUS_ERROR = 4, STATUS_BUSY = 8;

constexpr uint32_t ABI_MAGIC = 0x3155504E, ABI_MAJOR = 2;
/* MATRIX_INT8 | SATURATING_ACCUM_INT32 | CYCLE_COUNTER | ISA_FRONTEND, and
 * RISCV_SCALAR_CORE (bit 7): spec/abi.py REQUIRED_CAPABILITIES. */
constexpr uint32_t CAP_RISCV_SCALAR_CORE = 1u << 7;
constexpr uint32_t REQUIRED_CAPABILITIES = 0x53u | CAP_RISCV_SCALAR_CORE;

constexpr int32_t kTimeout = 0xfd;

inline void barrier() {
#if defined(__arm__) || defined(__aarch64__)
  __asm__ volatile("dsb sy" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif
}

} // namespace

bool rv_npu_rt::checkOverlay(volatile uint32_t *r, std::string &error) {
  if (!r) {
    error = "no CSR window in the platform";
    return false;
  }
  uint32_t magic = r[REG_MAGIC / 4], version = r[REG_VERSION / 4], caps = r[REG_CAPABILITIES / 4];
  char text[160];
  if (magic != ABI_MAGIC || (version >> 16) != ABI_MAJOR) {
    std::snprintf(text, sizeof text, "not an NPU ABI %u overlay (magic 0x%08x, version 0x%08x)",
                  ABI_MAJOR, magic, version);
    error = text;
    return false;
  }
  if (!(caps & CAP_RISCV_SCALAR_CORE)) {
    std::snprintf(text, sizeof text,
                  "the overlay has no CVA6 scalar core (capabilities 0x%08x lack bit 7, "
                  "RISCV_SCALAR_CORE); load the CVA6 build or export for npu",
                  caps);
    error = text;
    return false;
  }
  if ((caps & REQUIRED_CAPABILITIES) != REQUIRED_CAPABILITIES) {
    std::snprintf(text, sizeof text, "the overlay lacks required capabilities 0x%08x",
                  REQUIRED_CAPABILITIES & ~caps);
    error = text;
    return false;
  }
  return true;
}

int32_t rv_npu_rt::pynqRun(const Segment &g) {
  volatile uint32_t *r = rt.mmio;
  barrier();
  r[REG_PROG_ADDR / 4] = rt.code.phys + g.codeOffset;
  r[REG_PROG_LEN / 4] = (g.binBytes + 7u) / 8u;
  r[REG_DATA_BASE / 4] = rt.io.phys;
  barrier();
  r[REG_ISA_CONTROL / 4] = 1;
  const int64_t start = rt.timeoutNs ? nowNs() : 0;
  uint32_t status;
  uint32_t spins = 0;
  for (;;) {
    status = r[REG_ISA_STATUS / 4];
    rt.polls += 1;
    if (!(status & (STATUS_RUNNING | STATUS_BUSY)))
      break;
    if (rt.timeoutNs && (++spins & 0x3ff) == 0 && nowNs() - start > rt.timeoutNs) {
      rt.message = "a RISC-V program did not finish within the timeout (no DONE write?)";
      return kTimeout;
    }
  }
  barrier();
  rt.cycles += r[REG_ISA_CYCLES / 4];
  rt.jobs += r[REG_ISA_JOBS / 4];
  if ((status & STATUS_ERROR) || !(status & STATUS_DONE)) {
    uint32_t code = r[REG_ISA_ERROR / 4];
    return static_cast<int32_t>(code ? code : 0xff);
  }
  return 0;
}
