/* librv_npu_rt: the risc_v_npu delegate backend (the CVA6 scalar core).
 *
 * The runtime's __delegate__ dispatcher calls rv_npu_rt_execute(index, buffers)
 * for every delegate id that belongs to risc_v_npu. The library packs the INT8
 * input into A tiles, and for each of that program's segments stages the
 * segment's B block, writes the 64-byte parameter block that describes the
 * GEMM and the segment, starts the package's one RISC-V kernel on CVA6 and
 * waits for it; then
 * it unpacks the INT32 partial planes (one per K chunk) into the output,
 * summing them: the output is the M x N product, the delegated subgraph's
 * result, as libnpu_rt's.
 *
 * Modelled on src/vendors/npu_pynq/driver/npu_rt (libnpu_rt); the C interface
 * keeps its shape with the prefix rv_npu_rt_. Differences:
 *
 *   - the package holds one flat position-independent RV32IMC kernel, not ISA
 *     words, and it is never patched: before every segment run the driver
 *     writes the parameter block (shape, tiling, segment and data-block
 *     offsets) at DATA_BASE + 0 and the kernel reads it, so the staged B
 *     block sits inside the data block (io) at the program's b_base instead
 *     of in a separate stage buffer. The platform therefore has two buffers,
 *     io and code, not three;
 *   - PYNQ only. There is no SIM backend: npu_pynq's SIM is a model of the
 *     64-bit ISA and cannot run RISC-V. rv_npu_rt_load(dir, NULL) fails.
 *   - rv_npu_rt_load checks that the overlay reports the CVA6 scalar core
 *     (capability bit 7, RISCV_SCALAR_CORE in spec/abi.py).
 *
 * The package format (rv_programs.bin / rv_programs.txt) is described in
 * src/vendors/risc_v_npu/README.md and in load.cpp.
 */
#ifndef RV_NPU_RT_H
#define RV_NPU_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  void *virt;      /* CPU address */
  uint32_t phys;   /* what the NPU and CVA6 see */
  uint32_t size;
} rv_npu_rt_buffer;

typedef struct {
  volatile uint32_t *mmio;  /* the accelerator's CSR window (AXI-Lite) */
  rv_npu_rt_buffer io;      /* parameter block, A tiles, C planes, staged B block (uncached) */
  rv_npu_rt_buffer code;    /* the RISC-V kernel with room for its .bss and stack */
} rv_npu_rt_platform;

/* Bytes the package needs in io and code. 0 on success. */
int rv_npu_rt_requirements(const char *package_dir, uint32_t sizes[2]);

/* Read rv_programs.txt / rv_programs.bin, map weights.bin, check the overlay,
 * copy the kernel into code. platform == NULL is refused (no SIM backend).
 * 0 on success; see rv_npu_rt_error(). */
int rv_npu_rt_load(const char *package_dir, const rv_npu_rt_platform *platform);

/* Run program `index`: buffers[0] the INT8 M x K input, buffers[1] the INT32
 * M x N output (the K planes summed, wrapping modulo 2^32). 0 on success, else the ISA_ERROR code (a DONE code, an MMIO
 * fault or a matrix-core fault), 0xfd on a timeout, 0xfe on a bad call, 0xff
 * when the run ended without DONE or ERROR. */
int rv_npu_rt_execute(int64_t index, void **buffers, int count);

/* Give up on a run after this many milliseconds (default 10000; 0 waits
 * forever). The hardware has no timeout of its own: a program that never
 * writes DONE keeps CVA6 running until reset. */
void rv_npu_rt_set_timeout_ms(int64_t ms);

/* calls, jobs, cycles, macs, polls, error, error_program */
void rv_npu_rt_stats(int64_t out[7]);

/* Phase profiling, as npu_rt_profile. */
void rv_npu_rt_profile(int enable, int64_t event_capacity);
int64_t rv_npu_rt_program_count(void);

/* Per program `index`, RV_NPU_RT_PHASE_FIELDS values, in order (as npu_rt):
 *   m, n, k, segments, calls, call_ns_min, call_ns_max,
 *   prepare_activation_ns, prepare_weights_ns, npu_run_ns, reduce_output_ns,
 *   npu_run_cycles, polls, jobs,
 *   prepare_activation_bytes, prepare_weights_bytes, reduce_output_bytes */
#define RV_NPU_RT_PHASE_FIELDS 17
int rv_npu_rt_phase_stats(int64_t index, int64_t out[RV_NPU_RT_PHASE_FIELDS]);

/* Timeline events: program index, phase (0 prepare_activation,
 * 1 prepare_weights, 2 npu_run, 3 reduce_output), begin_ns, end_ns. As
 * npu_rt_events. */
#define RV_NPU_RT_EVENT_FIELDS 4
int64_t rv_npu_rt_events(int64_t *out, int64_t max, int64_t *dropped);
const char *rv_npu_rt_error(void);
void rv_npu_rt_close(void);

#ifdef __cplusplus
}

#include <string>
#include <vector>

namespace rv_npu_rt {

/* The parameter block at DATA_BASE + 0: 16 little-endian int32 words, word i
 * at byte 4 * i. Mirrors src/vendors/risc_v_npu/spec/layout.py (PARAM_BYTES
 * and the word indices), the single source of truth, which
 * spec/gen_mmio_header.py also emits into custom_dialect/include/RvNpu/
 * RvNpuMmio.h as RVNPU_PARAM_BYTES and RVNPU_PARAM_<NAME> (byte offsets). That
 * header is not included here: the board builds this library from driver/
 * and spec/ only. A test keeps the numbers in step. */
constexpr uint32_t kParamBytes = 64;   /* A tiles start here (DataLayout.a_base) */
enum ParamWord : uint32_t {
  PARAM_M = 0,
  PARAM_N = 1,
  PARAM_K = 2,
  PARAM_TM = 3,
  PARAM_TN = 4,
  PARAM_TK = 5,
  PARAM_MT = 6,
  PARAM_KT = 7,
  PARAM_NT = 8,
  PARAM_NI_LO = 9,
  PARAM_NI_HI = 10,
  PARAM_A_BASE = 11,
  PARAM_C_BASE = 12,
  PARAM_B_BASE = 13,
  PARAM_B_PANEL = 14,
  PARAM_C_PLANE = 15,
  PARAM_WORDS = 16
};
static_assert(PARAM_WORDS * 4 == kParamBytes, "the parameter block is 16 words");

struct Segment {
  int32_t niLo, niHi;      /* column tiles [niLo, niHi) */
  int64_t weightOffset;    /* in weights.bin */
  int32_t bBytes;          /* (niHi - niLo) * b_panel */
};

struct Program {
  int32_t m, n, k, tm, tn, tk, aBytes, cBase, cBytes, bBase;
  std::vector<Segment> segments;
};

/* The package's one kernel: its image in rv_programs.bin, and its footprint
 * from the load address at run time (image, .bss, stack). It sits at offset 0
 * of the code buffer. */
struct Kernel {
  uint32_t binBytes = 0, memBytes = 0;
};

/* spec/layout.py GemmPlan, the parts the parameter block needs. */
inline int32_t align8(int32_t v) { return (v + 7) & ~7; }
inline int32_t ceilDiv(int32_t a, int32_t b) { return (a + b - 1) / b; }
/* GemmPlan.b_panel: one column tile of B, all K chunks. */
inline int32_t bPanel(const Program &p) {
  int32_t kt = ceilDiv(p.k, p.tk), last = p.k - (kt - 1) * p.tk;
  return (kt - 1) * align8(p.tk * p.tn) + align8(last * p.tn);
}
/* GemmPlan.c_tile_stride(mi) and c_plane_bytes(): one K chunk's C. */
inline int32_t cTileStride(const Program &p, int32_t mi) {
  int32_t rows = p.m - mi * p.tm < p.tm ? p.m - mi * p.tm : p.tm;
  return align8(4 * rows * p.tn);
}
inline int32_t cPlane(const Program &p) {
  int32_t plane = 0, mt = ceilDiv(p.m, p.tm), nt = ceilDiv(p.n, p.tn);
  for (int32_t mi = 0; mi < mt; ++mi)
    plane += nt * cTileStride(p, mi);
  return plane;
}

struct PhaseStats {
  int64_t calls = 0, callMin = 0, callMax = 0;
  /* nanoseconds */
  int64_t prepareActivation = 0, prepareWeights = 0, npuRun = 0, reduceOutput = 0;
  int64_t npuRunCycles = 0, polls = 0, jobs = 0;
  int64_t prepareActivationBytes = 0, prepareWeightsBytes = 0, reduceOutputBytes = 0;
};

struct Event {
  int64_t index, phase, begin, end;
};

struct State {
  volatile uint32_t *mmio = nullptr;
  rv_npu_rt_buffer io{}, code{};
  const uint8_t *weights = nullptr;  /* weights.bin, mapped */
  size_t weightBytes = 0;
  std::vector<Program> programs;
  Kernel kernel;
  int64_t calls = 0, jobs = 0, cycles = 0, macs = 0, polls = 0;
  int32_t error = 0, errorProgram = -1;
  int64_t timeoutNs = 10000000000;
  std::string message;
  bool profile = false;
  std::vector<PhaseStats> phases;
  std::vector<Event> events;
  size_t eventCapacity = 0;
  int64_t eventsDropped = 0;
};

extern State rt;

/* load.cpp */
bool readTable(const std::string &dir, std::vector<Program> &programs, Kernel &kernel,
               int32_t limits[3], std::string &error);
/* pynq.cpp */
bool checkOverlay(volatile uint32_t *mmio, std::string &error);
/* Run the kernel once; the parameter block must already be in io. */
int32_t pynqRun();
/* profile.cpp */
int64_t nowNs();
void recordEvent(int64_t index, int64_t phase, int64_t begin, int64_t end);

} // namespace rv_npu_rt
#endif

#endif /* RV_NPU_RT_H */
