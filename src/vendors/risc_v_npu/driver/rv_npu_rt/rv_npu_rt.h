/* librv_npu_rt: the risc_v_npu delegate backend (the CVA6 scalar core).
 *
 * The runtime's __delegate__ dispatcher calls rv_npu_rt_execute(index, buffers)
 * for every delegate id that belongs to risc_v_npu. The library packs the INT8
 * input into A tiles, and for each of that program's segments stages the
 * segment's B block, starts its RISC-V binary on CVA6 and waits for it; then
 * it unpacks the INT32 partial planes into the output.
 *
 * Modelled on src/vendors/npu_pynq/driver/npu_rt (libnpu_rt); the C interface
 * keeps its shape with the prefix rv_npu_rt_. Differences:
 *
 *   - a program segment is a flat position-independent RV32IMC binary, not
 *     ISA words, and is never patched: it reads DATA_BASE and uses constant
 *     offsets from it, so the staged B block sits inside the data block (io)
 *     at the program's b_base instead of in a separate stage buffer. The
 *     platform therefore has two buffers, io and code, not three;
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
  rv_npu_rt_buffer io;      /* A tiles, C planes and the staged B block (uncached) */
  rv_npu_rt_buffer code;    /* the RISC-V binaries with room for their stacks */
} rv_npu_rt_platform;

/* Bytes the package needs in io and code. 0 on success. */
int rv_npu_rt_requirements(const char *package_dir, uint32_t sizes[2]);

/* Read rv_programs.txt / rv_programs.bin, map weights.bin, check the overlay,
 * copy the binaries into code. platform == NULL is refused (no SIM backend).
 * 0 on success; see rv_npu_rt_error(). */
int rv_npu_rt_load(const char *package_dir, const rv_npu_rt_platform *platform);

/* Run program `index`: buffers[0] the INT8 input, buffers[1] the INT32
 * partial planes. 0 on success, else the ISA_ERROR code (a DONE code, an MMIO
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
 *   pack_a_ns, stage_b_ns, pl_ns, unpack_c_ns, pl_cycles, polls, jobs,
 *   pack_a_bytes, stage_b_bytes, unpack_c_bytes */
#define RV_NPU_RT_PHASE_FIELDS 17
int rv_npu_rt_phase_stats(int64_t index, int64_t out[RV_NPU_RT_PHASE_FIELDS]);

/* Timeline events: program index, phase (0 pack_a, 1 stage_b, 2 pl,
 * 3 unpack_c), begin_ns, end_ns. As npu_rt_events. */
#define RV_NPU_RT_EVENT_FIELDS 4
int64_t rv_npu_rt_events(int64_t *out, int64_t max, int64_t *dropped);
const char *rv_npu_rt_error(void);
void rv_npu_rt_close(void);

#ifdef __cplusplus
}

#include <string>
#include <vector>

namespace rv_npu_rt {

struct Segment {
  uint32_t binOffset, binBytes, memBytes;  /* in rv_programs.bin; footprint at run time */
  int64_t weightOffset;                    /* in weights.bin */
  int32_t bBytes;
  uint32_t codeOffset = 0;                 /* where load() put it in the code buffer */
};

struct Program {
  int32_t m, n, k, tm, tn, tk, aBytes, cBase, cBytes, bBase;
  std::vector<Segment> segments;
};

struct PhaseStats {
  int64_t calls = 0, callMin = 0, callMax = 0;
  int64_t packA = 0, stageB = 0, pl = 0, unpackC = 0;  /* nanoseconds */
  int64_t plCycles = 0, polls = 0, jobs = 0;
  int64_t packABytes = 0, stageBBytes = 0, unpackCBytes = 0;
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
bool readTable(const std::string &dir, std::vector<Program> &programs, uint32_t &binBytes,
               int32_t limits[3], std::string &error);
/* pynq.cpp */
bool checkOverlay(volatile uint32_t *mmio, std::string &error);
int32_t pynqRun(const Segment &segment);
/* profile.cpp */
int64_t nowNs();
void recordEvent(int64_t index, int64_t phase, int64_t begin, int64_t end);

} // namespace rv_npu_rt
#endif

#endif /* RV_NPU_RT_H */
