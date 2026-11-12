/* libnpu_rt: the NPU delegate backend.
 *
 * The runtime's __delegate__ dispatcher calls npu_rt_execute(index, buffers)
 * for every delegate id that belongs to the NPU. The library packs the INT8
 * input into A tiles, runs that program's ISA segments, and unpacks the INT32
 * partial planes into the output. Two backends run programs:
 *
 *   PYNQ  the FPGA: PROG_ADDR / PROG_LEN / DATA_BASE, START, poll STATUS
 *   SIM   a C++ port of the ISA golden model (src/test/model/isa_sim.py) over
 *         host memory, for running packages on the export machine
 *
 * On the board the caller (src/runtime) loads the overlay and allocates the
 * physically contiguous buffers the NPU reaches in PYNQ's standard way, then
 * hands them over in npu_rt_platform; npu_rt_requirements says how large they
 * must be. Without a platform the library runs the simulator on host memory.
 */
#ifndef NPU_RT_H
#define NPU_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  void *virt;      /* CPU address */
  uint32_t phys;   /* what the NPU sees */
  uint32_t size;
} npu_rt_buffer;

typedef struct {
  volatile uint32_t *mmio;  /* the accelerator's register window */
  npu_rt_buffer io;         /* A tiles and C planes (uncached) */
  npu_rt_buffer stage;      /* one segment's weights at a time (uncached) */
  npu_rt_buffer code;       /* the relocated ISA programs */
} npu_rt_platform;

/* Bytes the package needs in io, stage and code. 0 on success. */
int npu_rt_requirements(const char *package_dir, uint32_t sizes[3]);

/* Read programs.txt / programs.bin, map weights.bin, relocate the programs.
 * platform == NULL runs the simulator. 0 on success; see npu_rt_error(). */
int npu_rt_load(const char *package_dir, const npu_rt_platform *platform);

/* Run program `index`: buffers[0] the INT8 input, buffers[1] the INT32
 * partial planes. 0 on success, else the ISA error code. */
int npu_rt_execute(int64_t index, void **buffers, int count);

/* calls, jobs, cycles, macs, polls, error, error_program */
void npu_rt_stats(int64_t out[7]);

/* Phase profiling, off by default. enable != 0 clears the per-program phase
 * counters and starts recording; event_capacity > 0 also keeps up to that many
 * timeline events (later ones are counted as dropped). Times are
 * CLOCK_MONOTONIC nanoseconds, the clock Python's time.perf_counter_ns reads,
 * so phases and engine stages share one timeline. */
void npu_rt_profile(int enable, int64_t event_capacity);
int64_t npu_rt_program_count(void);

/* Per program `index`, NPU_RT_PHASE_FIELDS values, in order:
 *   m, n, k, segments, calls, call_ns_min, call_ns_max,
 *   pack_a_ns, stage_b_ns, pl_ns, unpack_c_ns, pl_cycles, polls, jobs,
 *   pack_a_bytes, stage_b_bytes, unpack_c_bytes, runs, instructions
 * runs counts segment runs (one START each; polls and start overhead are per
 * run); instructions sums ISA_INSTRUCTIONS, the instruction words the scalar
 * core decoded (REPEAT and END included, a REPEAT counted once).
 * Sums cover the successful calls since profiling was enabled. 0 on success. */
#define NPU_RT_PHASE_FIELDS 19
int npu_rt_phase_stats(int64_t index, int64_t out[NPU_RT_PHASE_FIELDS]);

/* Timeline events, NPU_RT_EVENT_FIELDS values each: program index, phase
 * (0 pack_a, 1 stage_b, 2 pl, 3 unpack_c), begin_ns, end_ns. stage_b and pl
 * repeat per segment. Copies up to `max` events into out and returns how many
 * were recorded; *dropped gets the number that did not fit the capacity. */
#define NPU_RT_EVENT_FIELDS 4
int64_t npu_rt_events(int64_t *out, int64_t max, int64_t *dropped);
const char *npu_rt_error(void);
void npu_rt_close(void);

#ifdef __cplusplus
}

#include <string>
#include <vector>

namespace npu_rt {

struct Segment {
  int32_t wordOffset, length, bBytes;
  int64_t weightOffset;
};

struct Program {
  int32_t m, n, k, tm, tn, tk, aBytes, cBase, cBytes;
  std::vector<Segment> segments;
};

struct PhaseStats {
  int64_t calls = 0, callMin = 0, callMax = 0;
  int64_t packA = 0, stageB = 0, pl = 0, unpackC = 0;  /* nanoseconds */
  int64_t plCycles = 0, polls = 0, jobs = 0;
  int64_t packABytes = 0, stageBBytes = 0, unpackCBytes = 0;
  int64_t runs = 0, instructions = 0;
};

struct Event {
  int64_t index, phase, begin, end;
};

struct State {
  bool pynq = false;
  volatile uint32_t *mmio = nullptr;
  npu_rt_buffer io{}, stage{}, code{};
  std::vector<uint8_t> hostIo, hostStage, hostCode;  /* simulator memory */
  const uint8_t *weights = nullptr;                  /* weights.bin, mapped */
  size_t weightBytes = 0;
  std::vector<Program> programs;
  int64_t calls = 0, jobs = 0, cycles = 0, macs = 0, polls = 0, instructions = 0;
  int32_t error = 0, errorProgram = -1;
  std::string message;
  bool profile = false;
  std::vector<PhaseStats> phases;  /* per program, while profiling */
  std::vector<Event> events;       /* reserved to the capacity */
  size_t eventCapacity = 0;
  int64_t eventsDropped = 0;
};

extern State rt;

/* load.cpp */
bool readTable(const std::string &dir, std::vector<Program> &programs,
               uint32_t &words, int32_t limits[3], std::string &error);
/* pynq.cpp */
int32_t pynqRun(const Segment &segment);
/* sim.cpp */
int32_t simRun(const uint64_t *words, int32_t length, uint32_t dataBase);
/* profile.cpp */
int64_t nowNs();
void recordEvent(int64_t index, int64_t phase, int64_t begin, int64_t end);

} // namespace npu_rt
#endif

#endif /* NPU_RT_H */
