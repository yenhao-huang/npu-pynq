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

struct State {
  bool pynq = false;
  volatile uint32_t *mmio = nullptr;
  npu_rt_buffer io{}, stage{}, code{};
  std::vector<uint8_t> hostIo, hostStage, hostCode;  /* simulator memory */
  const uint8_t *weights = nullptr;                  /* weights.bin, mapped */
  size_t weightBytes = 0;
  std::vector<Program> programs;
  int64_t calls = 0, jobs = 0, cycles = 0, macs = 0, polls = 0;
  int32_t error = 0, errorProgram = -1;
  std::string message;
};

extern State rt;

/* load.cpp */
bool readTable(const std::string &dir, std::vector<Program> &programs,
               uint32_t &words, int32_t limits[3], std::string &error);
/* pynq.cpp */
int32_t pynqRun(const Segment &segment);
/* sim.cpp */
int32_t simRun(const uint64_t *words, int32_t length, uint32_t dataBase);

} // namespace npu_rt
#endif

#endif /* NPU_RT_H */
