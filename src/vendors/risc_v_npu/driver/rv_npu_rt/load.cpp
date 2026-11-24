/* Load a package: the program table, the one GEMM kernel, and the weights.
 *
 * rv_programs.txt (written by the back end's risc_v_npu delegate), format 2:
 *   rv-programs 2 <rows> <columns> <max_k> <bin_bytes> <mem_bytes> <count>
 *   task <id> <m> <n> <k> <tm> <tn> <tk> <a_bytes> <c_base> <c_bytes> <b_base> <segments>
 *   seg <ni_lo> <ni_hi> <weight_offset> <b_bytes>
 *
 * rv_programs.bin is the package's single parameterized RV32IMC kernel
 * (<bin_bytes>). <mem_bytes> (>= <bin_bytes>) is what it uses from its load
 * address while it runs: the image plus .bss and the stack that crt0 places
 * after it. The kernel gets one slot of mem_bytes rounded up to 64 at offset 0
 * of the code buffer, filled once here; .data is not reloaded between runs.
 * Format 1 (one binary per task and segment) is not read any more.
 *
 * The data block at DATA_BASE (the io buffer) is laid out as
 * src/vendors/risc_v_npu/spec/layout.py DataLayout says: the 64-byte
 * parameter block at 0 (rv_npu_rt.h kParam*, written by execute.cpp before
 * every segment run), A tiles at 64, C planes at <c_base> (<c_bytes>: every K
 * plane, which reduceOutput sums into the M x N output), the staged B block of
 * the running segment (column tiles [ni_lo, ni_hi)) at <b_base>. Nothing is
 * relocated: the kernel reads the parameter block and adds its offsets to
 * DATA_BASE itself.
 */
#include "rv_npu_rt.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rv_npu_rt {

State rt;

namespace {

/* CVA6 reaches DDR at 0x0000_0000 - 0x1FFF_FFFF only (spec/abi.py DDR_BYTES). */
constexpr uint64_t kCva6DdrBytes = 0x20000000ull;
constexpr uint32_t kSlotAlign = 64;

uint32_t alignUp(uint32_t v, uint32_t to) { return (v + to - 1) / to * to; }

bool parse(const std::string &dir, std::vector<Program> &programs, Kernel &kernel,
           int32_t limits[3], std::string &error) {
  std::ifstream in(dir + "/rv_programs.txt");
  if (!in) {
    error = "cannot read " + dir + "/rv_programs.txt";
    return false;
  }
  std::string tag;
  int version = 0, count = 0;
  in >> tag >> version;
  if (!in || tag != "rv-programs") {
    error = "rv_programs.txt: unknown format";
    return false;
  }
  if (version != 2) {
    error = "rv_programs.txt: format " + std::to_string(version) +
            " is not supported (this librv_npu_rt reads format 2, one parameterized kernel); "
            "export the model again";
    return false;
  }
  in >> limits[0] >> limits[1] >> limits[2] >> kernel.binBytes >> kernel.memBytes >> count;
  // A package without risc_v_npu tasks has no kernel (bin_bytes 0); every other has one.
  if (!in || count < 0 || (count > 0 && kernel.binBytes == 0) || kernel.memBytes < kernel.binBytes) {
    error = "rv_programs.txt: bad header";
    return false;
  }
  programs.assign(count, Program{});
  for (int i = 0; i < count; ++i) {
    int id = 0, segments = 0;
    Program &p = programs[i];
    in >> tag >> id >> p.m >> p.n >> p.k >> p.tm >> p.tn >> p.tk >> p.aBytes >> p.cBase >>
        p.cBytes >> p.bBase >> segments;
    if (!in || tag != "task" || id != i) {
      error = "rv_programs.txt: tasks out of order";
      return false;
    }
    if (std::min({p.m, p.n, p.k, p.tm, p.tn, p.tk}) < 1 || p.tm > p.m || p.tn > p.n ||
        p.tk > p.k) {
      error = "rv_programs.txt: task " + std::to_string(i) + " has a bad shape";
      return false;
    }
    if (static_cast<int64_t>(p.cBase) < static_cast<int64_t>(kParamBytes) + p.aBytes ||
        static_cast<int64_t>(p.bBase) < static_cast<int64_t>(p.cBase) + p.cBytes || segments < 1) {
      error = "rv_programs.txt: task " + std::to_string(i) + " has an overlapping data layout";
      return false;
    }
    const int32_t nt = ceilDiv(p.n, p.tn), panel = bPanel(p);
    for (int s = 0; s < segments; ++s) {
      Segment g{};
      in >> tag >> g.niLo >> g.niHi >> g.weightOffset >> g.bBytes;
      if (!in || tag != "seg") {
        error = "rv_programs.txt: bad segment";
        return false;
      }
      if (g.niLo < 0 || g.niHi <= g.niLo || g.niHi > nt ||
          static_cast<int64_t>(g.bBytes) != static_cast<int64_t>(g.niHi - g.niLo) * panel) {
        error = "rv_programs.txt: task " + std::to_string(i) + " has a bad segment record";
        return false;
      }
      p.segments.push_back(g);
    }
  }
  return true;
}

void sizes(const std::vector<Program> &programs, const Kernel &kernel, uint32_t out[2]) {
  uint32_t io = kParamBytes;
  for (const Program &p : programs) {
    io = std::max<uint32_t>(io, kParamBytes + p.aBytes);
    io = std::max<uint32_t>(io, p.cBase + p.cBytes);
    for (const Segment &g : p.segments)
      io = std::max<uint32_t>(io, p.bBase + g.bBytes);
  }
  out[0] = io;
  out[1] = std::max<uint32_t>(8, alignUp(kernel.memBytes, kSlotAlign));
}

bool inWindow(const rv_npu_rt_buffer &b) {
  return static_cast<uint64_t>(b.phys) + b.size <= kCva6DdrBytes;
}

} // namespace

bool readTable(const std::string &dir, std::vector<Program> &programs, Kernel &kernel,
               int32_t limits[3], std::string &error) {
  return parse(dir, programs, kernel, limits, error);
}

} // namespace rv_npu_rt

using namespace rv_npu_rt;

extern "C" int rv_npu_rt_requirements(const char *package_dir, uint32_t out[2]) {
  std::vector<Program> programs;
  Kernel kernel;
  int32_t limits[3];
  if (!readTable(package_dir, programs, kernel, limits, rt.message))
    return -1;
  sizes(programs, kernel, out);
  return 0;
}

extern "C" int rv_npu_rt_load(const char *package_dir, const rv_npu_rt_platform *platform) {
  rv_npu_rt_close();
  if (!platform) {
    rt.message = "risc_v_npu has no SIM backend: npu_pynq's SIM is an ISA model and cannot "
                 "run RISC-V";
    return -1;
  }
  std::string dir = package_dir;
  int32_t limits[3];
  if (!parse(dir, rt.programs, rt.kernel, limits, rt.message))
    return -1;
  uint32_t need[2];
  sizes(rt.programs, rt.kernel, need);

  if (!checkOverlay(platform->mmio, rt.message))
    return -1;
  rt.mmio = platform->mmio;
  rt.io = platform->io;
  rt.code = platform->code;
  if (rt.io.size < need[0] || rt.code.size < need[1]) {
    rt.message = "platform buffers are smaller than rv_npu_rt_requirements";
    return -1;
  }
  if (!inWindow(rt.io) || !inWindow(rt.code)) {
    rt.message = "platform buffers lie outside CVA6's DDR window (0x00000000-0x1FFFFFFF)";
    return -1;
  }
  if ((rt.io.phys | rt.code.phys) % 8) {
    rt.message = "platform buffers must be 8-byte aligned";
    return -1;
  }

  /* The kernel into its one slot at offset 0; the rest of the slot (.bss,
   * stack) zeroed. */
  std::ifstream bin(dir + "/rv_programs.bin", std::ios::binary);
  std::vector<char> image(rt.kernel.binBytes);
  if (!bin.read(image.data(), static_cast<std::streamsize>(rt.kernel.binBytes))) {
    rt.message = "cannot read rv_programs.bin";
    return -1;
  }
  uint8_t *code = static_cast<uint8_t *>(rt.code.virt);
  std::memset(code, 0, need[1]);
  std::memcpy(code, image.data(), rt.kernel.binBytes);

  /* weights.bin, mapped read-only; pages load on demand. */
  std::string path = dir + "/weights.bin";
  int fd = open(path.c_str(), O_RDONLY);
  struct stat st {};
  if (fd < 0 || fstat(fd, &st) != 0) {
    if (fd >= 0)
      close(fd);
    rt.message = "cannot open " + path;
    return -1;
  }
  rt.weightBytes = static_cast<size_t>(st.st_size);
  void *mapped = rt.weightBytes ? mmap(nullptr, rt.weightBytes, PROT_READ, MAP_SHARED, fd, 0) : nullptr;
  close(fd);
  if (mapped == MAP_FAILED) {
    rt.weightBytes = 0;
    rt.message = "cannot map " + path;
    return -1;
  }
  rt.weights = static_cast<const uint8_t *>(mapped);
  for (const Program &p : rt.programs)
    for (const Segment &g : p.segments)
      if (static_cast<uint64_t>(g.weightOffset) + static_cast<uint64_t>(g.bBytes) > rt.weightBytes ||
          g.weightOffset < 0) {
        rt.message = "rv_programs.txt: a segment's weights lie outside weights.bin";
        return -1;
      }
  return 0;
}

extern "C" void rv_npu_rt_close(void) {
  if (rt.weights)
    munmap(const_cast<uint8_t *>(rt.weights), rt.weightBytes);
  int64_t timeout = rt.timeoutNs;
  rt = State{};
  rt.timeoutNs = timeout;
}

extern "C" void rv_npu_rt_set_timeout_ms(int64_t ms) { rt.timeoutNs = ms > 0 ? ms * 1000000 : 0; }

extern "C" void rv_npu_rt_stats(int64_t out[7]) {
  out[0] = rt.calls;
  out[1] = rt.jobs;
  out[2] = rt.cycles;
  out[3] = rt.macs;
  out[4] = rt.polls;
  out[5] = rt.error;
  out[6] = rt.errorProgram;
}

extern "C" const char *rv_npu_rt_error(void) { return rt.message.c_str(); }
