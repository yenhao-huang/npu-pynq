/* Load a package: the program table, the RISC-V binaries, and the weights.
 *
 * rv_programs.txt (written by the back end's risc_v_npu delegate):
 *   rv-programs 1 <rows> <columns> <max_k> <bin_bytes> <count>
 *   task <id> <m> <n> <k> <tm> <tn> <tk> <a_bytes> <c_base> <c_bytes> <b_base> <segments>
 *   seg <bin_offset> <bin_bytes> <mem_bytes> <weight_offset> <b_bytes>
 *
 * rv_programs.bin holds every segment's flat RV32IMC binary, each starting at
 * an 8-byte aligned <bin_offset>. <mem_bytes> (>= <bin_bytes>) is what the
 * program uses from its load address while it runs: the image plus .bss and
 * the stack that crt0 places after it. Each binary gets a slot of mem_bytes
 * rounded up to 64 in the code buffer, filled once here; .data is not
 * reloaded between runs.
 *
 * The data block at DATA_BASE (the io buffer) is laid out as
 * src/vendors/risc_v_npu/spec/layout.py DataLayout says: A tiles at 0, C
 * planes at <c_base>, the staged B block of the running segment at <b_base>.
 * Nothing is relocated: the program adds these offsets to DATA_BASE itself.
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

bool parse(const std::string &dir, std::vector<Program> &programs, uint32_t &binBytes,
           int32_t limits[3], std::string &error) {
  std::ifstream in(dir + "/rv_programs.txt");
  if (!in) {
    error = "cannot read " + dir + "/rv_programs.txt";
    return false;
  }
  std::string tag;
  int version = 0, count = 0;
  in >> tag >> version >> limits[0] >> limits[1] >> limits[2] >> binBytes >> count;
  if (!in || tag != "rv-programs" || version != 1 || count < 0) {
    error = "rv_programs.txt: unknown format";
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
    if (p.cBase < p.aBytes || p.bBase < p.cBase + p.cBytes || segments < 1) {
      error = "rv_programs.txt: task " + std::to_string(i) + " has an overlapping data layout";
      return false;
    }
    for (int s = 0; s < segments; ++s) {
      Segment g{};
      in >> tag >> g.binOffset >> g.binBytes >> g.memBytes >> g.weightOffset >> g.bBytes;
      if (!in || tag != "seg") {
        error = "rv_programs.txt: bad segment";
        return false;
      }
      if (g.binOffset % 8 || g.binBytes == 0 || g.memBytes < g.binBytes ||
          static_cast<uint64_t>(g.binOffset) + g.binBytes > binBytes || g.bBytes < 0) {
        error = "rv_programs.txt: task " + std::to_string(i) + " has a bad segment record";
        return false;
      }
      p.segments.push_back(g);
    }
  }
  return true;
}

void sizes(std::vector<Program> &programs, uint32_t out[2]) {
  uint32_t io = 64, code = 0;
  for (Program &p : programs) {
    io = std::max<uint32_t>(io, p.cBase + p.cBytes);
    for (Segment &g : p.segments) {
      io = std::max<uint32_t>(io, p.bBase + g.bBytes);
      g.codeOffset = code;
      code += alignUp(g.memBytes, kSlotAlign);
    }
  }
  out[0] = io;
  out[1] = std::max<uint32_t>(8, code);
}

bool inWindow(const rv_npu_rt_buffer &b) {
  return static_cast<uint64_t>(b.phys) + b.size <= kCva6DdrBytes;
}

} // namespace

bool readTable(const std::string &dir, std::vector<Program> &programs, uint32_t &binBytes,
               int32_t limits[3], std::string &error) {
  return parse(dir, programs, binBytes, limits, error);
}

} // namespace rv_npu_rt

using namespace rv_npu_rt;

extern "C" int rv_npu_rt_requirements(const char *package_dir, uint32_t out[2]) {
  std::vector<Program> programs;
  uint32_t binBytes = 0;
  int32_t limits[3];
  if (!readTable(package_dir, programs, binBytes, limits, rt.message))
    return -1;
  sizes(programs, out);
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
  uint32_t binBytes = 0;
  int32_t limits[3];
  if (!parse(dir, rt.programs, binBytes, limits, rt.message))
    return -1;
  uint32_t need[2];
  sizes(rt.programs, need);

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

  /* The binaries, each into its slot; the rest of the slot (.bss, stack) zeroed. */
  std::ifstream bin(dir + "/rv_programs.bin", std::ios::binary);
  std::vector<char> image(binBytes);
  if (!bin.read(image.data(), static_cast<std::streamsize>(binBytes))) {
    rt.message = "cannot read rv_programs.bin";
    return -1;
  }
  uint8_t *code = static_cast<uint8_t *>(rt.code.virt);
  std::memset(code, 0, need[1]);
  for (const Program &p : rt.programs)
    for (const Segment &g : p.segments)
      std::memcpy(code + g.codeOffset, image.data() + g.binOffset, g.binBytes);

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
