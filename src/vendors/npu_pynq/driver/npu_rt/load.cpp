/* Load a package: the program table, the ISA words, and the weights.
 *
 * programs.txt (written by the back end's npu delegate):
 *   npu-programs 1 <rows> <columns> <max_k> <words> <count>
 *   task <id> <m> <n> <k> <tm> <tn> <tk> <a_bytes> <c_base> <c_bytes> <segments>
 *   seg <word_offset> <length> <weight_offset> <b_bytes> <nrel> <rel>...
 *
 * Every segment's weights are staged into one window before its program runs
 * (weights.bin stays mapped from the page cache), so each ADDR_B word is
 * relocated once, to the stage window relative to DATA_BASE (the io buffer),
 * wrapping at 32 bits.
 */
#include "npu_rt.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace npu_rt {

State rt;

namespace {

/* Pretend physical addresses for the simulator; any distinct ranges do. */
constexpr uint32_t kSimIo = 0x60000000u, kSimStage = 0x70000000u, kSimCode = 0x78000000u;

struct Relocations {
  std::vector<std::vector<int32_t>> perSegment;
};

bool parse(const std::string &dir, std::vector<Program> &programs, uint32_t &words,
           int32_t limits[3], Relocations *relocations, std::string &error) {
  std::ifstream in(dir + "/programs.txt");
  if (!in) {
    error = "cannot read " + dir + "/programs.txt";
    return false;
  }
  std::string tag;
  int version = 0, count = 0;
  in >> tag >> version >> limits[0] >> limits[1] >> limits[2] >> words >> count;
  if (tag != "npu-programs" || version != 1) {
    error = "programs.txt: unknown format";
    return false;
  }
  programs.assign(count, Program{});
  for (int i = 0; i < count; ++i) {
    int id = 0, segments = 0;
    Program &p = programs[i];
    in >> tag >> id >> p.m >> p.n >> p.k >> p.tm >> p.tn >> p.tk >> p.aBytes >> p.cBase >>
        p.cBytes >> segments;
    if (tag != "task" || id != i) {
      error = "programs.txt: tasks out of order";
      return false;
    }
    for (int s = 0; s < segments; ++s) {
      Segment g{};
      int nrel = 0;
      in >> tag >> g.wordOffset >> g.length >> g.weightOffset >> g.bBytes >> nrel;
      std::vector<int32_t> rel(nrel);
      for (int r = 0; r < nrel; ++r)
        in >> rel[r];
      if (tag != "seg") {
        error = "programs.txt: bad segment";
        return false;
      }
      p.segments.push_back(g);
      if (relocations)
        relocations->perSegment.push_back(std::move(rel));
    }
  }
  if (!in) {
    error = "programs.txt: truncated";
    return false;
  }
  return true;
}

void sizes(const std::vector<Program> &programs, uint32_t words, uint32_t out[3]) {
  uint32_t io = 64, stage = 8;
  for (const Program &p : programs) {
    io = std::max<uint32_t>(io, p.cBase + p.cBytes);
    for (const Segment &g : p.segments)
      stage = std::max<uint32_t>(stage, g.bBytes);
  }
  out[0] = io;
  out[1] = stage;
  out[2] = std::max<uint32_t>(8, 8 * words);
}

npu_rt_buffer hostBuffer(std::vector<uint8_t> &storage, uint32_t size, uint32_t phys) {
  storage.assign(size + 4096, 0);
  uintptr_t at = reinterpret_cast<uintptr_t>(storage.data());
  uint8_t *aligned = storage.data() + ((4096 - at % 4096) % 4096);
  return {aligned, phys, size};
}

} // namespace

bool readTable(const std::string &dir, std::vector<Program> &programs, uint32_t &words,
               int32_t limits[3], std::string &error) {
  return parse(dir, programs, words, limits, nullptr, error);
}

} // namespace npu_rt

using namespace npu_rt;

extern "C" int npu_rt_requirements(const char *package_dir, uint32_t out[3]) {
  std::vector<Program> programs;
  uint32_t words = 0;
  int32_t limits[3];
  if (!readTable(package_dir, programs, words, limits, rt.message))
    return -1;
  sizes(programs, words, out);
  return 0;
}

extern "C" int npu_rt_load(const char *package_dir, const npu_rt_platform *platform) {
  npu_rt_close();
  std::string dir = package_dir;
  uint32_t words = 0;
  int32_t limits[3];
  Relocations relocations;
  if (!parse(dir, rt.programs, words, limits, &relocations, rt.message))
    return -1;
  uint32_t need[3];
  sizes(rt.programs, words, need);

  if (platform) {
    rt.pynq = true;
    rt.mmio = platform->mmio;
    rt.io = platform->io;
    rt.stage = platform->stage;
    rt.code = platform->code;
    if (rt.io.size < need[0] || rt.stage.size < need[1] || rt.code.size < need[2]) {
      rt.message = "platform buffers are smaller than npu_rt_requirements";
      return -1;
    }
  } else {
    rt.pynq = false;
    rt.io = hostBuffer(rt.hostIo, need[0], kSimIo);
    rt.stage = hostBuffer(rt.hostStage, need[1], kSimStage);
    rt.code = hostBuffer(rt.hostCode, need[2], kSimCode);
  }

  /* Program words, relocated: ADDR_B -> stage window, relative to DATA_BASE. */
  std::ifstream bin(dir + "/programs.bin", std::ios::binary);
  std::vector<uint64_t> code(words);
  if (!bin.read(reinterpret_cast<char *>(code.data()), 8 * static_cast<std::streamsize>(words))) {
    rt.message = "cannot read programs.bin";
    return -1;
  }
  uint32_t delta = rt.stage.phys - rt.io.phys;
  size_t s = 0;
  for (const Program &p : rt.programs)
    for (const Segment &g : p.segments) {
      for (int32_t r : relocations.perSegment[s]) {
        uint64_t &w = code[g.wordOffset + r];
        uint32_t lo = static_cast<uint32_t>(w) + delta;
        w = (w & 0xffffffff00000000ull) | lo;
      }
      ++s;
    }
  std::memcpy(rt.code.virt, code.data(), code.size() * 8);

  /* weights.bin, mapped read-only; pages load on demand. */
  std::string path = dir + "/weights.bin";
  int fd = open(path.c_str(), O_RDONLY);
  struct stat st {};
  if (fd < 0 || fstat(fd, &st) != 0) {
    rt.message = "cannot open " + path;
    return -1;
  }
  rt.weightBytes = static_cast<size_t>(st.st_size);
  void *mapped = rt.weightBytes ? mmap(nullptr, rt.weightBytes, PROT_READ, MAP_SHARED, fd, 0) : nullptr;
  close(fd);
  if (mapped == MAP_FAILED) {
    rt.message = "cannot map " + path;
    return -1;
  }
  rt.weights = static_cast<const uint8_t *>(mapped);
  return 0;
}

extern "C" void npu_rt_close(void) {
  if (rt.weights)
    munmap(const_cast<uint8_t *>(rt.weights), rt.weightBytes);
  rt = State{};
}

extern "C" void npu_rt_stats(int64_t out[7]) {
  out[0] = rt.calls;
  out[1] = rt.jobs;
  out[2] = rt.cycles;
  out[3] = rt.macs;
  out[4] = rt.polls;
  out[5] = rt.error;
  out[6] = rt.errorProgram;
}

extern "C" const char *npu_rt_error(void) { return rt.message.c_str(); }
