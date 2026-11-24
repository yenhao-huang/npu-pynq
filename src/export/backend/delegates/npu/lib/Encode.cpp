//===- Encode.cpp - NPU contractions as ISA programs ----------------------===//
//
// One program per npu.matmul (its delegate id), split into segments: each
// segment covers a run of column tiles whose B panels form one contiguous
// block of at most segmentBytes, so the runtime can stage it. ADDR_B words
// hold offsets inside the segment's block; the loader relocates them to
// wherever the block sits in DDR.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuPasses.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir::npu;

Program mlir::npu::encode(int64_t id, const GemmPlan &plan, int64_t weightOffset,
                          int64_t segmentBytes, std::vector<uint64_t> &words) {
  Program program{id, plan, weightOffset, {}};
  int64_t cBase = align(plan.aBytes(), 64);
  int64_t perSegment = std::max<int64_t>(1, segmentBytes / plan.bPanel());
  for (int64_t lo = 0; lo < plan.nt(); lo += perSegment) {
    int64_t hi = std::min(plan.nt(), lo + perSegment);
    Segment seg{static_cast<int64_t>(words.size()), 0, lo, hi,
                weightOffset + lo * plan.bPanel(), (hi - lo) * plan.bPanel(), {}};
    std::vector<int64_t> bWords;
    encodeGemm(plan, 0, 0, cBase, lo, hi, words, bWords);
    words.push_back(encodeEnd());
    seg.length = static_cast<int64_t>(words.size()) - seg.wordOffset;
    for (int64_t w : bWords)
      seg.relocations.push_back(w - seg.wordOffset);
    program.segments.push_back(seg);
  }
  return program;
}

bool mlir::npu::writePrograms(const std::string &dir, const Limits &limits,
                              int64_t segmentBytes,
                              const std::vector<uint64_t> &words,
                              const std::vector<Program> &programs,
                              std::string &error) {
  std::error_code ec;
  {
    llvm::raw_fd_ostream bin(dir + "/programs.bin", ec, llvm::sys::fs::OF_None);
    if (ec) {
      error = "cannot write programs.bin: " + ec.message();
      return false;
    }
    for (uint64_t w : words)
      for (int byte = 0; byte < 8; ++byte)
        bin << static_cast<char>((w >> (8 * byte)) & 0xff);
  }

  // programs.txt: the table libnpu_rt reads, one record per line.
  //   npu-programs 1 <rows> <columns> <max_k> <words> <count>
  //   task <id> <m> <n> <k> <tm> <tn> <tk> <a_bytes> <c_base> <c_bytes> <segments>
  //   seg <word_offset> <length> <weight_offset> <b_bytes> <nrel> <rel>...
  {
    llvm::raw_fd_ostream txt(dir + "/programs.txt", ec, llvm::sys::fs::OF_Text);
    if (ec) {
      error = "cannot write programs.txt: " + ec.message();
      return false;
    }
    txt << "npu-programs 1 " << limits.rows << " " << limits.columns << " "
        << limits.maxK << " " << words.size() << " " << programs.size() << "\n";
    for (const Program &p : programs) {
      txt << "task " << p.id << " " << p.plan.m << " " << p.plan.n << " "
          << p.plan.k << " " << p.plan.tm << " " << p.plan.tn << " "
          << p.plan.tk << " " << p.plan.aBytes() << " "
          << align(p.plan.aBytes(), 64) << " " << p.plan.cBytes() << " "
          << p.segments.size() << "\n";
      for (const Segment &s : p.segments) {
        txt << "seg " << s.wordOffset << " " << s.length << " "
            << s.weightOffset << " " << s.bBytes << " " << s.relocations.size();
        for (int64_t r : s.relocations)
          txt << " " << r;
        txt << "\n";
      }
    }
  }

  llvm::raw_fd_ostream js(dir + "/programs.json", ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot write programs.json: " + ec.message();
    return false;
  }
  llvm::json::OStream j(js, 1);
  j.object([&] {
    j.attribute("isa_version", 1);
    j.attribute("rows", limits.rows);
    j.attribute("columns", limits.columns);
    j.attribute("max_k", limits.maxK);
    j.attribute("words", static_cast<int64_t>(words.size()));
    j.attribute("segment_bytes", segmentBytes);
    j.attributeArray("tasks", [&] {
      for (const Program &p : programs)
        j.object([&] {
          j.attribute("id", p.id);
          j.attribute("m", p.plan.m);
          j.attribute("n", p.plan.n);
          j.attribute("k", p.plan.k);
          j.attribute("tk", p.plan.tk);
          j.attribute("jobs", p.plan.jobs());
          j.attribute("weight_offset", p.weightOffset);
          j.attribute("segments", static_cast<int64_t>(p.segments.size()));
        });
    });
  });
  return true;
}
