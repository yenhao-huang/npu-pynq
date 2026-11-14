//===- Programs.cpp - rv_programs.bin / .txt / .json ----------------------===//
//
// The risc_v_npu counterpart of npu_pynq's writePrograms (Encode.cpp): the
// compiled segment binaries and the table librv_npu_rt reads
// (src/vendors/risc_v_npu/driver/rv_npu_rt/load.cpp).
//
//===----------------------------------------------------------------------===//

#include "RvNpu/RvNpuPasses.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::rvnpu;

bool mlir::rvnpu::writePrograms(
    const std::string &dir, const Limits &limits, int64_t segmentBytes,
    const std::vector<Program> &programs,
    const std::vector<std::vector<std::vector<uint8_t>>> &binaries,
    const std::vector<std::vector<int64_t>> &memBytes, std::string &error) {
  // Lay the binaries out back to back, each at an 8-byte aligned offset.
  std::vector<std::vector<int64_t>> offsets(programs.size());
  int64_t total = 0;
  for (size_t p = 0; p < programs.size(); ++p) {
    if (binaries[p].size() != programs[p].segments.size() ||
        memBytes[p].size() != programs[p].segments.size()) {
      error = "risc_v_npu: one binary per segment expected";
      return false;
    }
    for (const std::vector<uint8_t> &image : binaries[p]) {
      total = align(total, 8);
      offsets[p].push_back(total);
      total += static_cast<int64_t>(image.size());
    }
  }
  total = align(total, 8);

  std::error_code ec;
  {
    llvm::raw_fd_ostream bin(dir + "/rv_programs.bin", ec, llvm::sys::fs::OF_None);
    if (ec) {
      error = "cannot write rv_programs.bin: " + ec.message();
      return false;
    }
    int64_t at = 0;
    for (size_t p = 0; p < programs.size(); ++p)
      for (size_t s = 0; s < binaries[p].size(); ++s) {
        for (; at < offsets[p][s]; ++at)
          bin << '\0';
        bin.write(reinterpret_cast<const char *>(binaries[p][s].data()),
                  binaries[p][s].size());
        at += static_cast<int64_t>(binaries[p][s].size());
      }
    for (; at < total; ++at)
      bin << '\0';
  }

  // rv_programs.txt: the table librv_npu_rt reads, one record per line.
  //   rv-programs 1 <rows> <columns> <max_k> <bin_bytes> <count>
  //   task <id> <m> <n> <k> <tm> <tn> <tk> <a_bytes> <c_base> <c_bytes> <b_base> <segments>
  //   seg <bin_offset> <bin_bytes> <mem_bytes> <weight_offset> <b_bytes>
  {
    llvm::raw_fd_ostream txt(dir + "/rv_programs.txt", ec, llvm::sys::fs::OF_Text);
    if (ec) {
      error = "cannot write rv_programs.txt: " + ec.message();
      return false;
    }
    txt << "rv-programs 1 " << limits.rows << " " << limits.columns << " "
        << limits.maxK << " " << total << " " << programs.size() << "\n";
    for (size_t p = 0; p < programs.size(); ++p) {
      const Program &pr = programs[p];
      txt << "task " << pr.id << " " << pr.plan.m << " " << pr.plan.n << " "
          << pr.plan.k << " " << pr.plan.tm << " " << pr.plan.tn << " "
          << pr.plan.tk << " " << pr.data.aBytes << " " << pr.data.cBase << " "
          << pr.data.cBytes << " " << pr.data.bBase << " " << pr.segments.size()
          << "\n";
      for (size_t s = 0; s < pr.segments.size(); ++s)
        txt << "seg " << offsets[p][s] << " " << binaries[p][s].size() << " "
            << memBytes[p][s] << " " << pr.segments[s].weightOffset << " "
            << pr.segments[s].bBytes << "\n";
    }
  }

  llvm::raw_fd_ostream js(dir + "/rv_programs.json", ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot write rv_programs.json: " + ec.message();
    return false;
  }
  llvm::json::OStream j(js, 1);
  j.object([&] {
    j.attribute("format", "rv-programs/1");
    j.attribute("march", "rv32imc");
    j.attribute("mabi", "ilp32");
    j.attribute("rows", limits.rows);
    j.attribute("columns", limits.columns);
    j.attribute("max_k", limits.maxK);
    j.attribute("bin_bytes", total);
    j.attribute("segment_bytes", segmentBytes);
    j.attributeArray("tasks", [&] {
      for (size_t p = 0; p < programs.size(); ++p) {
        const Program &pr = programs[p];
        j.object([&] {
          j.attribute("id", pr.id);
          j.attribute("m", pr.plan.m);
          j.attribute("n", pr.plan.n);
          j.attribute("k", pr.plan.k);
          j.attribute("tm", pr.plan.tm);
          j.attribute("tn", pr.plan.tn);
          j.attribute("tk", pr.plan.tk);
          j.attribute("jobs", pr.plan.jobs());
          j.attribute("weight_offset", pr.weightOffset);
          j.attribute("a_bytes", pr.data.aBytes);
          j.attribute("c_base", pr.data.cBase);
          j.attribute("c_bytes", pr.data.cBytes);
          j.attribute("b_base", pr.data.bBase);
          j.attribute("b_bytes", pr.data.bBytes);
          j.attributeArray("segments", [&] {
            for (size_t s = 0; s < pr.segments.size(); ++s) {
              const Segment &g = pr.segments[s];
              j.object([&] {
                j.attribute("ni_lo", g.niLo);
                j.attribute("ni_hi", g.niHi);
                j.attribute("jobs", g.jobs);
                j.attribute("bin_offset", offsets[p][s]);
                j.attribute("bin_bytes", static_cast<int64_t>(binaries[p][s].size()));
                j.attribute("mem_bytes", memBytes[p][s]);
                j.attribute("weight_offset", g.weightOffset);
                j.attribute("b_bytes", g.bBytes);
              });
            }
          });
        });
      }
    });
  });
  return true;
}
