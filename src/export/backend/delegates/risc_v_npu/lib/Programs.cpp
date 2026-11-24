//===- Programs.cpp - rv_programs.bin / .txt / .json ----------------------===//
//
// The risc_v_npu counterpart of npu_pynq's writePrograms (Encode.cpp): the
// compiled kernel and the table librv_npu_rt reads
// (src/vendors/risc_v_npu/driver/rv_npu_rt/load.cpp).
//
// Format 2 (issue #206): rv_programs.bin is the one parameterized kernel
// (LowerToRiscv) that runs every task and segment; the table says what the
// driver writes to the parameter block at DATA_BASE (RVNPU_PARAM_*) and where
// it stages each segment's weights.
//
//===----------------------------------------------------------------------===//

#include "RvNpu/RvNpuMmio.h"
#include "RvNpu/RvNpuPasses.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::rvnpu;

bool mlir::rvnpu::writePrograms(const std::string &dir, const Limits &limits,
                                int64_t segmentBytes, const std::vector<Program> &programs,
                                const std::vector<uint8_t> &binary, int64_t memBytes,
                                std::string &error) {
  if (!programs.empty() &&
      (binary.empty() || memBytes < align(static_cast<int64_t>(binary.size()), 8))) {
    error = "risc_v_npu: a kernel binary with mem_bytes >= bin_bytes expected";
    return false;
  }
  // The image, zero-padded to 8 bytes (the code buffer copies whole words;
  // mem_bytes already covers the padding: .bss and the stack follow, aligned).
  int64_t binBytes = align(static_cast<int64_t>(binary.size()), 8);

  std::error_code ec;
  {
    llvm::raw_fd_ostream bin(dir + "/rv_programs.bin", ec, llvm::sys::fs::OF_None);
    if (ec) {
      error = "cannot write rv_programs.bin: " + ec.message();
      return false;
    }
    bin.write(reinterpret_cast<const char *>(binary.data()), binary.size());
    for (int64_t at = static_cast<int64_t>(binary.size()); at < binBytes; ++at)
      bin << '\0';
  }

  // rv_programs.txt: the table librv_npu_rt reads, one record per line.
  //   rv-programs 2 <rows> <columns> <max_k> <bin_bytes> <mem_bytes> <count>
  //   task <id> <m> <n> <k> <tm> <tn> <tk> <a_bytes> <c_base> <c_bytes> <b_base> <segments>
  //   seg <ni_lo> <ni_hi> <weight_offset> <b_bytes>
  // A is always at RVNPU_PARAM_BYTES, right after the parameter block.
  {
    llvm::raw_fd_ostream txt(dir + "/rv_programs.txt", ec, llvm::sys::fs::OF_Text);
    if (ec) {
      error = "cannot write rv_programs.txt: " + ec.message();
      return false;
    }
    txt << "rv-programs 2 " << limits.rows << " " << limits.columns << " "
        << limits.maxK << " " << binBytes << " " << memBytes << " " << programs.size()
        << "\n";
    for (const Program &pr : programs) {
      txt << "task " << pr.id << " " << pr.plan.m << " " << pr.plan.n << " "
          << pr.plan.k << " " << pr.plan.tm << " " << pr.plan.tn << " "
          << pr.plan.tk << " " << pr.data.aBytes << " " << pr.data.cBase << " "
          << pr.data.cBytes << " " << pr.data.bBase << " " << pr.segments.size()
          << "\n";
      for (const Segment &g : pr.segments)
        txt << "seg " << g.niLo << " " << g.niHi << " " << g.weightOffset << " "
            << g.bBytes << "\n";
    }
  }

  llvm::raw_fd_ostream js(dir + "/rv_programs.json", ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot write rv_programs.json: " + ec.message();
    return false;
  }
  llvm::json::OStream j(js, 1);
  j.object([&] {
    j.attribute("format", "rv-programs/2");
    j.attribute("march", "rv32imc");
    j.attribute("mabi", "ilp32");
    j.attribute("rows", limits.rows);
    j.attribute("columns", limits.columns);
    j.attribute("max_k", limits.maxK);
    j.attribute("bin_bytes", binBytes);
    j.attribute("mem_bytes", memBytes);
    j.attribute("param_bytes", static_cast<int64_t>(RVNPU_PARAM_BYTES));
    j.attribute("segment_bytes", segmentBytes);
    j.attributeArray("tasks", [&] {
      for (const Program &pr : programs) {
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
          j.attribute("a_base", pr.data.aBase);
          j.attribute("a_bytes", pr.data.aBytes);
          j.attribute("c_base", pr.data.cBase);
          j.attribute("c_bytes", pr.data.cBytes);
          j.attribute("b_base", pr.data.bBase);
          j.attribute("b_bytes", pr.data.bBytes);
          j.attributeArray("segments", [&] {
            for (const Segment &g : pr.segments) {
              j.object([&] {
                j.attribute("ni_lo", g.niLo);
                j.attribute("ni_hi", g.niHi);
                j.attribute("jobs", g.jobs);
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
