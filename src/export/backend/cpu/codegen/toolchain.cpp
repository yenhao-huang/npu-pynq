//===- toolchain.cpp - CPU code for one target ----------------------------===//
//
// From the LLVM dialect (cpu/lowering/pipeline.cpp) to model.so: LLVM IR, opt
// (kCanonicalPasses, then kVectorizePasses) and llc for the target CPU, and a
// shared library linked with the MLIR runtime support the generated code may
// call (memrefCopy, runtime_support.h). Between the two opt runs, every
// innermost loop that carries no floating-point value asks to be vectorized
// (vectorize_hints.cpp), so loops become NEON code on the Cortex-A9. Public
// functions keep MLIR's C interface (_mlir_ciface_<name>, memref
// descriptors); __delegate__ stays undefined for the runtime to provide.
//
//===----------------------------------------------------------------------===//

#include "cpu/codegen/toolchain.h"

#include "cpu/codegen/runtime_support.h"
#include "cpu/codegen/vectorize_hints.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace backend {
namespace {

// The IR passes before llc, in two opt runs with vectorizeHints between them.
// The first makes the loops what the loop vectorizer needs: canonical and
// rotated, invariants hoisted (accumulators promoted to registers), and
// invariant branches such as the cf.assert of a fused quantization unswitched
// out of them. The second vectorizes and cleans up. Not -O3: prefill and
// decode are single functions of about 100k instructions and 10k loops, and
// -O3 spends 37 minutes on them (-O1 9 minutes) in loop passes that run once
// per loop over the whole function. These take under a minute together.
constexpr const char *kCanonicalPasses =
    "function(sroa,early-cse,instcombine,simplifycfg,loop-simplify,lcssa,"
    "loop(loop-rotate),loop-mssa(licm,simple-loop-unswitch,licm),loop(indvars))";
constexpr const char *kVectorizePasses = "function(loop-vectorize,early-cse,instcombine,simplifycfg)";

bool run(const std::string &program, std::vector<std::string> args, std::string &error) {
  std::vector<llvm::StringRef> argv;
  argv.push_back(program);
  for (const std::string &a : args)
    argv.push_back(a);
  std::string message;
  int rc = llvm::sys::ExecuteAndWait(program, argv, std::nullopt, {}, 0, 0, &message);
  if (rc != 0) {
    error = llvm::sys::path::filename(program).str() + " failed (" + std::to_string(rc) +
            ") " + message;
    return false;
  }
  return true;
}

bool writeText(const std::string &path, llvm::StringRef text, std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot write " + path + ": " + ec.message();
    return false;
  }
  os << text;
  return true;
}

} // namespace

LogicalResult codegen(ModuleOp module, const CpuTarget &target, const std::string &dir,
                      const std::string &llvmBin, bool keepIr, std::string &error) {
  llvm::sys::fs::create_directories(dir);
  if (keepIr) {
    std::error_code ec;
    llvm::raw_fd_ostream os(dir + "/model.llvm.mlir", ec, llvm::sys::fs::OF_Text);
    module->print(os, OpPrintingFlags().elideLargeElementsAttrs(16));
  }
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    error = "translation to LLVM IR failed";
    return failure();
  }
  std::string ll = dir + "/model.ll", canonical = dir + "/model.canonical.bc",
              hinted = dir + "/model.hinted.bc", bc = dir + "/model.opt.bc", obj = dir + "/model.o";
  std::string rtC = dir + "/runtime_support.c", rtObj = dir + "/runtime_support.o";
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(ll, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      error = "cannot write " + ll;
      return failure();
    }
    llvmModule->print(os, nullptr);
  }
  if (!writeText(rtC, kRuntimeSupport, error))
    return failure();

  // opt and llc see the same target, so the vectorizer's cost model is the
  // target CPU's.
  std::vector<std::string> targetFlags = {"-mtriple=" + target.triple};
  if (!target.cpu.empty())
    targetFlags.push_back("-mcpu=" + target.cpu);
  if (!target.features.empty())
    targetFlags.push_back("-mattr=" + target.features);
  std::vector<std::string> first = targetFlags, second = targetFlags;
  first.insert(first.end(), {"-passes=" + std::string(kCanonicalPasses), ll, "-o", canonical});
  second.insert(second.end(), {"-passes=" + std::string(kVectorizePasses), hinted, "-o", bc});
  if (!run(llvmBin + "/opt", first, error) || !hintBitcode(canonical, hinted, error) ||
      !run(llvmBin + "/opt", second, error))
    return failure();
  std::vector<std::string> llc = targetFlags;
  llc.insert(llc.end(), {"-relocation-model=pic", "-O3", "-filetype=obj", bc, "-o", obj});
  if (!target.floatAbi.empty())
    llc.push_back("-float-abi=" + target.floatAbi);
  if (!run(llvmBin + "/llc", llc, error))
    return failure();

  std::vector<std::string> cc = {"--target=" + target.triple, "-O2", "-fPIC",
                                 "-ffreestanding", "-fno-builtin-memcpy", "-c", rtC,
                                 "-o", rtObj};
  if (!target.cpu.empty())
    cc.push_back("-mcpu=" + target.cpu);
  if (!target.floatAbi.empty())
    cc.push_back("-mfloat-abi=" + target.floatAbi);
  if (!run(llvmBin + "/clang", cc, error))
    return failure();

  // A cross link has no sysroot, so libm and libgcc_s symbols stay undefined.
  // The runtime loads those libraries globally before this one
  // (src/runtime/share/loader.py).
  std::string library = dir + "/model.so";
  bool linked = target.native
                    ? run(llvmBin + "/clang", {"-shared", "-o", library, obj, rtObj, "-lm"}, error)
                    : run(llvmBin + "/ld.lld", {"-shared", "--hash-style=both", "-soname",
                                                "model.so", "-o", library, obj, rtObj},
                          error);
  if (!linked)
    return failure();
  if (!keepIr) {
    llvm::sys::fs::remove(ll);
    llvm::sys::fs::remove(rtC);
    llvm::sys::fs::remove(bc);
  }
  llvm::sys::fs::remove(canonical);
  llvm::sys::fs::remove(hinted);
  llvm::sys::fs::remove(obj);
  llvm::sys::fs::remove(rtObj);
  return success();
}

} // namespace backend
