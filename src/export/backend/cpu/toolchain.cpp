//===- toolchain.cpp - CPU code for one target ----------------------------===//
//
// Upstream MLIR passes, in process, from linalg on tensors to the LLVM
// dialect:
//
//   bufferize (function boundaries included) -> results become out-params
//   -> deallocation -> loops -> LLVM dialect, index width of the target
//
// then LLVM IR, llc for the target CPU, and a shared library linked with the
// MLIR runtime support the generated code may call (memrefCopy). Public
// functions keep MLIR's C interface (_mlir_ciface_<name>, memref
// descriptors); __delegate__ stays undefined for the runtime to provide.
//
//===----------------------------------------------------------------------===//

#include "toolchain.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
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

// memref.copy between non-contiguous memrefs lowers to a call of this
// function (normally MLIR's c_runner_utils). Freestanding, so the same source
// compiles for every target without a sysroot.
constexpr const char *kRuntimeSupport = R"c(
typedef __INTPTR_TYPE__ intptr_t;
typedef __SIZE_TYPE__ size_t;
void *memcpy(void *dst, const void *src, size_t n);
typedef struct { intptr_t rank; void *descriptor; } unranked_memref;

void memrefCopy(intptr_t elem_size, unranked_memref *src, unranked_memref *dst) {
  intptr_t rank = src->rank;
  char *s_base = ((char **)src->descriptor)[1];
  char *d_base = ((char **)dst->descriptor)[1];
  intptr_t *s_meta = (intptr_t *)((char **)src->descriptor + 2);
  intptr_t *d_meta = (intptr_t *)((char **)dst->descriptor + 2);
  intptr_t s_off = s_meta[0], d_off = d_meta[0];
  intptr_t *sizes = s_meta + 1, *s_strides = s_meta + 1 + rank, *d_strides = d_meta + 1 + rank;
  if (rank == 0) {
    memcpy(d_base + d_off * elem_size, s_base + s_off * elem_size, (size_t)elem_size);
    return;
  }
  for (intptr_t i = 0; i < rank; ++i)
    if (sizes[i] == 0) return;
  intptr_t index[8] = {0};
  if (rank > 8) return;
  int contiguous = s_strides[rank - 1] == 1 && d_strides[rank - 1] == 1;
  intptr_t run = contiguous ? sizes[rank - 1] : 1;
  for (;;) {
    intptr_t so = s_off, d = d_off;
    for (intptr_t i = 0; i < rank; ++i) {
      so += index[i] * s_strides[i];
      d += index[i] * d_strides[i];
    }
    memcpy(d_base + d * elem_size, s_base + so * elem_size, (size_t)(run * elem_size));
    intptr_t axis = contiguous ? rank - 2 : rank - 1;
    for (; axis >= 0; --axis) {
      if (++index[axis] < sizes[axis]) break;
      index[axis] = 0;
    }
    if (axis < 0) return;
  }
}
)c";

std::string pipeline(unsigned indexBits) {
  std::string ib = "index-bitwidth=" + std::to_string(indexBits);
  // The pass manager is already anchored on builtin.module.
  return "canonicalize,cse,linalg-fuse-elementwise-ops,"
         "one-shot-bufferize{bufferize-function-boundaries "
         "function-boundary-type-conversion=identity-layout-map},"
         "buffer-results-to-out-params{hoist-static-allocs=true modify-public-functions=true},"
         "canonicalize,buffer-deallocation-pipeline,convert-bufferization-to-memref,"
         "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
         "convert-scf-to-cf,finalize-memref-to-llvm{" + ib + "},"
         "convert-math-to-llvm,convert-arith-to-llvm{" + ib + "},"
         "convert-cf-to-llvm{" + ib + "},convert-func-to-llvm{" + ib + "},"
         "convert-index-to-llvm{" + ib + "},reconcile-unrealized-casts";
}

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
  for (auto fn : module.getOps<func::FuncOp>())
    if (!fn.isPrivate())
      fn->setAttr("llvm.emit_c_interface", UnitAttr::get(module.getContext()));

  PassManager pm(module.getContext());
  if (failed(parsePassPipeline(pipeline(target.indexBits), pm))) {
    error = "bad CPU pass pipeline";
    return failure();
  }
  if (failed(pm.run(module))) {
    error = "lowering to the LLVM dialect failed for " + target.model;
    return failure();
  }
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
  std::string ll = dir + "/model.ll", obj = dir + "/model.o";
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

  std::vector<std::string> llc = {"-mtriple=" + target.triple, "-relocation-model=pic",
                                  "-O3", "-filetype=obj", ll, "-o", obj};
  if (!target.cpu.empty())
    llc.push_back("-mcpu=" + target.cpu);
  if (!target.features.empty())
    llc.push_back("-mattr=" + target.features);
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
  }
  llvm::sys::fs::remove(obj);
  llvm::sys::fs::remove(rtObj);
  return success();
}

} // namespace backend
