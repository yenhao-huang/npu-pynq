//===- pipeline.cpp - CPU graph to the LLVM dialect for one target --------===//
//
// Upstream MLIR passes, in process, from linalg on tensors to the LLVM
// dialect:
//
//   loop interchange for unit-stride innermost loops (interchange.cpp)
//   -> bufferize (function boundaries included) -> results become out-params
//   -> deallocation -> loops -> LLVM dialect, index width of the target
//
// cpu/codegen/toolchain.cpp takes the result to LLVM IR and model.so.
//
//===----------------------------------------------------------------------===//

#include "cpu/lowering/pipeline.h"

#include "cpu/lowering/interchange.h"
#include "cpu/profiling/kernel_probes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/FileSystem.h"

using namespace mlir;

namespace backend {
namespace {

// Bufferization, and what follows it to the LLVM dialect. --profile-kernels
// runs the kernel probes before the first part and between the two.
std::string bufferization() {
  return "one-shot-bufferize{bufferize-function-boundaries "
         "function-boundary-type-conversion=identity-layout-map},"
         "buffer-results-to-out-params{hoist-static-allocs=true modify-public-functions=true},"
         "canonicalize,buffer-deallocation-pipeline,convert-bufferization-to-memref";
}

std::string toLlvm(unsigned indexBits) {
  std::string ib = "index-bitwidth=" + std::to_string(indexBits);
  return "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
         "convert-scf-to-cf,finalize-memref-to-llvm{" + ib + "},"
         "convert-math-to-llvm,convert-arith-to-llvm{" + ib + "},"
         "convert-cf-to-llvm{" + ib + "},convert-func-to-llvm{" + ib + "},"
         "convert-index-to-llvm{" + ib + "},reconcile-unrealized-casts";
}

std::string pipeline(unsigned indexBits) {
  // The pass manager is already anchored on builtin.module.
  //
  // These options fix the ABI that the runtime calls:
  //   identity-layout-map           public memrefs stay plain, as the runtime's descriptors assume
  //   buffer-results-to-out-params  results become caller-owned arguments, written directly
  //   index-bitwidth                matches the target's pointer size, used by descriptor fields
  //
  // Graph-level cleanup and elementwise fusion already ran in pynqo-middle-end.
  // The canonicalize below cleans up bufferization, which is per target.
  return bufferization() + "," + toLlvm(indexBits);
}

// Run `passes` (a pipeline string) on `module`.
bool runPipeline(ModuleOp module, const std::string &passes, std::string &error) {
  PassManager pm(module.getContext());
  if (failed(parsePassPipeline(passes, pm))) {
    error = "bad CPU pass pipeline";
    return false;
  }
  if (failed(pm.run(module))) {
    error = "lowering failed";
    return false;
  }
  return true;
}

} // namespace

LogicalResult lowerToLlvm(ModuleOp module, const CpuTarget &target, const std::string &dir,
                          bool profileKernels, std::string &error) {
  llvm::sys::fs::create_directories(dir);
  for (auto fn : module.getOps<func::FuncOp>())
    if (!fn.isPrivate())
      fn->setAttr("llvm.emit_c_interface", UnitAttr::get(module.getContext()));
  interchangeForUnitStride(module);

  if (!profileKernels) {
    if (!runPipeline(module, pipeline(target.indexBits), error)) {
      error = "lowering to the LLVM dialect failed for " + target.model;
      return failure();
    }
    return success();
  }
  // kernel_probes.cpp: kernels on tensors, then the copies bufferization
  // left outside them.
  std::vector<KernelEntry> table;
  if (failed(probeKernels(module, table))) {
    error = "probing the kernels failed";
    return failure();
  }
  if (!runPipeline(module, bufferization(), error)) {
    error = "bufferization failed for " + target.model;
    return failure();
  }
  probeCopies(module, table);
  if (!runPipeline(module, toLlvm(target.indexBits), error)) {
    error = "lowering to the LLVM dialect failed for " + target.model;
    return failure();
  }
  if (!writeKernelTable(dir + "/kernels.json", table, error))
    return failure();
  return success();
}

} // namespace backend
