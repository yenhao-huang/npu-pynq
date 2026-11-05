//===- npu_middle_end.cpp - the middle end: torch.mlir -> model.mlir ------===//
//
//   npu-middle-end torch.mlir -o model.mlir
//
//   torch/transforms/   our rewrites on the torch dialect, while layers are recognisable
//   torch-mlir          its two pipelines, to linalg on tensors
//
// The result holds only upstream dialects, which npu-compile parses.
// --emit=torch stops after our rewrites, to inspect or test them.
//
//===----------------------------------------------------------------------===//
#include "torch/transforms/Passes.h"

#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/InitAll.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> input(cl::Positional, cl::desc("<torch.mlir>"), cl::Required);
static cl::opt<std::string> output("o", cl::desc("Output file"), cl::Required);
static cl::opt<std::string> emit("emit", cl::desc("linalg, or torch to stop before lowering"),
                                 cl::init("linalg"));

static constexpr const char *kLowering =
    "torch-function-to-torch-backend-pipeline,"
    "torch-backend-to-linalg-on-tensors-backend-pipeline";

static int fail(const std::string &message) {
  llvm::errs() << "npu-middle-end: " << message << "\n";
  return 1;
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "NPU middle end: torch dialect -> linalg\n");
  if (emit != "linalg" && emit != "torch")
    return fail("--emit must be linalg or torch");

  // The passes torch-mlir's pipelines name.
  torch::registerAllPasses();
  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();
  registerSymbolDCEPass();
  memref::registerExpandOpsPass();
  memref::registerResolveShapedTypeResultDimsPass();

  DialectRegistry registry;
  torch::registerAllDialects(registry);
  torch::registerAllExtensions(registry);
  torch::registerOptionalInputDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sources;
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(input, sources, &context);
  if (!module)
    return fail("cannot parse " + input);

  PassManager pm(&context);
  // BatchNorm first: it matches _int_mm, which the next pass replaces.
  pm.addPass(middle_end::createFuseConvBatchNormPass());
  pm.addPass(middle_end::createDecomposeIntMmPass());
  pm.addPass(middle_end::createAnyDimToAnyDimsPass());
  if (emit == "linalg" && failed(parsePassPipeline(kLowering, pm)))
    return fail("bad lowering pipeline");
  if (failed(pm.run(*module)))
    return fail("lowering failed");

  std::error_code ec;
  llvm::raw_fd_ostream os(output, ec, llvm::sys::fs::OF_None);
  if (ec)
    return fail("cannot write " + output + ": " + ec.message());
  module->print(os);
  os.flush();
  return os.has_error() ? fail("cannot write " + output) : 0;
}
