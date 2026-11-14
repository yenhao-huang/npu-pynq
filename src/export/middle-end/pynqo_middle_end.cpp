//===- pynqo_middle_end.cpp - the middle end: torch.mlir -> model.mlir ------===//
//
//   pynqo-middle-end torch.mlir -o model.mlir --devices cpu,npu
//                    [--delegates-dir <dir>, default <model.mlir's dir>/delegates]
//
//   torch/transforms/   our rewrites on the torch dialect, while layers are recognisable
//   torch-mlir          its two pipelines, to linalg on tensors
//   linalg/transforms/  device assignment, then each assigned operation is cut
//                       out into its device's module and a delegate call
//                       (@__delegate_<id>, a bodiless declaration) takes its place
//   device modules      written to <delegates-dir>/<device>.mlir (@npu_<id>
//                       functions) and removed from the graph
//   graph cleanups      canonicalize, cse, linalg-fuse-elementwise-ops on the CPU
//                       graph: target-agnostic, so they run once here, not once
//                       per CPU target in the back end
//
// Both outputs hold only upstream dialects, which pynqo-backend parses, plus
// the pynqo.* attributes (linalg/transforms/Passes.h). --emit=torch stops
// after our rewrites, to inspect or test them.
//
//===----------------------------------------------------------------------===//
#include "linalg/transforms/Passes.h"
#include "torch/transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"
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
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> input(cl::Positional, cl::desc("<torch.mlir>"), cl::Required);
static cl::opt<std::string> output("o", cl::desc("Output file"), cl::Required);
static cl::opt<std::string> emit("emit", cl::desc("linalg, or torch to stop before lowering"),
                                 cl::init("linalg"));
static cl::opt<std::string> devices("devices", cl::desc("cpu, cpu,npu or cpu,risc_v_npu: the devices to assign to"),
                                    cl::init("cpu"));
static cl::opt<std::string> delegatesDir("delegates-dir",
                                         cl::desc("Where the device modules go (<device>.mlir); "
                                                  "default: <output's directory>/delegates"),
                                         cl::init(""));

static constexpr const char *kLowering =
    "torch-function-to-torch-backend-pipeline,"
    "torch-backend-to-linalg-on-tensors-backend-pipeline";

static int fail(const std::string &message) {
  llvm::errs() << "pynqo-middle-end: " << message << "\n";
  return 1;
}

static bool print(Operation *op, const std::string &path, std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    error = "cannot write " + path + ": " + ec.message();
    return false;
  }
  op->print(os);
  os.flush();
  if (os.has_error()) {
    error = "cannot write " + path;
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Middle end: torch dialect -> fused linalg\n");
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
  if (emit == "linalg") {
    if (failed(parsePassPipeline(kLowering, pm)))
      return fail("bad lowering pipeline");
    // Which device runs each operation; each is cut out into its device's
    // module and the delegate is called in its place. Before any fusion, so
    // device work is never fused with CPU work.
    SmallVector<StringRef> names;
    StringRef(devices).split(names, ',', -1, false);
    std::vector<std::string> list;
    for (StringRef name : names)
      list.push_back(name.trim().str());
    pm.addPass(middle_end::createAssignDevicesPass(list));
    pm.addPass(middle_end::createPartitionDevicesPass());
  }
  if (failed(pm.run(*module)))
    return fail("lowering failed");

  std::string error;
  if (emit == "linalg") {
    // The device modules leave the graph: each goes to its own file, which the
    // device's back end compiles. What stays is the CPU graph with delegate
    // declarations.
    std::string dir = delegatesDir;
    if (dir.empty()) {
      SmallString<256> parent(output.getValue());
      llvm::sys::path::remove_filename(parent);
      llvm::sys::path::append(parent, "delegates");
      dir = std::string(parent);
    }
    SmallVector<ModuleOp> nested(module->getOps<ModuleOp>());
    if (!nested.empty())
      llvm::sys::fs::create_directories(dir);
    for (ModuleOp device : nested) {
      auto name = device->getAttrOfType<StringAttr>(middle_end::kDeviceAttr);
      if (!name)
        return fail("a nested module without pynqo.device");
      device->remove();
      bool written = print(device, dir + "/" + name.getValue().str() + ".mlir", error);
      device->erase();
      if (!written)
        return fail(error);
    }
    // The graph the back end compiles: cleaned up and with elementwise chains
    // fused into their producers. Neither depends on the device or the CPU target.
    PassManager graph(&context);
    graph.addPass(createCanonicalizerPass());
    graph.addPass(createCSEPass());
    graph.addPass(createLinalgElementwiseOpFusionPass());
    if (failed(graph.run(*module)))
      return fail("graph passes failed");
  }
  return print(module->getOperation(), output, error) ? 0 : fail(error);
}
