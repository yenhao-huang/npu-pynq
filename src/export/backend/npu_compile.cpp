//===- npu_compile.cpp - the back end: model.mlir -> a package ------------===//
//
//   npu-compile model.mlir --out <dir> --devices cpu,npu
//               --cpus arm-cortex-a9,native --llvm-bin /usr/lib/llvm-22/bin
//
//   model.mlir (linalg on tensors, weights as constants)
//     -> partition: delegates claim and compile their operations; each
//        becomes func.call @__delegate__(id, in, out)
//     -> externalize the remaining weights into weights.bin (`ws` argument)
//     -> CPU codegen, once per CPU model: <dir>/<model>/model.so
//     -> <dir>/compile.json: entry points, delegate table, weights, libraries
//
// src/export/pipeline.py runs it and adds what the front end knows (argument
// names, tokenizer, model metadata) to the package manifest.
//
//===----------------------------------------------------------------------===//

#include "cpu/toolchain.h"
#include "device_register.h"
#include "package.h"
#include "partition.h"

#include "Npu/NpuDialect.h"
#include "Npu/NpuPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> input(cl::Positional, cl::desc("<model.mlir>"), cl::Required);
static cl::opt<std::string> out("out", cl::desc("Package directory"), cl::Required);
static cl::opt<std::string> devices("devices", cl::desc("cpu or cpu,npu"), cl::init("cpu,npu"));
static cl::opt<std::string> cpus("cpus", cl::desc("CPU models: arm-cortex-a9, native"),
                                 cl::init("arm-cortex-a9"));
static cl::opt<std::string> llvmBin("llvm-bin", cl::desc("LLVM 22 tools"),
                                    cl::init("/usr/lib/llvm-22/bin"));
// These defaults are for running by hand.
// src/export/pipeline.py always passes the vendor's values.
static cl::opt<int64_t> rows("rows", cl::init(16));
static cl::opt<int64_t> columns("columns", cl::init(16));
static cl::opt<int64_t> maxK("max-k", cl::init(256));
static cl::opt<int64_t> segmentBytes("segment-bytes", cl::init(4 << 20));
static cl::opt<bool> keepIr("keep-ir", cl::desc("Keep intermediate IR"), cl::init(false));

namespace {

std::vector<std::string> split(const std::string &text) {
  SmallVector<StringRef> parts;
  StringRef(text).split(parts, ',', -1, false);
  std::vector<std::string> result;
  for (StringRef p : parts)
    result.push_back(p.trim().str());
  return result;
}

std::string dtype(Type type) {
  if (type.isF32())
    return "f32";
  if (auto i = dyn_cast<IntegerType>(type))
    return "i" + std::to_string(i.getWidth());
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}

void tensorJson(llvm::json::OStream &j, Type type) {
  j.object([&] {
    auto t = cast<RankedTensorType>(type);
    j.attributeArray("shape", [&] {
      for (int64_t d : t.getShape())
        j.value(d);
    });
    j.attribute("dtype", dtype(t.getElementType()));
  });
}

int fail(const std::string &message) {
  llvm::errs() << "npu-compile: " << message << "\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "NPU back end: model.mlir -> package\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registry.insert<npu::NpuDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerAllPasses();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sources;
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(input, sources, &context);
  if (!module)
    return fail("cannot parse " + input);
  llvm::sys::fs::create_directories(out);

  // Entry points as the front end wrote them (tensors in, tensors out).
  // Read them now, because the passes below rewrite these signatures.
  // compile.json lists these tensors; its `abi` field describes the final calling convention.
  std::vector<std::pair<std::string, FunctionType>> entries;
  for (auto fn : module->getOps<func::FuncOp>())
    if (!fn.isPrivate())
      entries.push_back({fn.getName().str(), fn.getFunctionType()});

  std::string error;
  backend::NpuOptions npu{rows, columns, maxK, segmentBytes};
  auto delegates = backend::registerDevices(split(devices), npu, error);
  if (failed(delegates))
    return fail(error);
  // The order matters. Delegates first take the constants they read, in their own layouts.
  // externalizeWeights then stores only what the CPU code still reads.
  backend::WeightArena arena;
  auto table = backend::partition(*module, *delegates, arena, out);
  if (failed(table))
    return fail("partition failed");
  if (failed(backend::externalizeWeights(*module, arena)))
    return fail("weight externalization failed");
  if (!arena.write(out + "/weights.bin", error))
    return fail(error);
  if (keepIr) {
    std::error_code ec;
    llvm::raw_fd_ostream os(out + "/partitioned.mlir", ec, llvm::sys::fs::OF_Text);
    module->print(os, OpPrintingFlags().elideLargeElementsAttrs(16).elideLargeResourceString(64));
  }

  std::vector<std::pair<std::string, std::string>> libraries;
  for (const std::string &name : split(cpus)) {
    auto target = backend::cpuTarget(name, error);
    if (failed(target))
      return fail(error);
    // Each target needs its own copy: codegen lowers the module in place,
    // and index widths differ per target.
    // All libraries still share one weights.bin, with the same offsets.
    OwningOpRef<ModuleOp> copy = module->clone();
    std::string dir = out + "/" + target->model;
    if (failed(backend::codegen(*copy, *target, dir, llvmBin, keepIr, error)))
      return fail(error);
    libraries.push_back({target->model, target->model + "/model.so"});
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(out + "/compile.json", ec, llvm::sys::fs::OF_Text);
  if (ec)
    return fail("cannot write compile.json");
  llvm::json::OStream j(os, 2);
  j.object([&] {
    j.attribute("format", "npu-compile/1");
    j.attributeArray("devices", [&] {
      for (const std::string &d : split(devices))
        j.value(d);
    });
    j.attributeObject("entry_points", [&] {
      for (auto &[name, type] : entries)
        j.attributeObject(name, [&] {
          j.attributeArray("inputs", [&] {
            for (Type t : type.getInputs())
              tensorJson(j, t);
          });
          j.attributeArray("outputs", [&] {
            for (Type t : type.getResults())
              tensorJson(j, t);
          });
        });
    });
    j.attribute("abi", "ws, inputs..., outputs... as memref descriptors (C interface)");
    j.attributeArray("delegates", [&] {
      for (const backend::DelegateEntry &e : *table)
        j.object([&] {
          j.attribute("id", e.id);
          j.attribute("device", e.device);
          j.attribute("index", e.index);
          j.attribute("input_bytes", e.inputBytes);
          j.attribute("output_bytes", e.outputBytes);
        });
    });
    j.attributeObject("weights", [&] {
      j.attribute("file", "weights.bin");
      j.attribute("bytes", arena.bytes());
      j.attribute("count", arena.count());
    });
    j.attributeObject("libraries", [&] {
      for (auto &[model, path] : libraries)
        j.attribute(model, path);
    });
    j.attributeObject("npu_limits", [&] {
      j.attribute("rows", static_cast<int64_t>(rows));
      j.attribute("columns", static_cast<int64_t>(columns));
      j.attribute("max_k", static_cast<int64_t>(maxK));
    });
  });
  return 0;
}
