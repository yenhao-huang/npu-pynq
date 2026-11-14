//===- pynqo_backend.cpp - the back end: model.mlir -> a package ------------===//
//
//   pynqo-backend model.mlir --out <dir> --devices cpu,npu   (or cpu,risc_v_npu)
//               --cpus arm-cortex-a9,native --llvm-bin /usr/lib/llvm-22/bin
//               [--delegates-dir <dir>, default <model.mlir's dir>/delegates]
//
//   model.mlir               the CPU graph: fused linalg on tensors, weights as
//                            constants, a call to a bodiless @__delegate_<id>
//                            where pynqo-middle-end cut a device subgraph out;
//                            assigned for pynqo.devices
//   delegates/<device>.mlir  that device's subgraphs, one @<device>_<id> each
//
//     -> check pynqo.devices against --devices, and that the declarations and
//        the device modules match (ids 0..n-1, one subgraph each, same types)
//     -> each delegate compiles its own module (the NPU: programs.*, B tiles;
//        risc_v_npu: rv_programs.*, B tiles)
//     -> CPU: delegate calls become func.call @__delegate__(id, in, out)
//     -> externalize the remaining weights into weights.bin (`ws` argument)
//     -> CPU codegen, once per CPU model: <dir>/<model>/model.so
//     -> <dir>/compile.json: entry points, delegate table, weights, libraries
//
// src/export/pipeline.py runs it and adds what the front end knows (argument
// names, tokenizer, model metadata) to the package manifest.
//
//===----------------------------------------------------------------------===//

#include "cpu/delegate_calls.h"
#include "cpu/toolchain.h"
#include "device_register.h"
#include "package.h"

#include "Npu/NpuDialect.h"
#include "Npu/NpuPasses.h"
#include "RvNpu/RvNpuDialect.h"

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
#include "llvm/Support/Path.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> input(cl::Positional, cl::desc("<model.mlir>"), cl::Required);
static cl::opt<std::string> out("out", cl::desc("Package directory"), cl::Required);
static cl::opt<std::string> devices("devices", cl::desc("cpu, cpu,npu or cpu,risc_v_npu"), cl::init("cpu,npu"));
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
static cl::opt<std::string> delegatesDir("delegates-dir",
                                         cl::desc("The device modules (<device>.mlir); "
                                                  "default: <model.mlir's directory>/delegates"),
                                         cl::init(""));

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
  llvm::errs() << "pynqo-backend: " << message << "\n";
  return 1;
}

// Written by the middle end (middle-end/linalg/transforms/Passes.h); the
// spellings must agree (src/test/tests/test_export_layering.py).
constexpr const char *kDevicesAttr = "pynqo.devices";
constexpr const char *kDeviceAttr = "pynqo.device";
constexpr const char *kDelegateIdAttr = "pynqo.delegate_id";

// One __delegate__ id: the device it belongs to, that device's own index for
// it (its program number), and the bytes the runtime passes in and out.
struct DelegateEntry {
  int64_t id;
  std::string device;
  int64_t index;
  int64_t inputBytes, outputBytes;
  FunctionType type;
};

int64_t byteSize(Type type) {
  auto t = cast<RankedTensorType>(type);
  return t.getNumElements() * ((t.getElementTypeBitWidth() + 7) / 8);
}

bool isStatic(Type t) {
  auto tensor = dyn_cast<RankedTensorType>(t);
  return tensor && tensor.hasStaticShape();
}

// The delegate declarations of the CPU graph, in id order: bodiless private
// functions with a device and an id, one static tensor in and one out, ids
// 0..n-1 without gaps or repeats.
FailureOr<std::vector<DelegateEntry>> declarations(ModuleOp module, std::string &error) {
  std::vector<DelegateEntry> table;
  for (auto fn : module.getOps<func::FuncOp>()) {
    auto id = fn->getAttrOfType<IntegerAttr>(kDelegateIdAttr);
    if (!id)
      continue;
    auto device = fn->getAttrOfType<StringAttr>(kDeviceAttr);
    FunctionType type = fn.getFunctionType();
    if (!device || !fn.isExternal() || !fn.isPrivate() || type.getNumInputs() != 1 ||
        type.getNumResults() != 1 || !isStatic(type.getInput(0)) || !isStatic(type.getResult(0))) {
      error = "@" + fn.getName().str() + ": a delegate declaration is a private function without "
              "a body, with a pynqo.device, one static tensor argument and one static tensor result";
      return failure();
    }
    table.push_back({id.getInt(), device.getValue().str(), -1, byteSize(type.getInput(0)),
                     byteSize(type.getResult(0)), type});
  }
  llvm::sort(table, [](const DelegateEntry &a, const DelegateEntry &b) { return a.id < b.id; });
  for (size_t i = 0; i < table.size(); ++i)
    if (table[i].id != static_cast<int64_t>(i)) {
      error = "delegate ids must be 0.." + std::to_string(table.size() - 1) +
              " without gaps or repeats; found " + std::to_string(table[i].id) + " at position " +
              std::to_string(i);
      return failure();
    }
  return table;
}

// `device`'s module holds exactly one function per declaration of that
// device, with the declaration's id and type, and nothing else with an id.
LogicalResult matches(ModuleOp deviceModule, const std::string &device,
                      const std::vector<DelegateEntry> &table, const std::string &path,
                      std::string &error) {
  auto attr = deviceModule->getAttrOfType<StringAttr>(kDeviceAttr);
  if (!attr || attr.getValue() != device) {
    error = path + " is not the module of device '" + device + "'";
    return failure();
  }
  std::vector<int> seen(table.size(), 0);
  for (auto fn : deviceModule.getOps<func::FuncOp>()) {
    auto id = fn->getAttrOfType<IntegerAttr>(kDelegateIdAttr);
    if (!id)
      continue;
    int64_t i = id.getInt();
    if (i < 0 || i >= static_cast<int64_t>(table.size()) || table[i].device != device ||
        table[i].type != fn.getFunctionType() || fn.isExternal() || seen[i]++) {
      error = path + ": @" + fn.getName().str() + " (delegate " + std::to_string(i) +
              ") matches no declaration of device '" + device + "' in model.mlir, or repeats one";
      return failure();
    }
  }
  for (const DelegateEntry &e : table)
    if (e.device == device && !seen[e.id]) {
      error = path + " has no subgraph for delegate " + std::to_string(e.id);
      return failure();
    }
  return success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Back end: model.mlir -> package\n");

  // 0. Context and input.
  // model.mlir holds only upstream dialects, so torch-mlir is not needed here.
  // The npu and rvnpu dialects are registered for the delegates' npu.matmul
  // and rvnpu.matmul; the LLVM IR translations for CPU codegen; all passes
  // because codegen builds its pipeline from a string.
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registry.insert<npu::NpuDialect, rvnpu::RvNpuDialect>();
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

  // 1. Entry points as the front end wrote them (tensors in, tensors out).
  // Read them now, because the passes below rewrite these signatures.
  // compile.json lists these tensors; its `abi` field describes the final calling convention.
  std::vector<std::pair<std::string, FunctionType>> entries;
  for (auto fn : module->getOps<func::FuncOp>())
    if (!fn.isPrivate())
      entries.push_back({fn.getName().str(), fn.getFunctionType()});

  // 2. The delegates for --devices: none for [cpu], the NPU for [cpu, npu],
  // the CVA6 build of the NPU for [cpu, risc_v_npu]
  // (device_register.cpp). The rest of the back end sees only the Delegate
  // interface (delegates/delegate.h).
  std::string error;
  backend::NpuOptions npu{rows, columns, maxK, segmentBytes};
  backend::RiscvNpuOptions riscvNpu{rows, columns, maxK, segmentBytes, llvmBin, keepIr};
  auto delegates = backend::registerDevices(split(devices), npu, riscvNpu, error);
  if (failed(delegates))
    return fail(error);
  // The middle end assigned the operations for one device list
  // (pynqo-middle-end --devices); a model.mlir assigned for another must not
  // be compiled for these delegates. The attribute is consumed here.
  std::vector<std::string> assigned;
  if (auto list = (*module)->getAttrOfType<ArrayAttr>(kDevicesAttr))
    for (Attribute a : list)
      if (auto s = dyn_cast<StringAttr>(a))
        assigned.push_back(s.getValue().str());
  if (assigned != split(devices))
    return fail("model.mlir was assigned for devices [" + llvm::join(assigned, ",") +
                "] but --devices is [" + llvm::join(split(devices), ",") +
                "]; run pynqo-middle-end with --devices " + devices.getValue());
  (*module)->removeAttr(kDevicesAttr);

  // 3. The delegate table, from the declarations the middle end left in the
  // CPU graph: id -> device, and the bytes in and out (the subgraph's own
  // argument and result types).
  auto table = declarations(*module, error);
  if (failed(table))
    return fail(error);
  for (const DelegateEntry &e : *table)
    if (!llvm::any_of(*delegates, [&](const auto &d) { return d->name() == e.device; }))
      return fail("delegate " + std::to_string(e.id) + " is for device '" + e.device +
                  "', which has no delegate in --devices");

  // 4. Each delegate compiles its device's module, in --devices order. The
  // order matters: delegates first take the constants they read, in their
  // own layouts; externalizeWeights then stores only what the CPU code still
  // reads. Both write into one arena, which becomes weights.bin.
  backend::WeightArena arena;
  std::string dir = delegatesDir;
  if (dir.empty()) {
    SmallString<256> parent(input.getValue());
    llvm::sys::path::remove_filename(parent);
    llvm::sys::path::append(parent, "delegates");
    dir = std::string(parent);
  }
  std::vector<OwningOpRef<ModuleOp>> deviceModules;
  for (const auto &delegate : *delegates) {
    if (!llvm::any_of(*table, [&](const DelegateEntry &e) { return e.device == delegate->name(); }))
      continue;
    std::string path = dir + "/" + delegate->name() + ".mlir";
    // A SourceMgr parses only its main buffer, so each module gets its own.
    llvm::SourceMgr deviceSources;
    OwningOpRef<ModuleOp> deviceModule = parseSourceFile<ModuleOp>(path, deviceSources, &context);
    if (!deviceModule)
      return fail("cannot parse " + path);
    if (failed(matches(*deviceModule, delegate->name(), *table, path, error)))
      return fail(error);
    auto ids = delegate->compile(*deviceModule, arena, out);
    if (failed(ids))
      return fail("delegate '" + delegate->name() + "' failed to compile " + path);
    for (size_t index = 0; index < ids->size(); ++index) {
      int64_t id = (*ids)[index];
      if (id < 0 || id >= static_cast<int64_t>(table->size()) ||
          (*table)[id].device != delegate->name() || (*table)[id].index != -1)
        return fail("delegate '" + delegate->name() + "' returned an unknown or repeated id");
      (*table)[id].index = static_cast<int64_t>(index);
    }
    deviceModules.push_back(std::move(deviceModule));
  }
  for (const DelegateEntry &e : *table)
    if (e.index < 0)
      return fail("delegate '" + e.device + "' did not compile delegate " + std::to_string(e.id));

  // 5. CPU: every delegate call becomes func.call @__delegate__(id, in, out)
  // on byte buffers (cpu/delegate_calls.cpp); the declarations go.
  if (failed(backend::lowerDelegateCalls(*module)))
    return fail("lowering the delegate calls failed");
  // Every remaining large constant moves into the arena and is read back
  // through a new first argument `ws` (package.cpp).
  if (failed(backend::externalizeWeights(*module, arena)))
    return fail("weight externalization failed");
  // The arena: delegate layouts and dense CPU weights together.
  if (!arena.write(out + "/weights.bin", error))
    return fail(error);
  // The CPU module after step 5, large constants elided.
  if (keepIr) {
    std::error_code ec;
    llvm::raw_fd_ostream os(out + "/cpu.mlir", ec, llvm::sys::fs::OF_Text);
    module->print(os, OpPrintingFlags().elideLargeElementsAttrs(16).elideLargeResourceString(64));
  }

  // 6. CPU code, once per --cpus entry (cpu/toolchain.cpp): bufferize, lower
  // to the LLVM dialect, llc for the target, link model.so. __delegate__
  // stays undefined; the runtime's dispatcher provides it. Only per-target
  // work belongs here: graph-level passes run once, in pynqo-middle-end.
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

  // 7. compile.json: what the runtime needs to call the package. pipeline.py
  // adds argument names and model metadata to it to make manifest.json.
  //   entry_points  the tensor signatures from step 1
  //   delegates     the dispatcher's table: id -> device, index
  //   libraries     model.so per CPU model, picked by the runtime's CPU
  //   npu_limits    checked against the overlay before loading
  std::error_code ec;
  llvm::raw_fd_ostream os(out + "/compile.json", ec, llvm::sys::fs::OF_Text);
  if (ec)
    return fail("cannot write compile.json");
  llvm::json::OStream j(os, 2);
  j.object([&] {
    // 2: a delegate's output is its subgraph's result (the NPU's K chunks are
    // summed in libnpu_rt), not the NPU's partial planes.
    j.attribute("format", "pynqo-backend/2");
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
      for (const DelegateEntry &e : *table)
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
