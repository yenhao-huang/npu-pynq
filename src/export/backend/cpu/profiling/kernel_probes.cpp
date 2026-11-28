//===- kernel_probes.cpp - time every CPU kernel --------------------------===//
//
// --profile-kernels: the middle end numbered every CPU kernel with
// pynqo.kernel_id (middle-end/linalg/transforms/AssignKernelIds.cpp). Here,
// per CPU target, each one is wrapped in calls the runtime defines
// (src/runtime/share/kernel_probes.c):
//
//   void __prof_enter(int32_t id);
//   void __prof_exit(int32_t id);
//
// in two steps, so that every byte model.so moves is inside some window:
//
//   probeKernels  on tensors, before bufferization: each numbered operation.
//                 Bufferization keeps the calls in place, so the allocations
//                 and copies it adds for an operation (a concat's copies, a
//                 copy before an in-place write) are timed with it.
//   probeCopies   on memrefs, after bufferization: each memref.copy still
//                 outside a window gets one of its own (a result copied into
//                 its out-parameter, a delegate's input staged in its buffer).
//
// kernels.json describes every id. Without --profile-kernels none of this
// runs and the attribute is dropped by lowering to loops.
//
//===----------------------------------------------------------------------===//

#include "cpu/profiling/kernel_probes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace mlir;

namespace backend {
namespace {

// Written by the middle end (middle-end/linalg/transforms/Passes.h); the
// spellings must agree (src/test/tests/test_export_layering.py).
constexpr const char *kKernelIdAttr = "pynqo.kernel_id";
constexpr const char *kEnter = "__prof_enter";
constexpr const char *kExit = "__prof_exit";
constexpr const char *kDispatcher = "__delegate__";

std::string text(Type type) {
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}

std::string text(Location loc) {
  if (isa<UnknownLoc>(loc))
    return "";
  std::string s;
  llvm::raw_string_ostream os(s);
  loc.print(os);
  return s;
}

func::FuncOp declare(ModuleOp module, StringRef name) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;
  OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
  auto type = b.getFunctionType({b.getI32Type()}, {});
  auto fn = func::FuncOp::create(b, module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

// enter before `op`, exit after it.
void wrap(Operation *op, int32_t id, func::FuncOp enter, func::FuncOp exit) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Value value = arith::ConstantIntOp::create(b, loc, id, 32);
  func::CallOp::create(b, loc, enter, ValueRange{value});
  b.setInsertionPointAfter(op);
  func::CallOp::create(b, loc, exit, ValueRange{value});
}

KernelEntry describe(Operation *op, int32_t id, std::string kind) {
  KernelEntry e;
  e.id = id;
  e.kind = std::move(kind);
  e.op = op->getName().getStringRef().str();
  if (auto fn = op->getParentOfType<func::FuncOp>())
    e.function = fn.getName().str();
  for (Value v : op->getOperands())
    e.inputs.push_back(text(v.getType()));
  for (Value v : op->getResults())
    e.outputs.push_back(text(v.getType()));
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    for (Region &region : op->getRegions())
      region.walk([&](Operation *inner) {
        if (!isa<linalg::YieldOp>(inner))
          e.body.push_back(inner->getName().getStringRef().str());
      });
  e.location = text(op->getLoc());
  return e;
}

bool isProbe(Operation *op, StringRef name) {
  auto call = dyn_cast<func::CallOp>(op);
  return call && call.getCallee() == name;
}

// Wrap each memref.copy of `block` (and nested blocks) that no window covers.
void scan(Block &block, int depth, int32_t &next, func::FuncOp enter, func::FuncOp exit,
          std::vector<KernelEntry> &table) {
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isProbe(&op, kEnter)) {
      ++depth;
      continue;
    }
    if (isProbe(&op, kExit)) {
      --depth;
      continue;
    }
    if (isa<memref::CopyOp>(op) && depth == 0) {
      table.push_back(describe(&op, next, "copy"));
      wrap(&op, next++, enter, exit);
      continue;
    }
    for (Region &region : op.getRegions())
      for (Block &inner : region)
        scan(inner, depth, next, enter, exit, table);
  }
}

} // namespace

LogicalResult probeKernels(ModuleOp module, std::vector<KernelEntry> &table) {
  std::map<int32_t, Operation *> kernels;
  WalkResult walked = module.walk([&](Operation *op) {
    auto id = op->getAttrOfType<IntegerAttr>(kKernelIdAttr);
    if (!id)
      return WalkResult::advance();
    if (!kernels.emplace(static_cast<int32_t>(id.getInt()), op).second) {
      op->emitError() << "pynqo.kernel_id " << id.getInt() << " is on two operations";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();
  if (kernels.empty())
    return success();
  func::FuncOp enter = declare(module, kEnter), exit = declare(module, kExit);
  for (auto &[id, op] : kernels) {
    KernelEntry e = describe(op, id, isProbe(op, kDispatcher) ? "delegate" : "kernel");
    APInt delegate;
    if (e.kind == "delegate" && matchPattern(op->getOperand(0), m_ConstantInt(&delegate)))
      e.delegate = delegate.getSExtValue();
    table.push_back(std::move(e));
    wrap(op, id, enter, exit);
  }
  return success();
}

void probeCopies(ModuleOp module, std::vector<KernelEntry> &table) {
  int32_t next = 0;
  for (const KernelEntry &e : table)
    next = std::max(next, e.id + 1);
  func::FuncOp enter = declare(module, kEnter), exit = declare(module, kExit);
  for (auto fn : module.getOps<func::FuncOp>())
    for (Block &block : fn.getBody())
      scan(block, 0, next, enter, exit, table);
}

bool writeKernelTable(const std::string &path, const std::vector<KernelEntry> &table,
                      std::string &error) {
  std::vector<const KernelEntry *> sorted;
  for (const KernelEntry &e : table)
    sorted.push_back(&e);
  llvm::sort(sorted, [](const KernelEntry *a, const KernelEntry *b) { return a->id < b->id; });
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot write " + path + ": " + ec.message();
    return false;
  }
  llvm::json::OStream j(os, 1);
  auto strings = [&](const char *key, const std::vector<std::string> &values) {
    j.attributeArray(key, [&] {
      for (const std::string &v : values)
        j.value(v);
    });
  };
  j.object([&] {
    j.attribute("format", "pynqo-kernels/1");
    j.attributeArray("kernels", [&] {
      for (const KernelEntry *e : sorted)
        j.object([&] {
          j.attribute("id", e->id);
          j.attribute("kind", e->kind);
          j.attribute("op", e->op);
          j.attribute("function", e->function);
          strings("inputs", e->inputs);
          strings("outputs", e->outputs);
          if (!e->body.empty())
            strings("body", e->body);
          if (e->delegate >= 0)
            j.attribute("delegate", e->delegate);
          if (!e->location.empty())
            j.attribute("location", e->location);
        });
    });
  });
  os << "\n";
  return true;
}

} // namespace backend
