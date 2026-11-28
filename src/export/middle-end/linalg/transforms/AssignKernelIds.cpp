//===- AssignKernelIds.cpp - name every CPU kernel ------------------------===//
//
// pynqo-assign-kernel-ids: on the CPU graph, after fusion, every operation
// that becomes its own piece of work at run time gets
//
//   pynqo.kernel_id = <i> : i32
//
// numbered 0.. in program order over the module's functions. A kernel is
//
//   - a linalg operation (after elementwise fusion, one loop nest),
//   - a tensor operation that bufferizes to a copy (concat, insert_slice, pad),
//   - a call to a delegate declaration (@__delegate_<id>).
//
// Views (extract_slice, expand_shape, collapse_shape) and scalar reads are
// not kernels: they move no data. Operations inside a kernel's region are
// part of that kernel.
//
// As ExecuTorch's debug handles and IREE's dispatch regions, kernels are named
// once here, on tensors, so every CPU target shares one numbering. The back
// end only reads the attribute: with --profile-kernels it wraps each kernel in
// timing calls (backend/cpu/profiling/kernel_probes.cpp). Lowering to loops drops the
// attribute, so it changes no generated code.
//
//===----------------------------------------------------------------------===//

#include "linalg/transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {

bool callsDelegate(Operation *op, ModuleOp module) {
  auto call = dyn_cast<func::CallOp>(op);
  if (!call)
    return false;
  auto callee = module.lookupSymbol<func::FuncOp>(call.getCalleeAttr());
  return callee && callee->hasAttr(middle_end::kDelegateIdAttr);
}

bool isKernel(Operation *op, ModuleOp module) {
  return isa<linalg::LinalgOp, tensor::ConcatOp, tensor::InsertSliceOp, tensor::PadOp>(op) ||
         callsDelegate(op, module);
}

struct AssignKernelIdsPass : PassWrapper<AssignKernelIdsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignKernelIdsPass)

  StringRef getArgument() const final { return "pynqo-assign-kernel-ids"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder b(module.getContext());
    int32_t next = 0;
    for (auto fn : module.getOps<func::FuncOp>()) {
      if (fn.isExternal())
        continue;
      fn.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (!isKernel(op, module))
          return WalkResult::advance();
        op->setAttr(middle_end::kKernelIdAttr, b.getI32IntegerAttr(next++));
        return WalkResult::skip();   // its body belongs to it
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> middle_end::createAssignKernelIdsPass() {
  return std::make_unique<AssignKernelIdsPass>();
}
