//===- kernel_probes.h - time every CPU kernel ------------------*- C++ -*-===//
#ifndef BACKEND_CPU_PROFILING_KERNEL_PROBES_H
#define BACKEND_CPU_PROFILING_KERNEL_PROBES_H

#include "mlir/IR/BuiltinOps.h"

#include <string>
#include <vector>

namespace backend {

// One timed region of model.so, as kernels.json lists it.
struct KernelEntry {
  int32_t id;
  std::string kind;      // "kernel", "delegate" or "copy"
  std::string op;        // linalg.generic, tensor.concat, func.call, memref.copy
  std::string function;  // the entry point it runs in
  std::vector<std::string> inputs, outputs;  // operand and result types
  std::vector<std::string> body;             // a linalg op's body operations
  int64_t delegate = -1;                     // the __delegate__ id of a delegate call
  std::string location;                      // empty when unknown
};

// On tensors, before bufferization: wrap every operation with
// pynqo.kernel_id (set by the middle end) in
//
//   func.call @__prof_enter(%id) : (i32) -> ()
//   <the operation>
//   func.call @__prof_exit(%id) : (i32) -> ()
//
// and describe it in `table`. Copies bufferization adds for that operation
// land inside its window.
mlir::LogicalResult probeKernels(mlir::ModuleOp module, std::vector<KernelEntry> &table);

// On memrefs, after bufferization: wrap every memref.copy outside any window
// (copies of function results, of a delegate's input) in a window of its own,
// with a new id after the last in `table`, and describe it there.
void probeCopies(mlir::ModuleOp module, std::vector<KernelEntry> &table);

// kernels.json: {"format": "pynqo-kernels/1", "kernels": [...]}, by id.
bool writeKernelTable(const std::string &path, const std::vector<KernelEntry> &table,
                      std::string &error);

} // namespace backend

#endif // BACKEND_CPU_PROFILING_KERNEL_PROBES_H
