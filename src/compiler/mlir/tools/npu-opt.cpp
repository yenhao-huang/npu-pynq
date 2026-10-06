//===- npu-opt.cpp - NPU partitioning and lowering driver -----------------===//
//
// mlir-opt with the npu dialect and its two passes. Only the dialects the
// passes read or create are registered; the standard bufferization and
// LLVM lowering run in the upstream mlir-opt.
//
//===----------------------------------------------------------------------===//

#include "Npu/NpuDialect.h"
#include "Npu/NpuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::npu::NpuDialect, mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  mlir::npu::registerPartitionPass();
  mlir::npu::registerLowerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU partitioning driver\n", registry));
}
