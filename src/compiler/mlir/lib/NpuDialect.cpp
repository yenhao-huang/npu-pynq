//===- NpuDialect.cpp - NPU dialect ---------------------------------------===//
#include "Npu/NpuDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::npu;

#include "Npu/NpuOpsDialect.cpp.inc"

void NpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Npu/NpuOps.cpp.inc"
      >();
}

LogicalResult MatmulOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto out = getPartials().getType();
  if (!lhs.hasStaticShape() || !rhs.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("requires static shapes");
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || out.getRank() != 3)
    return emitOpError("expects lhs MxK, rhs KxN and result KtxMxN");
  int64_t m = lhs.getDimSize(0), k = lhs.getDimSize(1), n = rhs.getDimSize(1);
  if (rhs.getDimSize(0) != k)
    return emitOpError("inner dimensions differ");
  int64_t tk = getTk();
  if (tk <= 0)
    return emitOpError("tk must be positive");
  int64_t kt = (k + tk - 1) / tk;
  if (out.getDimSize(0) != kt || out.getDimSize(1) != m ||
      out.getDimSize(2) != n)
    return emitOpError("result must be ") << kt << "x" << m << "x" << n;
  if (!getRhs().getDefiningOp<WeightOp>())
    return emitOpError("rhs must be an npu.weight");
  return success();
}

#define GET_OP_CLASSES
#include "Npu/NpuOps.cpp.inc"
