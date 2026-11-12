//===- RvNpuDialect.cpp - risc_v_npu dialect ------------------------------===//
//
// Copied from src/vendors/npu_pynq/custom_dialect/lib/NpuDialect.cpp, renamed
// to the rvnpu dialect. Keep the verifier in step with npu.matmul's.
//
//===----------------------------------------------------------------------===//
#include "RvNpu/RvNpuDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::rvnpu;

#include "RvNpu/RvNpuOpsDialect.cpp.inc"

void RvNpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RvNpu/RvNpuOps.cpp.inc"
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
  return success();
}

#define GET_OP_CLASSES
#include "RvNpu/RvNpuOps.cpp.inc"
