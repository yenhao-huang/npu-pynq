//===- interchange.cpp - unit-stride innermost loops ----------------------===//
//
// convert-linalg-to-loops nests a linalg op's loops in iterator order, so its
// innermost loop is its last dimension. This pass reorders the iterators so
// that the innermost one is the dimension the vectorizer can use. Tiling, when
// it comes, belongs in this folder too: it is per-target loop structure on
// linalg, before bufferization.
//
//===----------------------------------------------------------------------===//

#include "cpu/lowering/interchange.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace backend {
namespace {

// How many operands of `op` a loop over `dim` reads or writes with a stride
// other than one: those whose indexing map uses `dim` anywhere but as its
// whole last result. An operand that does not use `dim` is invariant in it.
unsigned stridedOperands(linalg::LinalgOp op, unsigned dim) {
  unsigned strided = 0;
  for (AffineMap map : op.getIndexingMapsArray()) {
    unsigned n = map.getNumResults();
    bool used = false, unitStride = false;
    for (unsigned r = 0; r < n; ++r) {
      AffineExpr e = map.getResult(r);
      if (!e.isFunctionOfDim(dim))
        continue;
      used = true;
      auto d = dyn_cast<AffineDimExpr>(e);
      unitStride = r == n - 1 && d && d.getPosition() == dim;
      if (!unitStride)
        break;
    }
    if (used && !unitStride)
      ++strided;
  }
  return strided;
}

// The dimension to make innermost: the one with the fewest strided operands,
// the current innermost (the last) on a tie. Too short a loop to fill a
// vector is not worth moving inward.
unsigned bestInnermost(linalg::LinalgOp op) {
  unsigned loops = op.getNumLoops();
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  unsigned best = loops - 1, fewest = stridedOperands(op, best);
  for (unsigned d = 0; d + 1 < loops; ++d) {
    if (!ShapedType::isDynamic(ranges[d]) && ranges[d] < 4)
      continue;
    unsigned strided = stridedOperands(op, d);
    if (strided < fewest) {
      best = d;
      fewest = strided;
    }
  }
  return best;
}

} // namespace

// Make a linalg op's innermost loop the dimension with the fewest strided
// operands (bestInnermost), so that LLVM's loop vectorizer sees unit-stride
// or invariant accesses, and work that does not depend on it leaves the loop.
// Named ops (linalg.matmul) are generalized first.
//
//   attention     Q·Kᵀ reduces over the head dimension with K stored
//                 [head_dim, seq], P·V over the sequence with V stored
//                 [seq, head_dim]: both read with a stride; with the output's
//                 last dimension innermost every access is unit-stride or
//                 invariant, and the RoPE fused into Q·Kᵀ is computed once
//                 per element of Q instead of once per key.
//   INT8 linear   matmuls reduce over K with B stored [K, N]; the activation
//                 quantization fused into them (and the SiLU before it)
//                 becomes invariant in N.
//
// The other loops keep their order, so every output element still sums its
// terms in the same order: the result is bit-identical.
void interchangeForUnitStride(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  SmallVector<linalg::LinalgOp> ops;
  module.walk([&](linalg::LinalgOp op) {
    if (op.getNumLoops() >= 2 && op.hasPureTensorSemantics() &&
        bestInnermost(op) != op.getNumLoops() - 1)
      ops.push_back(op);
  });
  for (linalg::LinalgOp op : ops) {
    unsigned loops = op.getNumLoops(), best = bestInnermost(op);
    rewriter.setInsertionPoint(op);
    auto generic = dyn_cast<linalg::GenericOp>(op.getOperation());
    if (!generic) {
      DictionaryAttr attrs = op->getDiscardableAttrDictionary();
      FailureOr<linalg::GenericOp> g = linalg::generalizeNamedOp(rewriter, op);
      if (failed(g))
        continue;
      generic = *g;
      generic->setDiscardableAttrs(attrs); // pynqo.kernel_id
    }
    SmallVector<unsigned> order;
    for (unsigned d = 0; d < loops; ++d)
      if (d != best)
        order.push_back(d);
    order.push_back(best);
    (void)linalg::interchangeGenericOp(rewriter, generic, order);
  }
}

} // namespace backend
