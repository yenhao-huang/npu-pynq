//===- NpuOps.td - NPU dialect ---------------------------------*- tablegen -*-===//
//
// npu.matmul is a contraction the NPU runs. --npu-claim rewrites every INT8
// contraction on a constant weight into one; the back end's partition then
// gives each a delegate id and leaves a __delegate__ call in its place, and
// the npu delegate packs its weight (Pack) and encodes its ISA program
// (Encode). Everything else stays in upstream dialects and becomes CPU code.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_OPS
#define NPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Npu_Dialect : Dialect {
  let name = "npu";
  let summary = "PYNQ-Z1 NPU contractions";
  let cppNamespace = "::mlir::npu";
}

class Npu_Op<string mnemonic, list<Trait> traits = []>
    : Op<Npu_Dialect, mnemonic, traits>;

def Npu_MatmulOp : Npu_Op<"matmul", [Pure]> {
  let summary = "INT8 x INT8 -> INT32 matrix product run on the NPU";
  let description = [{
    Computes the K-chunked partial products of `lhs[M, K] @ rhs[K, N]`:
    result plane `c` holds the product over K rows `[c*tk, (c+1)*tk)`.
    `rhs` is a constant weight; the delegate stores it as B tiles. Summing
    the planes gives the full product; keeping them apart lets the CPU apply
    a scale per K group (group-wise INT4 weights).
  }];
  let arguments = (ins RankedTensorOf<[I8]>:$lhs, RankedTensorOf<[I8]>:$rhs,
                       I64Attr:$tk);
  let results = (outs RankedTensorOf<[I32]>:$partials);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($partials)
  }];
  let hasVerifier = 1;
}

#endif // NPU_OPS
