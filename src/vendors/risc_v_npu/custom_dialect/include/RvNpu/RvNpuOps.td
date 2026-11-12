//===- RvNpuOps.td - risc_v_npu dialect ------------------------*- tablegen -*-===//
//
// rvnpu.matmul is a contraction the CVA6 build of the NPU runs. --rvnpu-claim
// rewrites every INT8 contraction on a constant weight into one; the back
// end's partition then gives each a delegate id and leaves a __delegate__ call
// in its place, and the risc_v_npu delegate packs its weight (Pack) and lowers
// it to an RV32IMC program for CVA6 (LowerToRiscv). Everything else stays in
// upstream dialects and becomes CPU code.
//
// Same semantics as npu.matmul (src/vendors/npu_pynq/custom_dialect/include/
// Npu/NpuOps.td); a separate dialect so both devices link into one
// pynqo-backend (plan 2026-1112-003 section 6.2).
//
//===----------------------------------------------------------------------===//

#ifndef RVNPU_OPS
#define RVNPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def RvNpu_Dialect : Dialect {
  let name = "rvnpu";
  let summary = "PYNQ-Z1 NPU contractions driven by the CVA6 scalar core";
  let cppNamespace = "::mlir::rvnpu";
}

class RvNpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<RvNpu_Dialect, mnemonic, traits>;

def RvNpu_MatmulOp : RvNpu_Op<"matmul", [Pure]> {
  let summary = "INT8 x INT8 -> INT32 matrix product run on the NPU (CVA6 build)";
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

#endif // RVNPU_OPS
