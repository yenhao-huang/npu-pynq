//===- RvNpuOps.td - risc_v_npu dialect ------------------------*- tablegen -*-===//
//
// rvnpu.matmul is a contraction the CVA6 build of the NPU runs. The
// risc_v_npu delegate rewrites the matmul of each device function the middle
// end cut out (rvnpu::claim) into one, packs its weight (Pack) and lowers it
// to an RV32IMC program for CVA6 (LowerToRiscv). --rvnpu-claim applies the
// same rewrite to every claimable matmul of a module. The op never reaches
// CPU code: the CPU graph only calls the delegate.
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
    Computes `lhs[M, K] @ rhs[K, N]` as INT32 (wrapping sum). The NPU runs
    K in chunks of `tk` rows and writes one partial plane per chunk; the
    delegate's runtime (librv_npu_rt) sums the planes, so the result is the
    full M x N product. `rhs` is a constant weight; the delegate stores it as
    B tiles.
  }];
  let arguments = (ins RankedTensorOf<[I8]>:$lhs, RankedTensorOf<[I8]>:$rhs,
                       I64Attr:$tk);
  let results = (outs RankedTensorOf<[I32]>:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // RVNPU_OPS
