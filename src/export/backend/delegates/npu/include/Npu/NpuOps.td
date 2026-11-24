//===- NpuOps.td - NPU dialect ---------------------------------*- tablegen -*-===//
//
// npu.matmul is a contraction the NPU runs. The npu delegate rewrites the
// matmul of each device function the middle end cut out (npu::claim) into
// one, packs its weight (Pack) and encodes its ISA program (Encode).
// --npu-claim applies the same rewrite to every claimable matmul of a module.
// The op never reaches CPU code: the CPU graph only calls the delegate.
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
    Computes `lhs[M, K] @ rhs[K, N]` as INT32 (wrapping sum). The NPU runs
    K in chunks of `tk` rows and writes one partial plane per chunk; the
    delegate's runtime (libnpu_rt) sums the planes, so the result is the
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

#endif // NPU_OPS
