//===- NpuOps.td - NPU dialect ---------------------------------*- tablegen -*-===//
//
// Two ops are enough to split a model between the Cortex-A9 and the NPU:
// npu.weight names a slice of the weight arena, and npu.matmul is an NPU
// task. Everything else stays in upstream dialects and becomes CPU code.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_OPS
#define NPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Npu_Dialect : Dialect {
  let name = "npu";
  let summary = "PYNQ-Z1 NPU tasks and weights";
  let cppNamespace = "::mlir::npu";
}

class Npu_Op<string mnemonic, list<Trait> traits = []>
    : Op<Npu_Dialect, mnemonic, traits>;

def Npu_WeightOp : Npu_Op<"weight", [Pure]> {
  let summary = "A named, read-only tensor stored in the weight arena";
  let description = [{
    `npu.weight %arena "layers.0.wq"` is the tensor the exporter stores under
    that name. The exporter decides the bytes and their layout after
    partitioning: a weight only an NPU task reads is stored as B tiles; any
    other is stored dense row-major. `--npu-lower` replaces the op with a view
    of the arena at the offset the exporter chose.
  }];
  let arguments = (ins AnyMemRef:$arena, StrAttr:$name);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$arena $name attr-dict `:` type($arena) `->` type($result)";
}

def Npu_MatmulOp : Npu_Op<"matmul", [Pure]> {
  let summary = "INT8 x INT8 -> INT32 matrix product run on the NPU";
  let description = [{
    Computes the K-chunked partial products of `lhs[M, K] @ rhs[K, N]`:
    result plane `c` holds the product over K rows `[c*tk, (c+1)*tk)`.
    `rhs` must come from npu.weight. Summing the planes gives the full
    product; keeping them apart lets the CPU apply a scale per K group,
    which is how group-wise INT4 weights stay exact on the INT8 array.
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
