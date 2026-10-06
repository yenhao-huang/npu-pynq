"""A small builder for MLIR text in the func/linalg/tensor/arith dialects.

Operators (src/compiler/ops) describe computations with ``FuncBuilder``
methods; the result is ordinary MLIR that ``mlir-opt`` parses. Values carry
their type string, so ops can derive result types without an MLIR binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

DTYPES = {"f32": "f32", "i8": "i8", "i32": "i32", "i64": "i64", "index": "index", "i1": "i1", "f16": "f16"}


def tensor_type(shape: Sequence[int], dtype: str) -> str:
    dims = "x".join(str(d) if d >= 0 else "?" for d in shape)
    return f"tensor<{dims}x{dtype}>" if shape else f"tensor<{dtype}>"


def memref_type(shape: Sequence[int], dtype: str) -> str:
    dims = "x".join(str(d) if d >= 0 else "?" for d in shape)
    return f"memref<{dims}x{dtype}>" if shape else f"memref<{dtype}>"


@dataclass(frozen=True)
class Value:
    name: str
    type: str

    @property
    def shape(self) -> tuple[int, ...]:
        inner = self.type[self.type.index("<") + 1:-1]
        parts = inner.split("x")
        return tuple(-1 if p == "?" else int(p) for p in parts[:-1])

    @property
    def dtype(self) -> str:
        if "<" not in self.type:
            return self.type
        return self.type[self.type.index("<") + 1:-1].split("x")[-1]

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __str__(self) -> str:
        return self.name


def float_literal(value: float) -> str:
    v = float(value)
    if v != v:
        return "0x7FC00000"
    if v in (float("inf"), float("-inf")):
        return "0x7F800000" if v > 0 else "0xFF800000"
    return f"{v:.9e}"


def _affine(dims: int, exprs: Iterable[str]) -> str:
    names = ", ".join(f"d{i}" for i in range(dims))
    return f"affine_map<({names}) -> ({', '.join(exprs)})>"


def identity_map(rank: int) -> str:
    return _affine(rank, [f"d{i}" for i in range(rank)])


def map_of(dims: int, exprs: Sequence[str]) -> str:
    return _affine(dims, exprs)


class Region:
    """Body of a linalg.generic/reduce: emits scalar ops on block arguments."""

    def __init__(self, func: "FuncBuilder", args: list[Value]) -> None:
        self.func = func
        self.args = args
        self.lines: list[str] = []

    def op(self, text: str, type_: str) -> Value:
        name = self.func.fresh()
        self.lines.append(f"{name} = {text}")
        return Value(name, type_)

    # Scalar helpers ------------------------------------------------------
    def const(self, value: float | int, type_: str) -> Value:
        if type_.startswith("f"):
            return self.op(f"arith.constant {float_literal(value)} : {type_}", type_)
        return self.op(f"arith.constant {int(value)} : {type_}", type_)

    def binary(self, op: str, a: Value, b: Value, fast: bool = True) -> Value:
        flags = " fastmath<fast>" if fast and op.endswith("f") and op.startswith("arith.") else ""
        return self.op(f"{op} {a}, {b}{flags} : {a.type}", a.type)

    def addf(self, a, b): return self.binary("arith.addf", a, b)
    def subf(self, a, b): return self.binary("arith.subf", a, b)
    def mulf(self, a, b): return self.binary("arith.mulf", a, b)
    def divf(self, a, b): return self.binary("arith.divf", a, b)
    def maxf(self, a, b): return self.binary("arith.maximumf", a, b, fast=False)
    def minf(self, a, b): return self.binary("arith.minimumf", a, b, fast=False)
    def addi(self, a, b): return self.binary("arith.addi", a, b)
    def subi(self, a, b): return self.binary("arith.subi", a, b)
    def muli(self, a, b): return self.binary("arith.muli", a, b)

    def unary(self, op: str, a: Value) -> Value:
        flags = " fastmath<fast>" if op.startswith("math.") or op == "arith.negf" else ""
        return self.op(f"{op} {a}{flags} : {a.type}", a.type)

    def exp(self, a): return self.unary("math.exp", a)
    def rsqrt(self, a): return self.unary("math.rsqrt", a)
    def sqrt(self, a): return self.unary("math.sqrt", a)
    def tanh(self, a): return self.unary("math.tanh", a)
    def log(self, a): return self.unary("math.log", a)
    def cos(self, a): return self.unary("math.cos", a)
    def sin(self, a): return self.unary("math.sin", a)
    def absf(self, a): return self.unary("math.absf", a)
    def neg(self, a): return self.unary("arith.negf", a)
    def roundeven(self, a): return self.op(f"math.roundeven {a} : {a.type}", a.type)

    def cast(self, op: str, a: Value, to: str) -> Value:
        return self.op(f"{op} {a} : {a.type} to {to}", to)

    def sitofp(self, a, to="f32"): return self.cast("arith.sitofp", a, to)
    def fptosi(self, a, to="i32"): return self.cast("arith.fptosi", a, to)
    def extsi(self, a, to="i32"): return self.cast("arith.extsi", a, to)
    def trunci(self, a, to="i8"): return self.cast("arith.trunci", a, to)
    def index_cast(self, a, to="index"): return self.cast("arith.index_cast", a, to)

    def cmpf(self, pred: str, a: Value, b: Value) -> Value:
        return self.op(f"arith.cmpf {pred}, {a}, {b} : {a.type}", "i1")

    def cmpi(self, pred: str, a: Value, b: Value) -> Value:
        return self.op(f"arith.cmpi {pred}, {a}, {b} : {a.type}", "i1")

    def select(self, c: Value, a: Value, b: Value) -> Value:
        return self.op(f"arith.select {c}, {a}, {b} : {a.type}", a.type)

    def index(self, dim: int) -> Value:
        return self.op(f"linalg.index {dim} : index", "index")

    def extract(self, tensor: Value, indices: Sequence[Value]) -> Value:
        idx = ", ".join(str(i) for i in indices)
        return self.op(f"tensor.extract {tensor}[{idx}] : {tensor.type}", tensor.dtype)


class FuncBuilder:
    def __init__(self, name: str, args: Sequence[tuple[str, str]], results: Sequence[str] = (),
                 c_interface: bool = True, private: bool = False) -> None:
        self.name = name
        self.args = [Value(f"%{n}", t) for n, t in args]
        self.results = list(results)
        self.c_interface = c_interface
        self.private = private
        self.lines: list[str] = []
        self._counter = 0
        self._consts: dict[tuple[str, str], Value] = {}

    def arg(self, name: str) -> Value:
        for a in self.args:
            if a.name == f"%{name}":
                return a
        raise KeyError(name)

    def fresh(self) -> str:
        self._counter += 1
        return f"%v{self._counter}"

    def emit(self, text: str) -> None:
        self.lines.append(text)

    def op(self, text: str, type_: str) -> Value:
        name = self.fresh()
        self.lines.append(f"{name} = {text}")
        return Value(name, type_)

    def ops(self, text: str, types: Sequence[str]) -> list[Value]:
        base = self.fresh()
        self.lines.append(f"{base}:{len(types)} = {text}")
        return [Value(f"{base}#{i}", t) for i, t in enumerate(types)]

    # Constants -----------------------------------------------------------
    def const(self, value: float | int, type_: str) -> Value:
        key = (repr(value), type_)
        if key not in self._consts:
            if type_.startswith("f"):
                literal = float_literal(value)
            else:
                literal = str(int(value))
            self._consts[key] = self.op(f"arith.constant {literal} : {type_}", type_)
        return self._consts[key]

    def index(self, value: int) -> Value:
        return self.const(value, "index")

    # Tensors -------------------------------------------------------------
    def empty(self, shape: Sequence[int], dtype: str, dynamic: Sequence[Value] = ()) -> Value:
        dyn = ", ".join(str(d) for d in dynamic)
        return self.op(f"tensor.empty({dyn}) : {tensor_type(shape, dtype)}", tensor_type(shape, dtype))

    def fill(self, value: float | int, shape: Sequence[int], dtype: str) -> Value:
        e = self.empty(shape, dtype)
        c = self.const(value, dtype)
        return self.op(f"linalg.fill ins({c} : {dtype}) outs({e} : {e.type}) -> {e.type}", e.type)

    def generic(
        self,
        inputs: Sequence[Value],
        outputs: Sequence[Value],
        maps: Sequence[str],
        iterators: Sequence[str],
        body: Callable[[Region], Sequence[Value]],
        attrs: str = "",
    ) -> list[Value]:
        """linalg.generic; ``body`` receives a Region whose args are the
        scalar block arguments (inputs then outputs) and returns the yields."""
        block_args = [Value(f"%a{i}_{self._counter + 1}", v.dtype) for i, v in enumerate([*inputs, *outputs])]
        region = Region(self, block_args)
        yields = body(region)
        ins = ", ".join(str(v) for v in inputs)
        ins_t = ", ".join(v.type for v in inputs)
        outs = ", ".join(str(v) for v in outputs)
        outs_t = ", ".join(v.type for v in outputs)
        res_types = [v.type for v in outputs]
        head = (f"linalg.generic {{indexing_maps = [{', '.join(maps)}], "
                f"iterator_types = [{', '.join(repr(i).replace(chr(39), chr(34)) for i in iterators)}]"
                f"{', ' + attrs if attrs else ''}}}")
        text = head
        if inputs:
            text += f" ins({ins} : {ins_t})"
        text += f" outs({outs} : {outs_t}) {{\n"
        text += "^bb0(" + ", ".join(f"{a}: {a.type}" for a in block_args) + "):\n"
        text += "".join(f"  {line}\n" for line in region.lines)
        text += f"  linalg.yield {', '.join(str(y) for y in yields)} : {', '.join(y.type for y in yields)}\n}}"
        text += f" -> {', '.join(res_types)}" if len(res_types) == 1 else f" -> ({', '.join(res_types)})"
        if len(outputs) == 1:
            return [self.op(text, res_types[0])]
        return self.ops(text, res_types)

    def elementwise(self, inputs: Sequence[Value], out_dtype: str, body: Callable[[Region], Value],
                    shape: Sequence[int] | None = None, maps: Sequence[str] | None = None) -> Value:
        shape = list(shape if shape is not None else inputs[0].shape)
        rank = len(shape)
        out = self.empty(shape, out_dtype)
        maps = list(maps) if maps else [identity_map(rank)] * len(inputs)
        return self.generic(inputs, [out], [*maps, identity_map(rank)], ["parallel"] * rank,
                            lambda r: [body(r)])[0]

    def extract_slice(self, src: Value, offsets: Sequence[int | Value], sizes: Sequence[int],
                      strides: Sequence[int] | None = None, result_shape: Sequence[int] | None = None) -> Value:
        strides = strides or [1] * len(sizes)
        off = ", ".join(str(o) for o in offsets)
        out_type = tensor_type(result_shape if result_shape is not None else sizes, src.dtype)
        return self.op(
            f"tensor.extract_slice {src}[{off}] [{', '.join(map(str, sizes))}] [{', '.join(map(str, strides))}]"
            f" : {src.type} to {out_type}", out_type)

    def insert_slice(self, src: Value, dst: Value, offsets: Sequence[int | Value], sizes: Sequence[int]) -> Value:
        off = ", ".join(str(o) for o in offsets)
        return self.op(
            f"tensor.insert_slice {src} into {dst}[{off}] [{', '.join(map(str, sizes))}] [{', '.join('1' for _ in sizes)}]"
            f" : {src.type} into {dst.type}", dst.type)

    def reshape(self, src: Value, shape: Sequence[int]) -> Value:
        """Collapse to 1-D then expand: any static reshape of a dense tensor."""
        total = 1
        for d in src.shape:
            total *= d
        flat_t = tensor_type([total], src.dtype)
        if list(src.shape) == list(shape):
            return src
        flat = src
        if src.rank != 1:
            flat = self.op(f"tensor.collapse_shape {src} [[{', '.join(str(i) for i in range(src.rank))}]]"
                           f" : {src.type} into {flat_t}", flat_t)
        if len(shape) == 1:
            return flat
        out_t = tensor_type(shape, src.dtype)
        return self.op(f"tensor.expand_shape {flat} [[{', '.join(str(i) for i in range(len(shape)))}]]"
                       f" output_shape [{', '.join(map(str, shape))}] : {flat_t} into {out_t}", out_t)

    def transpose(self, src: Value, perm: Sequence[int]) -> Value:
        shape = [src.shape[p] for p in perm]
        out = self.empty(shape, src.dtype)
        return self.op(f"linalg.transpose ins({src} : {src.type}) outs({out} : {out.type}) "
                       f"permutation = [{', '.join(map(str, perm))}]", out.type)

    def matmul(self, a: Value, b: Value, acc: Value) -> Value:
        return self.op(f"linalg.matmul ins({a}, {b} : {a.type}, {b.type}) outs({acc} : {acc.type}) -> {acc.type}",
                       acc.type)

    def to_tensor(self, memref: Value) -> Value:
        t = memref.type.replace("memref<", "tensor<")
        return self.op(f"bufferization.to_tensor {memref} restrict : {memref.type} to {t}", t)

    def store(self, tensor: Value, memref: Value) -> None:
        self.emit(f"bufferization.materialize_in_destination {tensor} in restrict writable {memref}"
                  f" : ({tensor.type}, {memref.type}) -> ()")

    def weight(self, arena: Value, name: str, shape: Sequence[int], dtype: str) -> Value:
        t = tensor_type(shape, dtype)
        return self.op(f'npu.weight {arena} "{name}" : {arena.type} -> {t}', t)

    def call(self, callee: str, args: Sequence[Value], results: Sequence[str] = ()) -> list[Value]:
        sig = f"({', '.join(a.type for a in args)}) -> ({', '.join(results)})"
        text = f"func.call @{callee}({', '.join(str(a) for a in args)}) : {sig}"
        if not results:
            self.emit(text)
            return []
        if len(results) == 1:
            return [self.op(text, results[0])]
        return self.ops(text, results)

    def render(self) -> str:
        args = ", ".join(f"{a}: {a.type}" for a in self.args)
        res = f" -> ({', '.join(self.results)})" if self.results else ""
        attrs = " attributes {llvm.emit_c_interface}" if self.c_interface else ""
        vis = "private " if self.private else ""
        body = "".join(f"    {line}\n".replace("\n", "\n    ").rstrip(" ") for line in self.lines)
        return f"  func.func {vis}@{self.name}({args}){res}{attrs} {{\n{body}  }}\n"


@dataclass
class ModuleBuilder:
    funcs: list[FuncBuilder] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)

    def func(self, *args, **kwargs) -> FuncBuilder:
        f = FuncBuilder(*args, **kwargs)
        self.funcs.append(f)
        return f

    def add_raw(self, text: str) -> None:
        """Hand-written functions (e.g. memref-level kernels)."""
        self.raw.append(text)

    def render(self) -> str:
        return "module {\n" + "".join(f.render() for f in self.funcs) + "".join(self.raw) + "}\n"
