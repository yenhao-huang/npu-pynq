# Registering a quantization scheme

`src/export/frontend/quantization/` is a general library. It implements
quantization techniques and knows nothing about any accelerator.

## Rules

- Do not mention a device there, in names, comments or imports: no `npu`, no
  vendor path, no backend pass.
- Name a scheme and its granularity for what they do: `W8A8Config`,
  per-channel weights, per-tensor activations.
- A scheme's parameters (bits, range, granularity, rounding) and the ops it
  quantizes (`ops`, framework-neutral names: `linear`, `conv2d`, `embedding`)
  live in the library, in `schemes.py`. A vendor names the schemes its device accepts
  (`spec/quant.py`), and `pipeline.py` compares the names; neither imports the other.
- What a device can run belongs to the vendor and the back end. They match the
  graph the scheme produces; a scheme is never shaped around one device's pass.

## Adding a scheme

Schemes are applied with FX. Two pieces in `fx/` are shared by every scheme
and know none: `calibrator.py` (the Calibrator) and `rewriter.py` (the ATen op
table and the Rewriter). A scheme is one file, `fx/quantizer/<scheme>.py`.

1. Define the scheme in `schemes.py`, with the ops it quantizes in `ops`
   (names from `rewriter.OPS`: `linear`, `conv2d`, `embedding`). A new ATen op
   gets an entry in `rewriter.OPS`: its target and how to parse its node into
   named arguments (`weight`, and `x` for an op whose input is quantized).
2. Write `fx/quantizer/<scheme>.py`, one class holding everything the scheme
   decides:
   - `calibrated_ops(ops)`: the op names whose input it quantizes, which the
     Calibrator observes;
   - `__init__(ops, stats)` and `run(programs)`: quantize the weight of every
     selected node once per `rewriter.layer_key` (op name and float weight
     storage, before any transpose), with its activation scale from `stats`,
     and refuse a node it cannot compute;
   - `tensors(key)`: a layer's tensors, in the order they become buffers;
   - `replacement(kind, args, buffers, weight)`: the function that replaces
     a node of op `kind` and its arguments;
   - those functions, which the re-export traces into the graph, written in
     ATen ops and the custom ops of `fx/custom_ops.py` only, each next to the
     weight layout it reads.
3. List the class by the scheme's name in `QUANTIZERS` in `quantize.py`.
   Calibration inputs reach `quantize(programs, scheme, calibration)` from the
   config's `calibration:` block, which `pipeline.py` loads.
4. Test its numerics against an independent reference.

## Custom ops

`fx/custom_ops.py` defines the quantization steps that stay one node in the graph,
as `torch.library` custom ops in the `quant` namespace (no `aten`, no device
name): `quantize_per_tensor` (float activations to INT8 with one constant
scale) and `dequantize` (an INT32 accumulator times the activation and weight
scales). The contraction between them stays `aten._int_mm`.

- Each op has an eager implementation and a `register_fake`, so
  `torch.export` traces it and `ep.module()` runs it.
- A scheme's parameters (qmax) and calibrated scales are op arguments, never
  constants of the op, so they stay in `schemes.py` and the graph.
- torch-mlir imports a custom op as `torch.operator "torch.quant.<name>"`,
  which it cannot lower. `middle-end/torch/transforms/DecomposeQuantOps.cpp`
  expands each op into the ATen ops of its eager implementation, op for op,
  so compiled results equal PyTorch's arithmetic. Changing an op's arithmetic
  means changing both, and adding an op means adding its expansion there.

## Layout

```text
quantization/
|-- __init__.py           exports quantize
|-- schemes.py            the schemes as data: parameters and the ops each quantizes
|-- quantize.py           the stage driver: Calibrator -> the scheme's quantizer -> Rewriter
`-- fx/
    |-- custom_ops.py     the quant.* custom ops
    |-- calibrator.py     Calibrator: runs the float programs on calibration inputs and
    |                     records each calibrated layer's largest |input|
    |-- rewriter.py       the ATen op table (target, argument parsing), select, layer_key,
    |                     and the Rewriter: buffers, node replacement, re-export
    `-- quantizer/
        `-- <scheme>.py   one scheme: what is calibrated, weight layouts and quantization,
                          activation scales, the functions that replace each op
```

`schemes.py` and `quantize.py` stay at the top: neither names a framework
detail beyond the classes it calls, and the pipeline only calls `quantize`.
Everything that depends on how a scheme is applied lives in one subpackage
per framework (`fx/` today). Inside it, the Calibrator and the Rewriter are
shared machinery and decide nothing about numerics; a quantizer decides all
of its scheme and does no graph surgery. A second scheme adds a file under
`quantizer/` and reuses both; code two frameworks need moves up only when the
second framework actually needs it.

`src/test/tests/test_export_layering.py` enforces the first rule under Rules.
