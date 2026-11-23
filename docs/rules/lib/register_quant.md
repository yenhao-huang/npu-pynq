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

Schemes are applied with FX, in three stages that quantize.py wires
together: the Calibrator (`fx/calibrator.py`), the Quantizer
(`fx/quantizer/base.py`), which runs the scheme's QuantizedLayer
(`fx/quantizer/<scheme>.py`), and the Rewriter (`fx/rewriter.py`). Only the
QuantizedLayer is the scheme's; the rest is shared and knows no scheme.

1. Define the scheme in `schemes.py`, with the ops it quantizes in `ops`
   (names from `OPS` in `fx/utils/share.py`: `linear`, `conv2d`, `embedding`)
   and, in `calibrated_ops`, the subset whose input it quantizes with a
   static scale: the Calibrator observes those, and the Quantizer passes
   their calibrated amax to the layer, so the list is declared once.
   A new ATen op gets an entry there: its target and how to parse its node
   into named arguments (`weight`, and `x` for an op whose input is quantized).
2. Subclass `QuantizedLayer` (`fx/quantizer/base.py`) in
   `fx/quantizer/<scheme>.py` and override the method of each op in `ops`,
   named after the op: `linear(weight, amax)`, `conv2d(weight, amax)`,
   `embedding(weight)` (an op in `calibrated_ops` gets its amax). Each
   quantizes the float weight and returns the layer's `Replacement`
   (`fx/utils/share.py`): the function that replaces its nodes, the tensors
   that become buffers (in order), and how to build the function's
   arguments from the node's parsed arguments and the buffers. Optionally
   override `check(kind, args)` to refuse a node, and set `name` to the
   scheme's name for error messages.

   The base methods raise NotImplementedError, and the Quantizer refuses a
   layer that lacks a method for one of the scheme's ops. It calls `check`
   on every selected node and the op's method once per layer key, and
   returns the plan, layer key -> `Replacement`. The module holds only the
   class: helpers and the replacement functions, which the re-export traces
   into the graph using ATen ops and the custom ops of `fx/custom_ops.py`
   only, are its members, each next to the weight layout it reads
   (`test_export_layering.py` checks there is no module-level function).
3. List the class by the scheme's name in `LAYERS` in `quantize.py`.
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
|-- quantize.py           the stage driver: Calibrator -> stats -> Quantizer(scheme's layer) -> plan -> Rewriter
`-- fx/
    |-- custom_ops.py     the quant.* custom ops
    |-- utils/
    |   `-- share.py      what the stages share: the ATen op table (target, argument parsing),
    |                     select, parse, layer_key, attr, and Replacement (a plan entry)
    |-- calibrator.py     Calibrator: runs the float programs on calibration inputs and
    |                     records each calibrated layer's largest |input|
    |-- rewriter.py       Rewriter: applies a plan to every program (buffers, node
    |                     replacement), re-exports
    `-- quantizer/
        |-- base.py       QuantizedLayer (one method per op, and check) and Quantizer,
        |                 which runs a layer over every program (run -> plan)
        `-- <scheme>.py   one scheme's QuantizedLayer subclass, and nothing else: weight
                          layouts and quantization, activation scales, the functions
                          that replace each op
```

`schemes.py` and `quantize.py` stay at the top: the pipeline only calls
`quantize`. Everything that depends on how a scheme is applied lives in one
subpackage per framework (`fx/` today). Inside it the three stages are
decoupled: the Calibrator, the quantizers and the Rewriter never import each
other, and nothing under `fx/` imports `quantize.py`. They exchange plain
data, the statistics and the plan, through `quantize.py`, and take what they
share from `fx/utils/share.py`. The Calibrator and the Rewriter decide nothing
about numerics; a scheme's QuantizedLayer decides all of its numbers and
does no graph surgery. A second scheme adds a file under `quantizer/`; code two frameworks
need moves up only when the second framework actually needs it.

`src/test/tests/test_export_layering.py` enforces the first rule under Rules and the
stages' decoupling.
