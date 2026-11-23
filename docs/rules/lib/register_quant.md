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

Schemes are applied as FX transforms in `fx/transforms/`.

1. Define the scheme in `schemes.py`, with the ops it quantizes in `ops`.
2. Write `fx/transforms/<scheme>.py`, the scheme's whole transform:
   - `select(gm, ops)`: the nodes the op names cover (each op name maps to
     its ATen target, `'linear'` to aten.linear);
   - `calibrated_layers(gm, ops)`, for a scheme with static activation
     scales: the nodes whose input `quantize.calibrate` observes, each with
     the key its statistic goes under (the layer's float weight); the input
     is the node's first argument;
   - `layer_key(gm, node)`: the key of a node's layer, its op name and where
     its float weight's storage lives (taken before any transpose), so
     programs that share a weight share the key;
   - `quantize_layer(gm, node, stats)`: the node's weight quantized, with its
     activation scale from `stats` (a `QuantizedLayer`); `quantize.prepare`
     calls it once per key;
   - `rewrite(gm, node, weights)`: graph surgery only. Look the node's
     tensors up in `weights` by key, register them as buffers, and replace
     the node with a call to the scheme's function for that op;
   - `finish(gm)`: drop dead nodes and the unused float weights, recompile;
   - the functions the re-export traces into the graph, written in ATen ops
     and the custom ops of `fx/custom_ops.py` only.
3. List the module by the scheme's name in `TRANSFORMS` in `quantize.py`.
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
|-- quantize.py           the stage driver: 1. calibrate (run the float programs on the
|                         calibration inputs, record each calibrated layer's largest
|                         |input|) 2. prepare (have the transform quantize every
|                         selected weight once) 3. replace (have the transform rewrite
|                         each node from 2.'s tensors, re-export); owns the statistics
|                         and the quantized weights
`-- fx/
    |-- custom_ops.py     the quant.* custom ops
    `-- transforms/
        `-- <scheme>.py   one scheme's whole transform
```

`schemes.py` stays at the top and names no framework. `quantize.py` is the
stage driver; it runs calibration itself, with `torch.fx.Interpreter` over the
captured programs, so it is written against torch and FX, but it holds no
scheme: which layers are calibrated, how weights are quantized and what each
node becomes live in the scheme's transform, in one subpackage per framework
(`fx/` today). A transform file holds
all of its scheme, graph machinery included; machinery moves into a module of
its own only when a second scheme needs it, and code two frameworks need moves
up only when the second framework actually needs it.

`src/test/tests/test_export_layering.py` enforces the first rule under Rules.
