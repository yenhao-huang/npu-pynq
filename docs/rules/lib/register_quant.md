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
   - `observe(gm, ops, inputs, stats)`, for a scheme with static activation
     scales: run calibration inputs through the float program and record
     what the scales need, keyed like the weight cache;
   - `rewrite(gm, node, cache, stats)`: quantize the node's weight once (keyed
     in `cache` by the float weight's storage, so programs share it), fix its
     activation scale from `stats`, register the results as buffers, and
     replace the node with a call to the scheme's function for that op;
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
|-- quantize.py           the stage driver: calibrate, select each scheme op's nodes,
|                         have the scheme's transform rewrite them, re-export; owns the
|                         statistics and weight cache shared by every program
`-- fx/
    |-- custom_ops.py     the quant.* custom ops
    `-- transforms/
        `-- <scheme>.py   one scheme's whole transform
```

`schemes.py` and `quantize.py` stay at the top: they name no framework, so the
pipeline never does either. Everything that depends on how a scheme is applied
lives in one subpackage per framework (`fx/` today). A transform file holds
all of its scheme, graph machinery included; machinery moves into a module of
its own only when a second scheme needs it, and code two frameworks need moves
up only when the second framework actually needs it.

`src/test/tests/test_export_layering.py` enforces the first rule under Rules.
