# Registering a quantization scheme

`src/export/frontend/quantization/` is a general library. It implements
quantization techniques and knows nothing about any accelerator.

## Rules

- Do not mention a device there, in names, comments or imports: no `npu`, no
  vendor path, no backend pass.
- Name a scheme and its granularity for what they do: `W8A8Config`,
  per-channel weights, per-token activations.
- A scheme's parameters (bits, range, granularity, rounding) live in the library,
  in `schemes.py`. A vendor names the schemes its device accepts
  (`spec/quant.py`), and `pipeline.py` compares the names; neither imports the other.
- What a device can run belongs to the vendor and the back end. They match the
  graph the scheme produces; a scheme is never shaped around one device's pass.

## Adding a scheme

1. Define the scheme in `schemes.py`.
2. Write `<scheme>.py` next to `graph.py`: a `prepare_*` per op quantizes that
   op's weight once, and `linear`, `conv2d` and `embedding` replace aten.linear,
   aten.conv2d and aten.embedding, using plain ATen ops only, since the
   re-export traces them into the graph.
3. List the module by the scheme's name in `SCHEMES` in `graph.py`.
4. Test its numerics against an independent reference.

`src/test/tests/test_export_layering.py` enforces the first rule.
