# Official SmolLM2 ONNX INT8 deployment

## Objective and acceptance

Deploy the actual `HuggingFaceTB/SmolLM2-135M-Instruct/onnx/model_int8.onnx` on the
PYNQ-Z1 through `smollm_board_official_quant.ipynb` in the existing board deployment
directory. Preserve the official quantized weights, zero points and scales.
Execute all notebook cells on the physical board and retain their outputs.
Provide the same sequential setup, arithmetic validation, prefill, generation,
reference comparison, repeatability, visual metrics and editable prompt experience.
Real NPU jobs/cycles must distinguish accelerated execution from CPU reference.

## Evidence and layers

- `packages/official-onnx/`: unchanged pinned official ONNX and tokenizer/configs,
  with source URL, commit, size and SHA-256 checked against the official LFS record.
- `packages/official-graph/`: the generated lossless representation derived from that
  graph; conversion must be lossless for the quantized parameters.
- `data/official-oracle/`: host ONNX Runtime CPU trace for the unchanged artifact.
- `packages/official-ort-ops/` and `packages/official-ort/`: native official ORT
  float-operator sessions and the pinned ARM32 runtime, preserving its numerical
  behavior around the exact integer NPU contractions.
- `runs/official-*/`: graph/operator inspection, numeric probes, physical traces,
  executed notebooks, plots and audit evidence.
- `scripts/official_*`: reproducible inspection, conversion and execution helpers.
- `OfficialResult.md`: hypothesis, command, result and decision for each iteration.

## Gates

1. Inspect the pinned official graph, inputs/outputs and quantization/operator
   contracts. Identify the required CPU and NPU operations from actual nodes.
2. Establish the unchanged graph's CPU ONNX Runtime reference for a fixed ChatML
   prompt, including logits, greedy token IDs and KV-cache handling.
3. Implement the necessary adapter to the existing integer ISA. Preserve unsigned
   activation/weight zero-point corrections, original scales and attention/cache
   semantics. Never substitute the earlier AWQ export for official weights.
4. Validate operator results against ONNX Runtime and run representative integer
   jobs on the FPGA under deadlines and bounded CMA windows.
5. Create the named notebook with robust directory discovery. Run all cells from
   the notebook directory without injected deployment path variables; validate
   model load, prefill, 24-token generation, reference comparison and reset/repeat.
6. Inspect all executed cell outputs and numeric evidence, update results, and
   provide the directly usable board URL only after the full notebook passes.

Use exact equality for integer arithmetic. Logit diagnostics start at
`rtol=1e-3, atol=1e-3`; do not weaken them after observing a failure. Explain and
resolve any graph/runtime semantic difference before claiming equivalence.
Measure TTFT, TPOT, load time, memory and physical hardware counters. Keep prior
AWQ deployment and its historical evidence intact.
