# SmolLM2 safetensors board validation

Validation date: 2026-10-25 (client clock). Target: PYNQ-Z1, `armv7l`,
Python 3.10.4 in `/usr/local/share/pynq-venv`. The board clock is incorrect;
executed notebook timestamps are preserved rather than rewritten.

## Result

The full 30-layer model has generated 32 tokens on the physical NPU, exactly
matching the independent ARM ISA simulator. The final sealed notebook passed a fresh root-kernel execution: all five code
cells completed without errors, including source checkpoint loading and buffer
release. The NPU reported 4,114 calls, 1,414,944 jobs and zero ISA errors. The entry point is
[smollm_safetensors.ipynb](../../smollm_safetensors.ipynb).

The board deployment is:

```text
/home/xilinx/jupyter_notebooks/smollm-safetensors-20261025/npu_matrix
```

Open its notebook in the root PYNQ kernel. The source notebook is output-free.
The executed notebook and `result.json` are generated deployment artifacts;
the local executed copy is under `build/smollm-safetensors/`.

## Model and source identity

The pinned input is `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Checkpoint SHA-256:

```text
5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c
```

The board reads all 272 tensor headers through the unchanged `SafeTensors`
loader, checks the complete file hash and reads real FP32-converted tensor
values. Host export uses that same checkpoint, all 30 layers, INT8 linear and
embedding weights, FP32 KV cache, online attention, 512 context positions and
16-token prefill chunks. The package has 242 tasks, 254 segments and
135,570,176 weight bytes. Inference streams those weights through CMA.

No `src` file was authored or modified by this task. All 280 files present in
the initial source snapshot remain byte-identical. Six unrelated new source
files appeared concurrently in the worktree; they were neither changed nor
used in this deployment. The checkout already contained a large staged and
unstaged refactor, so HEAD alone is not its source identity. The deployment
seal records the exact copied source bytes. Original source hashes and
compiler logs remain under `build/smollm-safetensors/`.

The existing matched 16 x 16 BIT/HWH pair was retrieved from
`smollm-compiled-stage-20261009`. No new synthesis, RTL edit or overlay rebuild
was needed. [summary.json](data/summary.json) records its hashes.

## Reference and acceptance

The final prompt is `Write a paragraph about a neural processing unit (NPU).`.
The original checkpoint's `tokenizer_config.json` supplies the system template;
the resulting prompt has 42 tokens. The example wrapper includes that asset
without changing the frontend or tokenizer source.

The reference was generated without the FPGA, using the unchanged ARM model
library and C ISA simulator. It was saved before the final physical acceptance
run. [arm-reference.json](data/arm-reference.json) contains its 32 expected IDs.
The default notebook verifies the deployment seal, original checkpoint,
full-layer manifest, physical NPU counters, zero ISA errors and exact ARM
reference tokens. It writes `result.json` and releases the CMA buffers.

The actual physical response is recorded in [board-final.json](data/board-final.json):

> A neural processing unit (NPU) is a type of computer hardware that processes information in a way that mimics the way the human brain processes information. It is

The output stops at the requested 32-token budget. Prompt and generation length
are editable. A custom prompt still runs the physical/correct-source checks;
only the recorded default prompt is checked against the frozen token list.

## Investigation record

| Hypothesis / configuration | Command or entry point | Observation | Decision |
| --- | --- | --- | --- |
| Full safetensors can use the existing AOT pipeline | `convert_model.py stage`, then `compile` | All 30 layers compile for host and ARM; 242 tasks | Keep the standard exporter |
| The initial short prompt is sufficient | Notebook with only `tokenizer.json`, `write a paragraph about npu` | ARM simulator and FPGA both output `npu`; x86 outputs different text | Inspect the model assets and chat template |
| Chunked prefill causes the target difference | `diagnose_arm.py --tokenwise` | Same ARM output | Keep standard chunked prefill |
| Online attention causes it | `convert_model.py stage --attention naive` | Same ARM output with identical weights and programs | Keep original online attention |
| LLVM machine optimization causes it | Recompile the same generated ARM LLVM IR with `llc -O0` | Same ARM output | Keep the original O3 build |
| Individual source operators fail on ARM | `check_cpu_ops.py stage`, source export, then `run --target pynq` | RMSNorm, SiLU, sliced QKV attention, RoPE, full-size SwiGLU, tied checkpoint embedding and output projection pass NumPy comparisons | Preserve source; no operator patch |
| Original chat assets are needed | Include `tokenizer_config.json` and the explicit NPU question | Both ARM simulator and FPGA generate the same complete 32-token explanation | Use original model assets in the final package |
| The final notebook executes completely | `execute_notebook.py` in a fresh root kernel | All five code cells pass, 32 tokens match the ARM reference, zero ISA errors | Retain the executed artifact |

The initial cross-x86 token gate was stricter than the requested board load/run
contract. The current gate compares physical NPU arithmetic against the ARM
ISA reference. Cross-CPU token parity is not established: the x86 explanation
uses slightly different wording. The raw-prompt first-logit comparison is
recorded in [logit-comparison.json](data/logit-comparison.json); its exact
cross-target cause was not isolated. This example does not claim equivalence
to an independent Transformers FP32 model.

## Reproduce

Follow the [Linux](../../README.md), [Windows](../../README.win.md), or
[macOS](../../README.macos.md) guide for model preparation, source-only
compilation, packaging, transfer and the notebook handoff. The standard
frontend requires Python 3.12; the staged graph compiler also runs with
Python 3.11 and NumPy.

For an independent board reference, copy `scripts/diagnose_arm.py` next to the
deployed notebook and run in a board Linux shell:

```bash
cd /home/xilinx/jupyter_notebooks/smollm-safetensors-20261025/npu_matrix
/usr/local/share/pynq-venv/bin/python3 diagnose_arm.py \
  --prompt "Write a paragraph about a neural processing unit (NPU)."
```

It builds a separate manifest pointing to the same ARM library and selects the
simulated ISA backend. It does not load or program the FPGA. Compare its IDs
with the fresh notebook's physical result. Credentials are not part of either
payload; acceptance uses the workspace `.env` password only on sudo stdin.

To reproduce the source operator diagnostics on a Python 3.12 frontend host:

```bash
python examples/smollm-safetensors/scripts/check_cpu_ops.py stage \
  --root examples/smollm-safetensors/model/cpu-ops \
  --checkpoint examples/smollm-safetensors/model/checkpoint
```

On the Linux compiler host, after setting the LLVM environment as in the guide:

```bash
python3 examples/smollm-safetensors/scripts/convert_model.py compile \
  --stage examples/smollm-safetensors/model/cpu-ops --out dist/smollm-cpu-ops
python3 examples/smollm-safetensors/scripts/check_cpu_ops.py run \
  --repo . --root examples/smollm-safetensors/model/cpu-ops \
  --package dist/smollm-cpu-ops --target host
```

Copy the diagnostic package, `inputs.npz`, `expected.npz` and checker to the
board and run the same checker with `--target pynq --repo <deployed source root>`.
This uses the ARM CPU ISA simulator; it does not access physical MMIO. Historical
compact results are retained in `data/`. Generated checkpoints, arrays, build
products, raw logits and executed notebooks remain ignored.
