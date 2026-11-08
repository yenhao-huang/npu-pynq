# Official ONNX INT8 board results

Status: **PASSED**. All eight code cells executed on the physical PYNQ-Z1,
with saved outputs and zero errors. The independent final audit passed.

Open the verified output-bearing notebook:
[smollm_board_official_quant.ipynb](http://192.168.2.99:9090/notebooks/smollm_validation_20261009/exp/1007_sw_stack/deploy/smollm_board_official_quant.ipynb).
Run ID: `official-board-20261009-02`.

The unchanged official ONNX artifact was downloaded from HuggingFaceTB at commit
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, 137,147,867 bytes, SHA-256
`a7c33f9ef85d06734cc9d1f943f7e2ba57c77e769df727f4d3e217d9f672b0cc`.
Size and SHA match the official LFS metadata. Source provenance and tokenizer
files are stored in `packages/official-onnx/source.json`.

| Gate | Status | Evidence |
| --- | --- | --- |
| Official artifact identity | Passed | Pinned source metadata and SHA-256 |
| Graph and quantization inspection | Passed | 1,212 nodes; 210 integer projections; 30 layers |
| Unchanged ONNX Runtime reference | Passed on host | Native ORT floating operators; 24/24 identical tokens; KV exact; logits below 1e-3 |
| Adapter operator equivalence | Passed on host | 1,571 outputs checked on identical operator inputs |
| Physical NPU arithmetic and 24-token inference | Passed | Run 02: three exact probes; all 24 tokens and 24 x 49,152 logits match unchanged ARM ONNX; max error 0.00003194809; 842,400 physical jobs |
| ARM CPU versus physical NPU | Passed | All 24 token IDs and all logits exactly equal; maximum error zero |
| Reset/repeat | Passed | All 24 token IDs and every logit exactly equal; maximum error zero |
| Named notebook, every cell executed | Passed | Eight execution counts 1 through 8, zero errors, four saved and visually inspected figures |
| Interactive prompt | Passed | 32 generated tokens; full stream and timing retained |
| Final audit | Passed | All 25 gates true; named notebook SHA equals the executed file; live Jupyter root maps the requested URL |

Plan: [OfficialPlan.md](OfficialPlan.md).

## Iteration record

| Hypothesis / implementation | Command | Observed result | Decision |
| --- | --- | --- | --- |
| A lossless initializer map can retain the official artifact | `official_export.py` | 698 unchanged tensors, 136,744,704 tensor bytes | Keep pinned ONNX and tensor hashes |
| NumPy float operators can reproduce the full official graph | `official_reference.py` with NumPy operators | Failed: tiny float differences cross dynamic-quantization rounding boundaries; full trace diverges | Reject this execution path for acceptance |
| Every operator has the correct semantics | `official_operator_check.py` | 1,212 nodes and 1,571 outputs: integer exact; float tolerances 1e-5, final FP32 contraction 1e-4 | Keep original graph and parameters |
| Native ORT float kernels resolve quantization boundary drift | `official_reference.py` with `OrtOperators` | Both ORT 1.31.0 and pinned 1.20.1: all 24 tokens match unchanged full ONNX; KV cache exact; logits about 1e-5 | Use official ORT 1.20.1 on ARM32 |
| Signed NPU integer products plus original zero-point corrections are feasible | Physical `official_probe.py` | Three integer shapes exact; full official prefill outputs `The`, 64,800 model jobs | Integrate native ORT float kernels before notebook acceptance |

The NumPy-only path is diagnostic and is **not** the delivered acceptance path.
The early board probe above does not prove completion of the notebook. The
official ARM32 runtime has now built successfully and loaded on the physical
board. The complete notebook acceptance and final artifact audit passed.

Additional verified progress:

- Restored the official default system message from `tokenizer_config.json`.
  The former user-only ChatML path reproduced an immediate EOS for the user's
  `write a paragraph about npu` prompt. With the official template, generation
  proceeds; the notebook example spells out "neural processing unit (NPU)".
- `official_tokenizer_check.py` compared the exact Jinja template and BPE IDs
  against Hugging Face `tokenizers` on five cases, including explicit system
  turns and Chinese text: all passed.
- Re-ran the unchanged full ONNX oracle with the official default system turn:
  24/24 tokens match, KV error is zero, maximum logit error is `3.814697265625e-5`.
- The reduced CMA cache-maintenance ranges passed all three physical integer
  probes again. The subsequent native ORT integration probe also passed.
- ARM build initially stopped because `patch` was absent; installed it and
  resumed from the retained cache. A subsequent Eigen download failed its
  pinned SHA-1 (GitLab regenerated that commit archive); the container exited
  and interrupted the source-copy process. No complete copy is assumed.
  Existing `/work/source`, `/work/build` and partial `/opt/ort-source` artifacts
  are retained. The resumed build now clones the same ORT commit directly to
  `/opt/ort-linux-source` and builds in `/opt/ort-build`, avoiding measured
  Windows bind-mount file-tree overhead. It fetches exactly Eigen commit
  `e7248b26a1ed53fa030c5c459f7ea095dfd276ac` from the GitHub mirror adopted in
  [Microsoft's current dependency manifest](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt).
  No numeric-kernel or dependency-version substitution is made.

## Physical board acceptance

- The pinned ORT 1.20.1 cross-build completed with exit code zero. Its shared
  library and C bridge loaded on the board; dynamic dependencies resolved.
- `official_probe.py --native-ort` passed three exact integer probes and the
  full official 40-token prefill. Prefill took 49.381560209 seconds and the
  model executed 210 integer projections / 97,200 physical jobs. Its first
  generated token was `The`.
- `run_arm_ort_oracle.sh` executed the unchanged full ONNX for 24 greedy tokens
  under Cortex-A9 ARMv7 QEMU using the same ORT build. Evidence is in
  `data/official-arm-oracle/{summary.json,reference.npz}`. ARM and x86 greedy
  traces differ because native float rounding can cross dynamic-quantization
  boundaries; physical equivalence is therefore measured against ARM.
- The physical notebook run `official-board-20261009-01` completed five code
  cells, including generation. All 24 physical generation tokens exactly match
  the unchanged ARM ONNX oracle; maximum logit error is 0.00003194809 and all
  logits pass `rtol=1e-3, atol=1e-3`. TTFT was 45.3555 seconds, TPOT was
  21.4638 seconds/token, and 5,040 integer calls executed 842,400 physical jobs.
- The sixth cell failed constructing its second model: 698 separate tensor
  mappings per model exhausted the process file-descriptor limit. The failed
  run and its outputs are retained. `OfficialGraph` now uses one read-only
  whole-file mapping with tensor views at the original offsets. No tensor,
  arithmetic or quantization parameters changed. The two-engine mapping check
  passed on the physical board: 698 tensors per engine match byte-for-byte and
  are read-only; only two extra descriptors are used (`fd_before=4`,
  `fd_after=6`). Evidence: `runs/official-mapping-check/result.json`.
- The old runner and kernel were confirmed terminal before starting full run
  `official-board-20261009-02`. All eight code cells passed; its runner exited
  zero and its kernel shut down normally. All 24 generated IDs match the full
  unchanged ARM oracle; all 24 x 49,152 logits pass the fixed tolerance, with
  maximum absolute error 0.00003194809. TTFT is 41.2222 seconds and TPOT is
  17.1520 seconds/token. All 5,040 integer projections execute 842,400 physical
  jobs / 460,911,894 cycles. The independent exact-integer ARM CPU graph has
  completed all 24 tokens: IDs and every logit are exactly equal to physical
  execution (maximum error zero). Its shared kernel had only 79 open
  descriptors after loading both models. Reset/repeat completed with all 24
  token IDs and every logit exactly equal (maximum error zero). All notebook
  verdict fields are true, including every integer call and planned job
  running on the physical NPU. The editable prompt produced 32 tokens.
- `audit_official_notebook.py --run-id official-board-20261009-02` passed all
  25 gates, including original ONNX and lossless tensor hashes, identical
  ARM runtime binary, the full ARM oracle, exact CPU comparison, reset/repeat,
  all eight execution counts, zero errors, four figures and interactive output.
  Evidence: `runs/official-board-20261009-02/final-audit.json` and
  `smollm_board_official_quant.executed.ipynb`. The verified named notebook
  has SHA-256
  `6b1ab8ef3cb6f065f68018e69645fe1ff2ed425d3b46f967fc8a80e1c0728753`.
  The live Jupyter listener on port 9090 has the expected notebook root and
  base URL `/`. The named file is byte-identical to the executed notebook.

## Observed model quality

The editable example asks for a paragraph about a neural processing unit (NPU),
but the generated text begins `NLP is a subfield of artificial intelligence`.
An additional unchanged full ARM ONNX run with that exact official chat prompt
produced the same first 24 token IDs as the physical 32-token response.
This topic substitution is also present in the official ARM reference.
Evidence: `data/official-arm-oracle/interactive-check.json`.
Numerical and execution acceptance should be distinguished from answer quality.
The original model and its quantization parameters remain unchanged.
