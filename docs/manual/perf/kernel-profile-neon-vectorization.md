# CPU kernel profile: NEON vectorization (#202)

Where the board's CPU time goes, before and after the back end started
vectorizing the CPU code for the Cortex-A9's NEON unit
(`src/export/backend/cpu/toolchain.cpp`). Measured on the PYNQ-Z1 on
2026-11-24.

## What changed

Before, the CPU code was scalar. `model.ll` went straight to `llc`, which does
no loop vectorization. `convert-linalg-to-loops` also nests a linalg op's loops
in iterator order, so the hot kernels ended in a strided reduction, and the
work fused into them was redone for every output element:

| Kernel | Innermost loop before | Cost |
| --- | --- | --- |
| attention Q·Kᵀ | head dim, K stored `[head_dim, seq]` | stride-256 reads; RoPE fused into Q recomputed once per key |
| attention P·V | sequence, V stored `[seq, head_dim]` | stride-64 reads |
| INT8 linear | K, B stored `[K, N]` | stride-N reads; SiLU and activation quantization fused into the matmul recomputed once per output column |

Now the back end does three things:

1. **Interchange** (`interchangeForUnitStride`, on linalg before
   bufferization). Each op's innermost loop becomes the dimension with the
   fewest strided operands, which is usually the output's last dimension.
   Named ops are generalized first. Each output still sums its terms in the
   same order, so the result is bit-identical. Work that does not depend on the
   new innermost dimension is hoisted out of it.
2. **opt in two runs**, with the target's CPU and features. The first run
   canonicalizes the loops, hoists invariants and unswitches the fused
   `cf.assert`. The second runs loop-vectorize. Plain `-O3` is not used: it
   takes 37 minutes on the 100k-instruction `prefill` and `decode` functions.
3. **Vectorize hints**. Between the two runs, every innermost loop that does
   not carry a float value gets `llvm.loop.vectorize.enable`. LLVM otherwise
   refuses to vectorize float loops for ARMv7 NEON, because NEON is not
   IEEE 754. Loops that carry a float (float reductions) get no hint: the hint
   would let the vectorizer reorder their sums.

No fast-math flags are set.

## End to end

SmolLM2-135M-Instruct W8A8 is run with `--chat --prompt "What is the capital
of France?" --max-new 32`: a 37-token prompt, three prefill chunks, and greedy
decoding. Packages are exported without `--profile-kernels`.

| Package | Phase | Before | After | Speed-up |
| --- | --- | ---: | ---: | ---: |
| `devices: [cpu, risc_v_npu]` (CVA6 bitstream) | prefill | 47.25 s | 18.37 s | 2.57x |
| | decode | 0.495 tok/s | 0.713 tok/s | 1.44x |
| `devices: [cpu]` | prefill | 588.4 s | 74.8 s | 7.87x |
| | decode | 0.036 tok/s | 0.422 tok/s | 11.7x |

ResNet-18 W8A8 with `devices: [cpu]`, averaged over 10 images: 214.4 s per
image before, 21.8 s after (9.84x).

Both versions write the same text: "The capital of France is Paris." (7
tokens).

## Numerics

On the board, the outputs are **bit-identical** to the scalar code:

- **SmolLM2:** the teacher-forced logits (prefill plus 12 forced decode steps,
  13 x 49152 values) are equal for both the `cpu` and the `risc_v_npu` package.
- **ResNet-18:** the logits of all 10 images are equal.

NEON flushes denormals to zero, but no output changed.

Reordering does change results. A first version hinted every loop, float
reductions included. Its logits differed from the scalar code by a relative
L2 of 0.25 to 2.2 per step, the top-1 token differed at 3 of the 13
teacher-forced steps, and greedy decoding gave a different answer ("Paris,
located in…"). The static W8A8 activation scales amplify a
reordered RMSNorm or softmax sum. `test_cpu_vectorize` guards against this
with an innermost float reduction.

## Where the time goes

Each table lists the top 10 kernel groups before the change. A group is a set
of alike kernels, summed. After-times are matched by kernel id
(`--profile-kernels` packages, `--profile`).

### `risc_v_npu`

The CPU kernels and copies took 43.4 s before and 13.2 s after (3.28x). The
delegates are unchanged at about 14 s.

| Entry | Kernel | Calls | Before ms | After ms | Speed-up |
| --- | --- | ---: | ---: | ---: | ---: |
| prefill | P·V `generic(divf,cmpi,select,mulf,addf) 1x9x16x64xf32` | 90 | 15,634 | 2,652 | 5.9x |
| prefill | Q·Kᵀ + RoPE `generic(mulf,addf,truncf) 1x9x16x256xf32` | 90 | 14,623 | 2,458 | 5.9x |
| decode | P·V `1x9x1x64xf32` | 210 | 2,364 | 481 | 4.9x |
| decode | Q·Kᵀ + RoPE `1x9x1x256xf32` | 210 | 2,279 | 464 | 4.9x |
| decode | KV repeat `generic 1x3x3x256x64xf32` | 420 | 1,302 | 1,078 | 1.2x |
| decode | `generic(truncf,mulf) 1x3x3x64x256xf32` | 210 | 943 | 813 | 1.2x |
| decode | K transpose `linalg.transpose 1x3x3x64x256xf32` | 210 | 840 | 939 | 0.9x |
| decode | KV cache update `generic(index_cast,extract,select) 1x3x256x64xf32` | 420 | 701 | 465 | 1.5x |
| prefill | KV repeat `generic 1x3x3x256x64xf32` | 180 | 621 | 474 | 1.3x |
| prefill | SiLU + quantize `generic(negf,exp,…,fptosi) 1x16x1536xi8` | 90 | 528 | 567 | 0.9x |

### `cpu`

The CPU kernels and copies took 783.8 s before and 90.0 s after (8.71x).

| Entry | Kernel | Calls | Before ms | After ms | Speed-up |
| --- | --- | ---: | ---: | ---: | ---: |
| prefill | down_proj: SiLU + quantize + INT8 matmul `generic(negf,exp,…,muli,addi) 1x16x576xi32` | 90 | 344,544 | 15,872 | 21.7x |
| prefill | gate/up INT8 `linalg.matmul 16x1536xi32` | 180 | 113,541 | 28,050 | 4.0x |
| decode | lm_head: RMSNorm + quantize + INT8 `1x49152xi32` | 7 | 74,203 | 2,939 | 25.2x |
| decode | down_proj `1x1x576xi32` | 210 | 55,031 | 2,224 | 24.7x |
| prefill | o_proj: quantize + INT8 `1x16x576xi32` | 90 | 44,928 | 5,211 | 8.6x |
| decode | gate/up: quantize + INT8 `1x1x1536xi32` | 420 | 39,315 | 3,842 | 10.2x |
| prefill | lm_head (last token) `1x49152xi32` | 3 | 25,254 | 8,863 | 2.8x |
| prefill | q_proj INT8 `linalg.matmul 16x576xi32` | 90 | 17,383 | 5,158 | 3.4x |
| prefill | P·V `1x9x16x64xf32` | 90 | 15,647 | 2,628 | 6.0x |
| prefill | Q·Kᵀ + RoPE `1x9x16x256xf32` | 90 | 14,345 | 2,461 | 5.8x |

### Reading the gains

- **Fused generics gain the most (21x to 25x).** Their activation
  quantization depends only on the row and K, so once N is innermost it leaves
  the loop. That removes N-fold redundant `exp`, `roundeven` and compares.
  Vectorization only adds to that.
- **Plain INT8 matmuls gain 3x to 4x.** The interchange alone makes their
  inner loop a unit-stride i8 x i8 → i32 multiply-accumulate.
- **Attention gains about 6x.** The interchange makes the inner loop
  unit-stride, RoPE is hoisted, and the f32 inner loop is 4-wide NEON.
- **KV-cache copies, repeat and transpose barely change (0.9x to 1.5x).** They
  are memory-bound copies of the whole 256-slot cache at every step, and they
  are now the largest CPU cost of `risc_v_npu` decode.

## Not addressed

- **`risc_v_npu`:** the delegates are now about half of package time. Of
  that, `prepare_weights` is 57% (7.9 s: weights repacked on every call), and
  the `m = 1, n = 49152` lm_head is the top program.
- **`cpu`:** the INT8 multiply-accumulate is widened to i32 lanes
  (`vmla.i32`, 4 lanes). An i16 `vmlal.s16` path could double the lanes.
- **Decode:** materializing the GQA repeat and the K transpose of the whole
  cache at every step is structural. It needs a different lowering, not
  vectorization.

## Cost

`pynqo-backend` takes about 25 s longer per board CPU target: 21 to 29 s
before, 47 to 53 s after, for SmolLM2 with one `cpus` entry.

## Reproduce

Use the configs in `examples/smollm-safetensors/configs/`, with
`cpus: [arm-cortex-a9]`, and `devices: [cpu]` or `devices: [cpu, risc_v_npu]`:

```bash
python -m src.export.cli build <config> --out dist/smollm/package [--profile-kernels]
bash src/deploy/deploy.sh dist/smollm
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/smollm && sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli llm package --overlay overlay/npu_accelerator.bit \
  --chat --prompt "What is the capital of France?" --max-new 32 --json run.json --profile prof'
```

Check for NEON code in the library:

```bash
llvm-objdump-22 -d --mattr=+neon dist/smollm/package/arm-cortex-a9/model.so | grep -cE 'v(mul|mla|add)\.(f32|i32)\s+q'
```

| Package | Before | After |
| --- | ---: | ---: |
| `devices: [cpu]` | 2 | 2,624 |
| `devices: [cpu, risc_v_npu]` | 2 | 2,413 |

The two before are in `memrefCopy`, the C runtime support that clang compiles.
