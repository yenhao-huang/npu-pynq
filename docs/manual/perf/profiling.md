# Profiling a run

Add `--profile DIR` to a task command. It prints a table to stderr and writes
`DIR/profile.json`. Without the flag nothing is measured.

```bash
python -m src.runtime.cli llm <package> --prompt "Hello" --chat --max-new 8 --profile out/
python -m src.runtime.cli classifier --package <package> --image cat.jpg --labels labels.txt --profile out/
```

Add `--profile-trace` to also write `DIR/trace.json`, a timeline you can open
in <https://ui.perfetto.dev>.

## On the board

1. On the host, send an export directory (`package/` plus `overlay/`) with
   the runtime. It lands in `~/jupyter_notebooks/<name>`:

   ```bash
   src/deploy/deploy.sh dist/<name>
   ```

2. On the board, run as root in the PYNQ environment, and pass the overlay:

   ```bash
   cd ~/jupyter_notebooks/<name>
   sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli \
       llm package --overlay overlay/npu_matrix.bit \
       --prompt "Hello" --chat --max-new 8 --profile out/
   ```

The first run builds libnpu_rt on the board and loads the bitstream, so it
is slower. Compare the second run onward.

## Reading the table

Output from a SmolLM2-135M run on the board, shortened:

```text
stage               count   total ms    mean ms     min ms     max ms
prefill                 2 29658.657  14829.328  13340.061  16318.595
decode                  7 14466.144   2066.592   2007.852   2316.192

package calls 44124.801 ms = delegates 9850.080 ms (22.3%) + model.so and dispatch 34274.721 ms

npu: 1899 calls, 1953 runs, 374544 jobs, 48636 instructions, 13552316 polls
per run: 191.8 jobs, 24.9 instructions, 6939.2 polls, 678.0 µs start and poll overhead
phase        total ms   share        MB    GB/s
pack_a        20.239     0.2%      5.85   0.289
stage_b     5499.403    55.8%   1210.32   0.220
pl          3513.086    35.7%      0.00       -
unpack_c     817.352     8.3%     86.18   0.105
pl busy from cycles 2188.609 ms; start and poll overhead 1324.477 ms
```

- **stage**: engine steps. `prefill` and `decode` call the package.
- **package calls**: package-call time, split into the NPU delegate and
  everything else (`model.so` CPU code and dispatch).
- **npu**: delegate calls; `runs`, NPU segment runs (one start each);
  `jobs`, matrix jobs; `instructions`, ISA words the NPU decoded (a `REPEAT`
  counts once); `polls`, status reads.
- **per run**: those counters and the `pl` time the PL was not computing,
  divided by `runs`.
- **phase**: where NPU delegate time goes.
  - `pack_a`: input tiling.
  - `stage_b`: the weight copy into the NPU buffer.
  - `pl`: PL run, start to done.
  - `unpack_c`: output unpacking.
- **pl busy from cycles**: time the PL was actually computing. The rest of
  `pl` is host start and polling.
- **top programs** (below the phases): the slowest NPU matmuls by shape.

Times from `--backend sim` are host times; do not compare them with board
runs.
