# Selected TPU-Gen OpenROAD results

These are archived local proof-of-concept runs for the upstream TPU-Gen RTL
library at commit `0abb8511369f859d3286d31caab2c37ab6bddde9`. The flow
used the Nangate45 ASIC platform and a 5 ns clock constraint. The current
default image is `openroad/orfs:26Q3-605-g2d29bdaf8`; the archived state files
do not record the image digest. Values below were
transcribed from each local `iter_00/state.json` and `summary.txt`; the raw
`runs/` directory is not committed.

| Local run ID | M×N | DW/WW | MULT_DW | Retrieved Verilog files | Cell area (µm²) | WNS (ns) | Total power (W) | Wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20261003-demo-validation` | 4×4 | 8/4 | 3 | 21 | 36,300.0 | 2.4452 | 0.0138475 | 427 |
| `20261003-demo-codex-verified` | 4×4 | 8/8 | 4 | 21 | 49,445.4 | 2.26712 | 0.0168515 | 629 |

Both archived state files report `source: openroad`, zero setup violations,
and no flow errors. Their generated headers differ in weight width and
multiplier coefficient, so the two rows are **not** a controlled PPA comparison.
The local run names identify the archived records; the records do not prove an
LLM model version or capture a portable command line for regenerating exactly
the same header. Use `run_flow.py` or `demo.ipynb` with the setup in the parent
README for a new measured run.

The figures are place-and-route results for a Nangate45 ASIC flow. They do not
measure LUT, DSP, BRAM, FPGA timing, or physical PYNQ-Z1 execution. No OpenROAD
run was repeated as part of this documentation PR.
