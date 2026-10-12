# Reproduce: Hexagon-style heterogeneous NPU (#136)

Everything below runs from the repository root of branch `npu/issue136-a`.
Results of the original runs: [Report.md](Report.md), [board.md](board.md),
[PPA/](PPA/).

## 1. Tools

| Step | Tool | Version used |
| --- | --- | --- |
| Lint | Verilator | 5.006 |
| Simulation | Icarus Verilog | 11.0 |
| Golden models, generators, unit tests | Python 3 + numpy | 3.11, numpy 1.24 |
| Rough area (inner loop) | Yosys via `ic synth --mode estimate` | 0.23 |
| PPA, overlay | AMD Vivado (self-hosted runner `DESKTOP-54U632L`) | 2026.1 |
| Board | PYNQ-Z1 with PYNQ, at 192.168.2.99 | |

## 2. Functional verification (any Linux host)

```bash
make -C src/test lint                    # verilator -Wall over all RTL
make -C src/test sim PYTHON=python3      # 30 self-checking testbenches
python3 -m unittest discover -s src/test/tests
```

Only the new units:

```bash
make -C src/test sim PYTHON=python3 TESTS="npu_fp16_mul npu_fp16_add npu_fp32_add \
  npu_pe_mp npu_systolic_array_mp npu_vector_unit npu_scalar_core npu_tcm npu_tcm_dma \
  npu_axi4_arbiter npu_cache npu_mem_arbiter npu_axi4_line_bridge npu_cache_hierarchy \
  npu_hx_subsystem"
```

Each testbench prints one `PASS tb_<name> ...` line; the Makefile fails a
test whose log contains `error`, `fail` or `mismatch`.

Regenerate and check the committed vectors (each generator is
deterministic; `--check` compares with the committed file where offered):

```bash
python3 -m src.test.vectors.generate_hx_numeric_vectors
python3 -m src.test.vectors.generate_scalar_vectors --check
python3 -m src.test.vectors.generate_vector_unit_vectors
python3 -m src.test.vectors.generate_hx_subsystem_vectors --check
git status --short src/test/vectors      # no changes expected
```

The board self-test's programs and expected results can be checked on the
host without hardware:

```bash
python3 exp/1011_hexagon_npu/board/hx_selftest.py --host-check
```

## 3. Rough area on the host

```bash
ic synth --mode estimate --top npu_vector_unit \
  --files $(find src/hw/rtl/npu_matrix/npu_hexagon/vector src/hw/rtl/npu_matrix/npu_hexagon/numeric -name '*.sv')
```

Yosys numbers are for the inner loop only; the PPA pages quote Vivado.

## 4. PPA (Vivado, out of context)

On a machine with Vivado (Linux or Windows):

```bash
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/ppa_module.tcl \
  -tclargs --top npu_hx_subsystem --name hx_subsystem_full --clock-port s_axi_aclk
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/ppa_module.tcl \
  -tclargs --top npu_hx_subsystem --name hx_subsystem_no_vector \
  --generic HX_ENABLE_VECTOR:0 --clock-port s_axi_aclk
python3 exp/1011_hexagon_npu/scripts/ppa_table.py build/ppa
```

Each run writes `build/ppa/<name>/{utilization,timing_summary,power}.rpt`
and `summary.json`. The exact configurations of every table are the
`ppa` lists of the exp-board requests named on each PPA page; to rerun one
through the self-hosted runner, put that list in
`exp/1011_hexagon_npu/request.json` with a new `id` and push to
`npu/issue136-a` (the `exp-board` workflow's `ppa` job runs them and
uploads `exp-ppa-<id>`).

## 5. Overlay

```bash
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl -tclargs --hexagon
```

Output: `build/vivado/npu_matrix_16x16_hx/artifacts/npu_matrix.{bit,hwh}`
plus `reports/` (utilization, routed timing, `power_routed.rpt`,
`build_evidence.txt`). Without `--hexagon` the same script builds the
baseline overlay (`npu_matrix_16x16`). The script refuses a dirty tree
unless `--allow-dirty` is given. Bitstreams are never committed.

## 6. Board

Through the self-hosted runners (what produced [board.md](board.md)):

```json
{
  "id": "<new id>",
  "overlay": {"mode": "build", "tclargs": "--hexagon"},
  "export": {"cmd": "bash exp/1011_hexagon_npu/export_smollm.sh build/exp-payload"},
  "board": {"script": "exp/1011_hexagon_npu/board/run_hexagon.sh", "timeout_minutes": 120}
}
```

Write it to `exp/1011_hexagon_npu/request.json` and push to
`npu/issue136-a`. `"overlay": {"mode": "run", "run_id": "<run>"}` reuses
the bitstream of an earlier run instead of rebuilding it. The `board`
artifact holds `results/hx_selftest.json`, `isa_gemm.json`, `smollm.json`
and `board.log`.

By hand on the board (overlay files in `overlay/artifacts/`, SmolLM2
package in `export/smollm`, repository `src/` and `exp/` copied next to
them):

```bash
bash exp/1011_hexagon_npu/board/run_hexagon.sh "$PWD"
```

which runs `hx_selftest.sh` (identity, scalar program vs the ISS, INT8 and
FP16 vector kernels vs `hx_vector`/numpy, the tensor ISA GEMM check) and
then SmolLM2 with the prompt "What is the capital of France?".

## 7. Review loop

Each unit went through one implementation agent and three independent
review-and-improve rounds; what each round found is summarised in
[Report.md](Report.md) and in each unit's PPA page. The rounds are not
scripted: they are repeated by giving a reviewer the unit's files, the
rules in [architecture.md](architecture.md) and the testbench, and asking
for bugs, verification gaps (proved by mutation), timing, area and power.
