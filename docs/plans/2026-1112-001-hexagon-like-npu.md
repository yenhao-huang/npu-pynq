# Hexagon-like NPU: VLIW scalar core, caches, vector core, TCM

| Plan ID | Topic | Revision | Status |
|---|---|---|---|
| 2026-1112-001 | hexagon-like-npu | rev.1 (2026-11-12) | Draft: under discussion; nothing implemented |

Tracking issue: #166, sub-issues #167 (caches), #168 (VLIW scalar core),
#169 (vector core), #170 (TCM). Board:
[Hexagon-like NPU](https://github.com/users/yenhao-huang/projects/4).
Supersedes the approach of #136 / PR #137, which is based on
`npu/issue119-a` and predates the `npu_accelerator` restructure.

Base: `dev` at `d1d8dd5`, design `src/hw/rtl/npu_accelerator/`.

## 1. Order of work

1. Baseline and switch (a separate PR before #168): the `scalar_core`
   wrapper with an implementation parameter, the A/B harness, and recorded
   numbers for today's core (section 2).
2. #168 VLIW scalar core: every other unit hangs off its issue logic and
   instruction format.
3. #167 caches: they sit in front of the new fetch and load/store paths.
4. #170 TCM: the shared scratchpad that the vector and matrix cores use.
5. #169 vector core: dispatched by the scalar core, working on TCM data.

## 2. A/B method (applies to every unit)

Every new unit is compared with the design before it on the same workload.
An A/B has two halves:

| Axis | Metric | Source |
|---|---|---|
| Correctness | final memory image equals `src/test/model/isa_sim.py` | `src/hw/tb/npu_accelerator/tb_scalar_core.sv`, once per implementation |
| Performance | `isa_cycles` per program, cycles per job | existing CSR counters |
| Area | LUT / FF / BRAM / DSP | `ic synth` (Yosys) now, Vivado later |
| Timing | Fmax / WNS | Vivado on the self-hosted host |
| Power | estimated power | Vivado `report_power` |

Rules:

- A and B have identical ports, so nothing outside the swapped module
  changes.
- Selection is a parameter at the module's wrapper (`generate if`), not a
  file-list swap: both implementations always compile and lint, and the
  baseline cannot rot.
- The simulation switch is a Makefile variable, for example
  `make -C src/test sim SCALAR_IMPL=VLIW`; one script runs the same vector set
  on both implementations and prints the table above.
- The baseline (A) numbers are recorded on `dev` before B work starts.
- Workloads include cases where the unit under test is the bottleneck, not
  only cases where the matrix core is.

## 3. VLIW scalar core (#168)

### 3.1 Today's core (A)

`scalar_core/scalar_core.sv` is split into fetch (IF), decode (ID), an EX
issue handshake, the LSU and a controller. Its port list is the contract
with the rest of `npu_accelerator`:

- CSR side: `isa_start`, `prog_addr`, `prog_len`, `data_base`, status and
  the `isa_cycles` / `isa_instructions` / `isa_jobs` counters;
- matrix-core dispatch: `ctl_start`, `ctl_m/n/k`, `ctl_accept`;
- the AXIS operand stream to and from the matrix core;
- one AXI4 master, shared by IF and the LSU.

The ISA is 64-bit words with the opcode in `[63:56]`: `NOP`, `END`, `FENCE`,
`SHAPE`, `ADDR_A/B/C`, `INCR`, `REPEAT`, `GEMM`. There are no scalar
load/store or branch instructions; the core is a command sequencer.

### 3.2 Proposed structure

```
src/hw/rtl/npu_accelerator/scalar_core/
├── scalar_core.sv              thin wrapper: same ports, SCALAR_IMPL selects A or B
├── common/                     shared by A and B
│   ├── scalar_core_lsu.sv
│   ├── scalar_core_controller.sv
│   └── sync_fifo.sv
├── inorder/                    A: today's fetch and decode
│   └── scalar_core_inorder.sv
└── vliw/                       B
    ├── scalar_core_vliw.sv
    ├── vliw_fetch.sv           fetches a packet, not a word
    ├── vliw_decode.sv          N slots
    └── vliw_issue.sv           intra-packet hazards and slot routing
```

```systemverilog
parameter string SCALAR_IMPL = "INORDER";   // or "VLIW"
generate
  if (SCALAR_IMPL == "VLIW") scalar_core_vliw    #(...) u_core (.*);
  else                       scalar_core_inorder #(...) u_core (.*);
endgenerate
```

The VLIW change is confined to fetch, decode and issue; the LSU and
controller are reused so the A/B isolates the issue logic.

### 3.3 Workloads

Today the core mostly sets shape and addresses and issues `GEMM` jobs. When
the matrix core is the bottleneck (large GEMMs), VLIW gains almost no cycles
and costs area. Before B work starts, measure on A the fraction of
`isa_cycles` spent waiting on `ctl_accept`; that bounds the possible gain.
The A/B set must include issue-bound programs: many small GEMMs,
`REPEAT`/`INCR` loops, address-heavy programs, and later programs that drive
the vector core and TCM together with the matrix core.

### 3.4 Open decisions

> Current direction for the scalar core: a standard RISC-V core (CVA6)
> instead of a custom VLIW, see
> [2026-1112-003-cva6-scalar-core.md](2026-1112-003-cva6-scalar-core.md).
> The options below are kept for reference.

1. ISA compatibility.
   - Option 1 (recommended): Hexagon-style packets. A packet-end bit in each
     64-bit word; a one-word packet is exactly an old instruction, so today's
     programs run unchanged on B and the encoder gains a packer. `isa_sim.py`
     learns packet semantics (reads within a packet see pre-packet values),
     a deliberate change to the numeric contract.
   - Option 2: a new encoding. Cleaner, but A and B need different binaries,
     and `isa_sim` and the encoder carry two formats.
2. Slot count: 2 slots (scalar ALU + dispatch) or 4 slots like Hexagon
   (ALU, ALU, load/store, dispatch).
3. Whether the baseline-and-switch step lands as its own PR before #168
   (recommended).

## 4. Caches (#167)

### 4.1 Today's memory traffic

All NPU memory traffic goes through one AXI4 master, shared by two
requesters with a one-bit tag in an in-order FIFO:

- IF (`scalar_core_fetch.sv`) streams `PROG_LEN` instruction words from
  `PROG_ADDR` once, in bursts of up to 16 beats, into a prefetch queue.
  `REPEAT` is expanded in decode, so the program is not re-fetched.
- The LSU (`scalar_core_lsu.sv`) streams each job's A and B tiles from DDR
  into the matrix core's input stream and writes the INT32 results back. This
  is streaming traffic; the only reuse is a tile read by more than one job.

Bursts never exceed 16 beats or cross a 128-byte line. The CPU writes the
program and operands to DDR before `isa_start` and reads results after
`isa_done`.

### 4.2 What this means for the cache design

- L1I has little to cache today: a straight-line program is read once.
  It starts to pay when B has branches and loops, and across repeated runs
  of the same program if it fits.
- L1D has no client today: there are no scalar load/store instructions. It
  needs the load/store slot from #168.
- Streaming tensor tiles through a cache evicts everything else for no
  reuse. Hexagon keeps caches for the scalar side and moves tensor and
  vector data through TCM and DMA.
- BRAM is the shared budget: the XC7Z020 has 140 BRAM36 (about 630 KB),
  already partly used by the matrix core's operand buffers, and the TCM
  (#170) will want most of what is left.
- Coherence with the CPU: the NPU caches must start clean (invalidate at
  `isa_start`) and leave DDR up to date (write back at `END`, and at `FENCE`
  if `FENCE` is defined as a memory barrier).

### 4.3 Proposed design (for discussion)

```
                 ┌──────── scalar_core ────────┐
 IF ──> L1I ──┐  │                             │
              ├──> L2 ──┐                      │
 LD/ST > L1D ─┘         ├──> AXI4 master ──> DDR
 LSU tensor stream ─────┘   (uncached path)
```

- L1I: direct-mapped, read-only, `L1I_BYTES` and `LINE_BYTES` parameters.
- L1D: small set-associative, write-back with write-allocate, or
  write-through to keep coherence simple (open decision).
- L2: unified behind L1I and L1D, set-associative, one AXI read and write
  path to DDR. Inclusive or non-inclusive is open.
- The LSU's tensor stream bypasses the caches (uncached path); reuse of
  tensor tiles is the TCM's job (#170).
- Each cache is its own module with a `CACHE_EN` parameter. With
  `CACHE_EN = 0` it is a pass-through, which is the A/B baseline.
- Per-cache hit, miss and write-back counters, readable through the CSRs,
  so the A/B reports hit rates and not only total cycles.

### 4.4 Open decisions

1. Order: build caches after #168 (L1D needs load/store), or build L1I and
   L2 first against today's fetch and add L1D later.
2. Does the LSU tensor stream go through L2 or bypass it?
3. L1D write policy: write-back + write-allocate, or write-through.
4. Sizes and line size within the BRAM budget left for the TCM; line size
   64 or 128 bytes (128 matches today's burst limit).
5. Coherence: invalidate at start and write back at `END` only, or also
   a `FENCE` / explicit flush instruction.
6. A/B workloads that exercise the caches: looping programs for L1I, scalar
   load/store-heavy programs for L1D.

## 5. TCM (#170)

To be discussed.

## 6. Vector core (#169)

To be discussed.
