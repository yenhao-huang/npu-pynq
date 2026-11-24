# risc_v_npu: the NPU with the CVA6 scalar core

`risc_v_npu` is the device for the NPU overlay built with
`SCALAR_IMPL = "CVA6"`: the scalar core is a CVA6 RISC-V core
(`src/hw/rtl/npu_accelerator/scalar_core/cva6/`) that runs one parameterized
RISC-V GEMM kernel for every delegate and drives the same matrix core through
memory-mapped job registers. Plan: [docs/plans/2026-1112-003-cva6-scalar-core.md](../../../docs/plans/2026-1112-003-cva6-scalar-core.md),
sections 5 and 6.

A package exported for `devices: [cpu, risc_v_npu]` uses this vendor; one for
`[cpu, npu]` uses `npu_pynq`. The two never import each other: the matrix-core
knowledge (`spec/layout.py`, `quant.py`, `board.py`, and on the compiler side
`Match.cpp` / `Claim.cpp` / `Pack.cpp`) is copied, with a note naming the
source file.

## The device as a delegate

The device follows the delegate framework of plan
[2026-1113-001](../../../docs/plans/2026-1113-001-delegate-framework.md),
exactly as `npu` does:

1. **Middle end: the matcher.** `custom_dialect/include/RvNpu/RvNpuMatch.h`
   and `lib/Match.cpp` (the `RvNpuMatch` library, upstream dialects only, no
   link libraries) define `rvnpu::isClaimable`, a copy of `npu::isClaimable`:
   an INT8 x INT8 -> INT32 `linalg.matmul` with static 2-D operands, a
   constant right operand (the weight) and an accumulator that is
   `linalg.fill` of 0 on `tensor.empty`. `pynqo-middle-end --devices
   cpu,risc_v_npu` links it (`AssignDevices.cpp`), cuts each claimed matmul out
   into `delegates/risc_v_npu.mlir` as `@risc_v_npu_<id>` and leaves a bodiless
   `@__delegate_<id>` declaration and its call in the CPU graph.
2. **Back end: the delegate.** `src/export/backend/delegates/risc_v_npu/`
   compiles `delegates/risc_v_npu.mlir` only: per `@risc_v_npu_<id>` in id
   order it checks the subgraph (one activation argument, one claimable
   matmul on it, its result returned), rewrites it with `rvnpu::claim` into
   `rvnpu.matmul {tk}` (M x N INT32 result), packs B and plans the GEMM and
   its segments. The package holds one RV32IMC kernel, the same for every
   delegate and segment; the plan reaches it at run time through the
   parameter block (below), not as code. It contributes no CPU code; the CPU
   graph only calls `__delegate__(id, in, out)`.
3. **Runtime: the output contract.** `in` is the subgraph's M x K INT8
   argument and `out` its M x N INT32 result, so the delegate table has
   `output_bytes = M*N*4`. The NPU still writes one partial plane per K chunk
   (`tk = min(K, MAX_K)`; the RISC-V kernel is unchanged); librv_npu_rt sums
   the planes while unpacking C, wrapping modulo 2^32 exactly as libnpu_rt.

| | `npu_pynq` (A) | `risc_v_npu` (B) |
|---|---|---|
| Scalar core | CUSTOM: replays 64-bit ISA words | CVA6: runs an RV32IMC kernel |
| Back end emits | `programs.bin` (ISA words per task and segment) | `rv_programs.bin` (one flat RISC-V kernel per package) |
| B relocation | the loader patches each ADDR_B word to the stage buffer | none: the kernel adds the offsets of the parameter block to DATA_BASE; B is staged inside the data block |
| Driver | `libnpu_rt` (PYNQ and SIM backends) | `librv_npu_rt` (PYNQ only; no SIM) |
| Platform buffers | io, stage, code | io, code |
| Capability check | ISA_FRONTEND | ISA_FRONTEND and RISCV_SCALAR_CORE (bit 7) |
| Overlay artifacts | `npu_accelerator.{bit,hwh,manifest.json}` | `npu_accelerator_cva6.{bit,hwh,manifest.json}` |

There is no SIM backend: npu_pynq's SIM is a model of the 64-bit ISA and cannot
run RISC-V. A risc_v_npu package runs only on the board with the CVA6 overlay.

## Layout

| Path | Contents |
|---|---|
| `spec/abi.py` | the CSR window (npu_pynq ABI 2.0 plus `Capability.RISCV_SCALAR_CORE = 1 << 7`), the MMIO window (registers, access types, STATUS fields, error and DONE codes), CVA6's address map |
| `spec/layout.py` | tiled A / B / C layout (copied), `DataLayout`: where the parameter block, A, C and the staged B block sit from `data_base`, and the parameter block itself (`PARAM_*`, `param_block`) |
| `spec/quant.py`, `spec/board.py` | copied from npu_pynq |
| `spec/gen_mmio_header.py` | writes `custom_dialect/include/RvNpu/RvNpuMmio.h` from `spec/abi.py` and the parameter block of `spec/layout.py` (`--check` compares) |
| `custom_dialect/` | compile time: `RvNpuMatch` (the claim rule the middle end links), the `rvnpu` dialect, `claim`, `LowerToRiscv`, `crt0.S` / `link.ld` (agent C, #174) |
| `driver/rv_npu_rt/` | `librv_npu_rt`, the ARM-side delegate backend (packs A, writes the parameter block and runs the kernel per segment, sums the K planes of C into the M x N output) |
| `driver/overlay/` | manifest (`build.py`) and verification (`verify.py`) of the CVA6 overlay |

The runtime binds the device in `src/runtime/delegate/__init__.py`
(`BACKENDS`), which `src/runtime/share/loader.py` uses for every device in the
package's delegate table; `src/runtime/delegate/risc_v_npu.py` loads
`librv_npu_rt`.

## The two register windows

**CSR window** (AXI-Lite, ARM side). Unchanged from npu_pynq except that the
CVA6 build reports bit 7 in `CAPABILITIES`. To run a segment the driver writes
`PROG_ADDR` (0x48) = the kernel's DDR address (CVA6's boot address; the same
for every run),
`PROG_LEN` (0x4C) = its size in 64-bit words rounded up, `DATA_BASE` (0x50) =
the data block, then `ISA_CONTROL` (0x40) = 1, and polls `ISA_STATUS` (0x44)
until neither RUNNING nor BUSY is set. DONE without ERROR is success; otherwise
`ISA_ERROR` (0x54) holds the code. `ISA_JOBS` / `ISA_INSTRUCTIONS` count NPU
jobs, `ISA_CYCLES` counts start to done.

**MMIO window** (CVA6 side), base `0x4000_0000`, 4 KiB, uncached and
non-idempotent; 32-bit single-beat accesses only (a burst, an unknown offset or
a write to a read-only register is SLVERR):

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x00 | `SHAPE_MN` | RW | M `[15:0]`, N `[31:16]` |
| 0x04 | `SHAPE_K` | RW | K `[15:0]` |
| 0x08 | `ADDR_A` | RW | A operand address (8-byte aligned) |
| 0x0C | `ADDR_B` | RW | B operand address (8-byte aligned) |
| 0x10 | `ADDR_C` | RW | C result address (8-byte aligned) |
| 0x14 | `DOORBELL` | W | issue one job; the write response waits while the queue is full |
| 0x18 | `STATUS` | R | `[0]` busy, `[1]` error (sticky), `[15:8]` error code, `[23:16]` jobs outstanding |
| 0x1C | `FENCE` | R | as STATUS, answered once every issued job is written back (or a fault is latched) |
| 0x20 | `DONE` | W | `[7:0]` code, 0 = success; ends the program |
| 0x24 | `DATA_BASE` | R | the CSR `data_base` of this run |
| 0x28 | `PROG_LEN` | R | the CSR `prog_len` of this run |

MMIO faults (`spec/abi.py` `MmioError`, the CUSTOM core's codes):
`BAD_SHAPE` 0x11, `MISALIGNED` 0x12, `AXI_READ` 0x13, `AXI_WRITE` 0x14;
matrix-core faults forward `ErrorCode` (1-6, 255). A latched fault drops later
DOORBELLs and ends the program, and wins over the DONE code. `DoneCode.TRAP`
(0x20) is proposed for `crt0.S`'s trap handler; it is not in the RTL. librv_npu_rt
adds 0xfd (timeout), 0xfe (bad call) and 0xff (ended without DONE or ERROR).

CVA6's address map: DDR `0x0000_0000`-`0x1FFF_FFFF` (cached, write-through
D$, executable), the MMIO window, everything else DECERR. The driver refuses
io or code buffers outside the DDR window.

## Package artifacts

The `risc_v_npu` delegate of `pynqo-backend` writes, next to the shared
`weights.bin`:

| File | Read by | Contents |
|---|---|---|
| `rv_programs.bin` | librv_npu_rt | the package's one flat RV32IMC GEMM kernel |
| `rv_programs.txt` | librv_npu_rt | the program table, one record per line (below) |
| `rv_programs.json` | people, tools | the same table as JSON (below); not read by the driver |

This mirrors the npu delegate: `programs.bin` + `programs.txt` (the table
libnpu_rt parses) + `programs.json` (a summary). The driver reads the text
table, as libnpu_rt does, so it needs no JSON parser on the board. The JSON
file carries the same fields.

Until issue #206 the back end emitted one RV32IMC binary per task and segment,
with every bound and offset a constant (format 1). Format 1 is not read any
more: librv_npu_rt refuses it and asks for the model to be exported again.

### rv_programs.txt (format 2)

```
rv-programs 2 <rows> <columns> <max_k> <bin_bytes> <mem_bytes> <count>
task <id> <m> <n> <k> <tm> <tn> <tk> <a_bytes> <c_base> <c_bytes> <b_base> <segments>
seg <ni_lo> <ni_hi> <weight_offset> <b_bytes>
```

- `bin_bytes`: the size of `rv_programs.bin`, the kernel. `PROG_LEN =
  ceil(bin_bytes / 8)`.
- `mem_bytes` (>= `bin_bytes`): what the kernel uses from its load address
  while it runs: the image, `.bss` and the stack that `crt0.S` places after
  it. The driver gives the kernel one slot of `mem_bytes` rounded up to 64 at
  offset 0 of the code buffer, copies the image once at load and zeroes the
  rest; `.data` is not reloaded between runs, so the kernel must not rely on
  initialised writable data surviving a run (crt0 clears `.bss` itself).
- One `task` line per delegate index `0..count-1` in order (the `index` of the
  package's delegate table), each followed by its `segments` `seg` lines.
- `m n k tm tn tk` as `spec/layout.py` `plan_gemm` (and npu's `programs.txt`).
- `a_bytes`, `c_base`, `c_bytes`, `b_base`: the data block, as
  `spec/layout.py` `data_layout(plan, segment_bytes)`: the parameter block at
  offset 0 (`PARAM_BYTES` = 64), A tiles at 64 (`a_bytes`), C planes at
  `c_base = align(64 + a_bytes, 64)` (`c_bytes`, all `ceil(k / tk)` planes,
  which the driver sums into the M x N output), the staged B block at
  `b_base = align(c_base + c_bytes, 64)`. The driver checks
  `c_base >= 64 + a_bytes` and `b_base >= c_base + c_bytes`. The delegate's
  output size (`M*N*4`) is in the package's delegate table, not here.
- A segment is one run of the kernel, as an npu segment is one ISA program:
  column tiles `[ni_lo, ni_hi)`, whose B panels form one contiguous block of
  at most `segment_bytes` (`GemmPlan.segments`). With a large enough
  `segment_bytes` there is one segment, so one run per delegate call.
- `weight_offset`, `b_bytes`: the segment's B block in `weights.bin`, packed as
  `spec/layout.py` `pack_b` (column-tile panels), exactly as npu's segment.
  The driver checks `b_bytes = (ni_hi - ni_lo) * b_panel`.

### rv_programs.json

```json
{
  "format": "rv-programs/2",
  "march": "rv32imc", "mabi": "ilp32",
  "rows": 16, "columns": 16, "max_k": 256,
  "bin_bytes": 312, "mem_bytes": 1336, "param_bytes": 64, "segment_bytes": 65536,
  "tasks": [
    {"id": 0, "m": 1, "n": 64, "k": 64, "tm": 1, "tn": 16, "tk": 64, "jobs": 4,
     "weight_offset": 0, "a_base": 64, "a_bytes": 64, "c_base": 128,
     "c_bytes": 256, "b_base": 384, "b_bytes": 4096,
     "segments": [{"ni_lo": 0, "ni_hi": 4, "jobs": 4,
                   "weight_offset": 0, "b_bytes": 4096}]}
  ]
}
```

(`bin_bytes` and `mem_bytes` above are illustrative.)

### The parameter block

Before every segment run librv_npu_rt writes 16 little-endian int32 words at
`DATA_BASE + 0`, word `i` at byte `4 * i`, then starts the kernel.
`spec/layout.py` is the single source of truth (`PARAM_BYTES`, `PARAM_<NAME>`
word indices, `param_block(plan, layout, ni_lo, ni_hi)`); `gen_mmio_header.py`
emits them into `RvNpuMmio.h` as `RVNPU_PARAM_BYTES` and `RVNPU_PARAM_<NAME>`
(byte offsets), and `rv_npu_rt.h` mirrors them (`kParamBytes`, `ParamWord`)
because the board builds the driver from `driver/` and `spec/` only.
`test_risc_v_npu_abi` keeps all three in step.

| Word | Name | Value |
|---|---|---|
| 0 | `M` | `plan.m` |
| 1 | `N` | `plan.n` |
| 2 | `K` | `plan.k` |
| 3 | `TM` | `plan.tm` |
| 4 | `TN` | `plan.tn` |
| 5 | `TK` | `plan.tk` |
| 6 | `MT` | `ceil(M / TM)` |
| 7 | `KT` | `ceil(K / TK)` |
| 8 | `NT` | `ceil(N / TN)` |
| 9 | `NI_LO` | the segment's first column tile |
| 10 | `NI_HI` | the segment's end column tile (exclusive) |
| 11 | `A_BASE` | 64 (`PARAM_BYTES`, `DataLayout.a_base`) |
| 12 | `C_BASE` | `c_base` |
| 13 | `B_BASE` | `b_base`, the staged window; not biased by `NI_LO` |
| 14 | `B_PANEL` | `plan.b_panel`: bytes of one column tile of B, all K |
| 15 | `C_PLANE` | `plan.c_plane_bytes()`: one K chunk's C |

The kernel reads the block with ordinary loads through CVA6's D$. That is
safe because CVA6 is held in reset between runs, so every run starts with
cold caches and cannot see a stale line of a previous run's block.

### What the kernel does

Contract with `custom_dialect/` (LowerToRiscv, `crt0.S`, `link.ld`):

1. It is position independent and loaded at `PROG_ADDR` (8-byte aligned, in
   DDR); `crt0.S` sets the stack inside `mem_bytes`, clears `.bss`, and jumps
   to the entry. It uses RV32IMC only (`-march=rv32imc -mabi=ilp32`); no B
   or Zcb.
2. It loads `DATA_BASE` (MMIO 0x24) and the parameter block from it, and
   computes every operand address as `DATA_BASE + offset`: A tile `(mi, kc)`
   at `A_BASE + a_offset(mi, kc)`, B tile `(kc, ni)` at
   `B_BASE + (ni - NI_LO) * B_PANEL + kc * align(TK * TN)` (the staged block
   starts at column tile `NI_LO`'s panel), C tile `(kc, mi, ni)` at
   `C_BASE + c_offset(kc, mi, ni)` (`kc * C_PLANE` plus the tile's place in
   the plane).
3. Loops `mi`, then `kc`, then `ni` over `[NI_LO, NI_HI)`; per job it sets
   the job registers (SHAPE_MN, SHAPE_K, ADDR_A, ADDR_B, ADDR_C), then stores
   DOORBELL. The job sequence is the npu ISA program's for the same segment.
4. At the end: load FENCE, then store DONE = 0 (or a nonzero code). A trap
   should store DONE = `DoneCode.TRAP`.

## Building and checking

```bash
python src/vendors/risc_v_npu/spec/gen_mmio_header.py          # after changing spec/abi.py
cmake -S src/vendors/risc_v_npu/driver/rv_npu_rt -B build/rv_npu_rt && cmake --build build/rv_npu_rt
python -m unittest src.test.tests.test_risc_v_npu_abi
```

`test_risc_v_npu_abi` checks `spec/abi.py` against `cva6_npu_mmio.sv`,
`cva6_config.sv` and `csr.sv`, the header against its generator, the driver's
constants against the spec, the layout copy against npu_pynq, and the overlay
manifest, and the parameter block in `spec/layout.py`, `RvNpuMmio.h` and
`rv_npu_rt.h` against each other. On Linux with g++ it also builds
librv_npu_rt and runs it against a fake CSR window (format 1 and inconsistent
tables refused, kernel placement, the parameter block, CSR writes, B staging,
C unpacking with the K planes summed, including wrap-around, error codes).
`test_risc_v_npu_backend` feeds pynqo-backend what the middle end writes for
one GEMM, runs the kernel on an RV32IMC interpreter once per segment with the
parameter block the driver would write, checks that it issues the npu ISA
program's jobs (offsets taken within the A, B and C regions) and reads nothing
of the data block but its parameters, and that the delegate table gives
`output_bytes = M*N*4`.

## Not verified

- No board run: nothing here has started CVA6. The CVA6 overlay is not built
  (`build_overlay.tcl` does not read `cva6.f` yet), so the HWH parameter
  `SCALAR_IMPL` that `verify.py` expects, the artifact names and the
  capability word in the manifest are this vendor's proposal, to be matched by
  the build.
- That CVA6's reset invalidates its caches between runs. The kernel reads the
  parameter block through the D$, and the driver rewrites it before every run;
  a stale line would hand the kernel the previous segment's parameters. (The
  driver rewrites nothing in the code buffer after load, and A / B / C go
  through the DMA, not CVA6's D$.)
- `src/deploy/deploy.sh` copies only `src/vendors/npu_pynq/` to the board; it
  must also copy this vendor's `__init__.py`, `spec/` and `driver/`.
