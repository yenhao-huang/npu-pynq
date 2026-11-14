# risc_v_npu: the NPU with the CVA6 scalar core

`risc_v_npu` is the device for the NPU overlay built with
`SCALAR_IMPL = "CVA6"`: the scalar core is a CVA6 RISC-V core
(`src/hw/rtl/npu_accelerator/scalar_core/cva6/`) that runs a RISC-V program
per delegate and drives the same matrix core through memory-mapped job
registers. Plan: [docs/plans/2026-1112-003-cva6-scalar-core.md](../../../docs/plans/2026-1112-003-cva6-scalar-core.md),
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
   `rvnpu.matmul {tk}` (M x N INT32 result), packs B, plans the program and
   lowers each segment to an RV32IMC binary. It contributes no CPU code; the
   CPU graph only calls `__delegate__(id, in, out)`.
3. **Runtime: the output contract.** `in` is the subgraph's M x K INT8
   argument and `out` its M x N INT32 result, so the delegate table has
   `output_bytes = M*N*4`. The NPU still writes one partial plane per K chunk
   (`tk = min(K, MAX_K)`; the RISC-V program is unchanged); librv_npu_rt sums
   the planes while unpacking C, wrapping modulo 2^32 exactly as libnpu_rt.

| | `npu_pynq` (A) | `risc_v_npu` (B) |
|---|---|---|
| Scalar core | CUSTOM: replays 64-bit ISA words | CVA6: runs an RV32IMC program |
| Back end emits | `programs.bin` (ISA words) | `rv_programs.bin` (flat RISC-V binaries) |
| B relocation | the loader patches each ADDR_B word to the stage buffer | none: the program adds constant offsets to DATA_BASE; B is staged inside the data block |
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
| `spec/layout.py` | tiled A / B / C layout (copied), and `DataLayout`: where A, C and the staged B block sit from `data_base` |
| `spec/quant.py`, `spec/board.py` | copied from npu_pynq |
| `spec/gen_mmio_header.py` | writes `custom_dialect/include/RvNpu/RvNpuMmio.h` from `spec/abi.py` (`--check` compares) |
| `custom_dialect/` | compile time: `RvNpuMatch` (the claim rule the middle end links), the `rvnpu` dialect, `claim`, `LowerToRiscv`, `crt0.S` / `link.ld` (agent C, #174) |
| `driver/rv_npu_rt/` | `librv_npu_rt`, the ARM-side delegate backend (packs A, runs the segments, sums the K planes of C into the M x N output) |
| `driver/overlay/` | manifest (`build.py`) and verification (`verify.py`) of the CVA6 overlay |

The runtime binds the device in `src/runtime/delegate/__init__.py`
(`BACKENDS`), which `src/runtime/share/loader.py` uses for every device in the
package's delegate table; `src/runtime/delegate/risc_v_npu.py` loads
`librv_npu_rt`.

## The two register windows

**CSR window** (AXI-Lite, ARM side). Unchanged from npu_pynq except that the
CVA6 build reports bit 7 in `CAPABILITIES`. To run a program the driver writes
`PROG_ADDR` (0x48) = the binary's DDR address (CVA6's boot address),
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
| `rv_programs.bin` | librv_npu_rt | every segment's flat RV32IMC binary, concatenated, each starting 8-byte aligned (zero padding) |
| `rv_programs.txt` | librv_npu_rt | the program table, one record per line (below) |
| `rv_programs.json` | people, tools | the same table as JSON (below); not read by the driver |

This mirrors the npu delegate exactly: `programs.bin` + `programs.txt` (the
table libnpu_rt parses) + `programs.json` (a summary). The coordination brief
asked for a JSON table read by the driver; the driver reads the text table
instead, as libnpu_rt does, so it needs no JSON parser on the board. The JSON
file carries the same fields.

### rv_programs.txt

```
rv-programs 1 <rows> <columns> <max_k> <bin_bytes> <count>
task <id> <m> <n> <k> <tm> <tn> <tk> <a_bytes> <c_base> <c_bytes> <b_base> <segments>
seg <bin_offset> <bin_bytes> <mem_bytes> <weight_offset> <b_bytes>
```

- One `task` line per delegate index `0..count-1` in order (the `index` of the
  package's delegate table), each followed by its `segments` `seg` lines.
- `m n k tm tn tk` as `spec/layout.py` `plan_gemm` (and npu's `programs.txt`).
- `a_bytes`, `c_base`, `c_bytes`, `b_base`: the data block, as
  `spec/layout.py` `data_layout(plan, segment_bytes)`: A tiles at offset 0
  (`a_bytes`), C planes at `c_base = align(a_bytes, 64)` (`c_bytes`, all
  `ceil(k / tk)` planes, which the driver sums into the M x N output), the
  staged B block at `b_base = align(c_base + c_bytes, 64)`. The driver checks
  `c_base >= a_bytes` and `b_base >= c_base + c_bytes`. The delegate's output
  size (`M*N*4`) is in the package's delegate table, not here.
- A segment is one binary and one run, as an npu segment is one ISA program:
  it covers a range of column tiles whose B panels form one contiguous block
  of at most `segment_bytes` (`GemmPlan.segments`). With a large enough
  `segment_bytes` there is one segment, so one binary per delegate.
- `bin_offset`, `bin_bytes`: the binary in `rv_programs.bin` (`bin_offset`
  8-byte aligned, `bin_bytes` > 0). `PROG_LEN = ceil(bin_bytes / 8)`.
- `mem_bytes` (>= `bin_bytes`): what the program uses from its load address
  while it runs: the image, `.bss` and the stack that `crt0.S` places after it.
  The driver gives each binary a slot of `mem_bytes` rounded up to 64 in the
  code buffer, copies the image once at load and zeroes the rest; `.data` is not
  reloaded between runs, so programs must not rely on initialised writable data
  surviving a run (crt0 clears `.bss` itself).
- `weight_offset`, `b_bytes`: the segment's B block in `weights.bin`, packed as
  `spec/layout.py` `pack_b` (column-tile panels), exactly as npu's segment.

### rv_programs.json

```json
{
  "format": "rv-programs/1",
  "march": "rv32imc", "mabi": "ilp32",
  "rows": 16, "columns": 16, "max_k": 256,
  "bin_bytes": 1234, "segment_bytes": 65536,
  "tasks": [
    {"id": 0, "m": 1, "n": 64, "k": 64, "tm": 1, "tn": 16, "tk": 64,
     "a_bytes": 64, "c_base": 64, "c_bytes": 256, "b_base": 320, "jobs": 4,
     "weight_offset": 0,
     "segments": [{"bin_offset": 0, "bin_bytes": 120, "mem_bytes": 1144,
                   "weight_offset": 0, "b_bytes": 4096}]}
  ]
}
```

### What a program does

Contract with `custom_dialect/` (LowerToRiscv, `crt0.S`, `link.ld`):

1. It is position independent and loaded at `PROG_ADDR` (8-byte aligned, in
   DDR); `crt0.S` sets the stack inside the program's `mem_bytes`, clears
   `.bss`, and jumps to the entry. It uses RV32IMC only (`-march=rv32imc
   -mabi=ilp32`); no B or Zcb.
2. It loads `DATA_BASE` (MMIO 0x24) and computes every operand address as
   `DATA_BASE + offset`: A tile `(mi, kc)` at `a_offset(mi, kc)`, B tile
   `(kc, ni)` at `b_base + b_offset(kc, ni) - b_offset(0, ni_lo)` for the
   segment's first column tile `ni_lo` (the staged block starts at that
   tile's panel), C tile `(kc, mi, ni)` at `c_base + c_offset(kc, mi, ni)`.
3. Per job: store SHAPE_MN, SHAPE_K, ADDR_A, ADDR_B, ADDR_C, then DOORBELL.
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
manifest. On Linux with g++ it also builds librv_npu_rt and runs it against a
fake CSR window (placement, CSR writes, B staging, C unpacking with the K
planes summed, including wrap-around, error codes).
`test_risc_v_npu_backend` feeds pynqo-backend what the middle end writes for
one GEMM and checks that the RISC-V binaries issue the npu ISA program's jobs
and that the delegate table gives `output_bytes = M*N*4`.

## Not verified

- No board run: nothing here has started CVA6. The CVA6 overlay is not built
  (`build_overlay.tcl` does not read `cva6.f` yet), so the HWH parameter
  `SCALAR_IMPL` that `verify.py` expects, the artifact names and the
  capability word in the manifest are this vendor's proposal, to be matched by
  the build.
- That CVA6's reset invalidates its caches between runs (the driver rewrites
  nothing in the code buffer after load, and A / B / C go through the DMA, not
  CVA6's D$).
- `src/deploy/deploy.sh` copies only `src/vendors/npu_pynq/` to the board; it
  must also copy this vendor's `__init__.py`, `spec/` and `driver/`.
