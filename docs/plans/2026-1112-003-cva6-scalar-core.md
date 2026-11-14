# CVA6 RISC-V core as an alternative scalar core

| Plan ID | Topic | Revision | Status |
|---|---|---|---|
| 2026-1112-003 | cva6-scalar-core | rev.6 (2026-11-14) | Implemented on branch `npu/issue174-a` (#174, uncommitted): hardware simulated (7/7 programs match `isa_sim.py`), `risc_v_npu` vendor, driver and back-end delegate built and unit-tested; A/B in `exp/scalar_core_compare/`; no synthesis, no board run |

Tracking issue: #166 (scalar-core track). This plan replaces the direction of
#168 ("make the scalar core VLIW", plan
[2026-1112-001](2026-1112-001-hexagon-like-npu.md) section 3) with a standard
RISC-V core: [CVA6](https://github.com/openhwgroup/cva6). How issues map to
this work is an open decision (section 8).

Base: `dev`, design `src/hw/rtl/npu_accelerator/`.

## 1. Decisions already made

- CVA6 is integrated first. PicoRV32 is deferred; it is a later candidate only.
- Implementations are independent siblings under
  `src/hw/rtl/npu_accelerator/scalar_core/`: `custom/` (today's instruction
  sequencer) and `cva6/`. There is no shared `common/` folder; each
  implementation is self-contained.
- `custom/` holds today's files, moved, with every module name prefixed
  `custom_`. SystemVerilog module names are global and all implementations
  compile together, so names must not collide.
- `scalar_core.sv` stays a thin wrapper with the unchanged port list and a
  `SCALAR_IMPL` parameter (`"CUSTOM"` | `"CVA6"`) selecting via `generate if`.
- No custom RISC-V instructions. CVA6 runs standard RISC-V only and controls
  the matrix core through memory-mapped registers (MMIO).
- CVA6 runs a complete RISC-V program compiled per model, not a fixed
  firmware walking a job list. The back end lowers the NPU contractions to
  RISC-V code (MLIR → LLVM IR → RV32IMC → ELF); section 6. A fixed schedule
  is what A already does, so a CVA6 that only replays a list would add area
  without adding capability.
- The software of B is a separate device, fully apart from A: vendor
  `src/vendors/risc_v_npu/` and delegate
  `src/export/backend/delegates/risc_v_npu/`; `npu_pynq` and
  `delegates/npu` stay unchanged (section 6.2).
- CVA6 upstream is a git submodule at `scalar_core/cva6/upstream/`, pinned to
  tag `v5.3.0` and never modified. Customization goes only through CVA6's own
  configuration parameters. If a source change is ever unavoidable, it lives
  in `cva6/patches/`, applied at build time and documented in the README.

## 2. Goal and A/B framing

| Variant | Implementation | Config |
|---|---|---|
| A | `CUSTOM`: today's 64-bit ISA sequencer | — |
| B | `CVA6`, single issue | `cv32a60x`-derived |
| B' (later) | `CVA6`, dual issue | `cv32a65x`-derived (`SuperscalarEn=1`) |

Same ports, same workloads. The axes follow plan 001
[section 2](2026-1112-001-hexagon-like-npu.md#2-ab-method-applies-to-every-unit):
final memory image equals `src/test/model/isa_sim.py` (correctness),
`isa_cycles` per program (performance), LUT / FF / BRAM / DSP (area),
Fmax / WNS (timing), and Vivado `report_power` (power). B' against B isolates
single- versus dual-issue.

## 3. Today's core (A)

`scalar_core/` holds `scalar_core.sv`, `scalar_core_fetch.sv`,
`scalar_core_decode.sv`, `scalar_core_controller.sv`, `scalar_core_lsu.sv`
and `sync_fifo.sv`. `scalar_core.sv` itself holds only the EX issue handshake
and the arbitration of the AXI read channel between IF and the LSU. Its header
requires that `ctl_start` and the job shape come straight from flip-flops,
because the 100 MHz margin of the 16 x 16 array leaves no room for logic in
front of the matrix controller. Any B must keep that property.

Port list (the contract, unchanged by this plan):

| Group | Signals |
|---|---|
| CSR side | `isa_start`, `prog_addr`, `prog_len`, `data_base`; `isa_running`, `isa_busy`, `isa_done`, `isa_error`, `isa_error_code`, `isa_pc`, `isa_instructions`, `isa_jobs`, `isa_cycles` |
| Matrix dispatch | `ctl_start`, `ctl_m/n/k`, `ctl_accept`, `ctl_error`, `ctl_error_code` |
| Operand stream | AXIS out (`s_axis_*`, 64-bit) and in (`m_axis_*`, 32-bit) |
| Memory | one AXI4 master: 32-bit address, 64-bit data, no ID signals |

## 4. Hardware layout

### 4.1 Lean layout (chosen)

```
src/hw/rtl/npu_accelerator/scalar_core/
├── scalar_core.sv              wrapper, SCALAR_IMPL selects
├── custom/                     A: today's files, moved, custom_ prefix
│   ├── custom_scalar_core.sv
│   ├── custom_fetch.sv, custom_decode.sv, custom_controller.sv
│   └── custom_dma.sv, custom_sync_fifo.sv   (custom_dma.sv is today's scalar_core_lsu.sv)
└── cva6/                       B
    ├── upstream/               submodule at v5.3.0, unmodified (includes the pulp axi library)
    ├── cva6_config.sv          CVA6 configuration (package cva6_npu_config_pkg):
    │                           DDR cacheable, MMIO non-cacheable and non-idempotent,
    │                           CvxifEn=0
    ├── scalar_core_cva6.sv     top wrapper: cva6 + start/done control + axi_demux / axi_mux /
    │                           axi_id_serialize from the pulp axi library
    ├── cva6_npu_mmio.sv        透過暫存器與 npu 溝通
    ├── cva6_dma.sv             收到命令後，自動把 A、B 搬進 matrix core，再把 C 搬回 DDR
    ├── cva6.f                  file list in package order
    ├── waiver.vlt              Verilator lint off for upstream/
    └── README.md               pin, config choices, patch policy
```

```systemverilog
parameter string SCALAR_IMPL = "CUSTOM";   // or "CVA6"
generate
  if (SCALAR_IMPL == "CVA6") scalar_core_cva6   #(...) u_core (.*);
  else                       custom_scalar_core #(...) u_core (.*);
endgenerate
```

### 4.2 Why each function exists

| Function in B | File | Why it must exist | Same function in A |
|---|---|---|---|
| Address map | `cva6_config.sv` | CVA6 must know which addresses are cacheable DDR and which are side-effecting MMIO | not needed: A has no cache and no MMIO |
| Start / done control | `scalar_core_cva6.sv` | turn `isa_start` into a CPU run and a DONE write into `isa_done` / `isa_error` | `scalar_core_controller.sv` |
| Address split | `scalar_core_cva6.sv` (`axi_demux`) | CPU accesses go either to DDR or to the NPU registers | not needed: A has no loads/stores of its own |
| Registers + issue | `cva6_npu_mmio.sv` | the firmware's view of a GEMM job; `ctl_start` and shape registered | decode in `scalar_core_decode.sv` + EX issue in `scalar_core.sv` |
| Tensor DMA | `cva6_dma.sv` | move A/B operands into the AXIS stream and C back to DDR | `scalar_core_lsu.sv` (becomes `custom/custom_dma.sv` when moved) |
| AXI arbitration | `scalar_core_cva6.sv` (`axi_mux`, `axi_id_serialize`) | CPU and tensor DMA share the single ID-less AXI4 master | inside `scalar_core.sv` (read tag FIFO) |

CVA6 replaces fetch and decode. The only genuinely new function is the
address split; the rest exists in A in another form.

Naming: CVA6 has its own internal load/store unit, which executes `lw` / `sw`
through the D$. Ours is a tensor DMA that streams the A / B operands to the
matrix core and writes C back with AXI bursts while the CPU continues, so it
is named DMA to avoid confusing the two.

Alternative, not chosen: a finer split into `cva6_ctrl.sv`, `cva6_xbar.sv`,
`cva6_issue.sv` and `cva6_axi_arb.sv`. It adds files without adding
separable test surface at this size.

### 4.3 Configuration

The stock `cv32a60x_config_pkg.sv` (v5.3.0) has: XLEN 32, `MmuPresent=0`,
`RVS=0`, `RVU=0`, `CvxifEn=1`, `SuperscalarEn=0`, `IcacheByteSize=2048`,
`DCacheType=HPDCACHE_WT`, `AxiDataWidth=64`, `AxiAddrWidth=64`,
`AxiIdWidth=4`, `NrCachedRegionRules=1` with base `0x8000_0000` and length
`0x4000_0000`, `NrNonIdempotentRules=0`. `cv32a65x` is the same with
`SuperscalarEn=1`.

`cva6_config.sv` is a configuration file only: it holds one constant
`cva6_cfg` struct and generates no hardware. Upstream customizes CVA6 this
way: `cva6.sv` expands the user struct with
`build_config_pkg::build_config(cva6_cfg)`, so no upstream RTL changes.

`cv32a60x` has the right shape but targets another board. The PYNQ-Z1 DDR
sits at `0x0000_0000` (512 MB), so `cva6_config.sv` derives from `cv32a60x`
and changes:

| Field | `cv32a60x` | Ours | Why |
|---|---|---|---|
| `CachedRegionAddrBase` / `Length` | `0x8000_0000` / 1 GB | `0x0` / DDR size | otherwise DDR is never cached and the 2 KB I$ / D$ do nothing |
| `NrNonIdempotentRules` + base / length | 0 | 1, covering the MMIO window (outside the cached region) | `DOORBELL` and `FENCE` have side effects; CVA6 must not speculate or repeat those accesses |
| `NrExecuteRegionRules` + base / length | 0 (unrestricted) | 1, DDR only (recommended) | a jump into MMIO raises an exception instead of running garbage |
| `CvxifEn` | 1 | 0 | no custom instructions; saves area |
| `AxiAddrWidth` | 64 | try 32 | matches the 32-bit `m_axi`; if upstream requires 64, keep it and truncate the upper bits in `scalar_core_cva6.sv` |

Area knobs, decided at the P2 synthesis gate (no functional effect for the
firmware, which is built with `-march=rv32imc`): `RVB` and `RVZCB` 1 → 0,
`RvfiTrace` 1 → 0 (verification trace only), fewer `BHTEntries`.

Unchanged: `XLEN=32`, `MmuPresent=0`, `RVS=0`, `RVU=0`, `SuperscalarEn=0`
(dual issue is B' with its own configuration), cache sizes,
`DCacheType=HPDCACHE_WT` (write-through keeps DDR current at the end),
`DebugEn=0`, `AxiIdWidth=4` (removed by `axi_id_serialize`, so the external
port stays ID-less).

Naming. Every upstream configuration file declares the same package,
`cva6_config_pkg`, and one of them stays in `cva6.f` because the default
value of `cva6.sv`'s `CVA6Cfg` parameter refers to it. Our package therefore
has a distinct name, `cva6_npu_config_pkg` (not `config`, a SystemVerilog
keyword, and not `config_pkg`, which upstream already declares), and
`scalar_core_cva6.sv` passes it explicitly:

```systemverilog
cva6 #(
  .CVA6Cfg(build_config_pkg::build_config(cva6_npu_config_pkg::cva6_cfg))
) u_cva6 (...);
```

The file name differs from the package name; lint already runs with
`-Wno-DECLFILENAME`. A search of upstream `core/` found no other direct
reference to `cva6_config_pkg::`; confirm when the submodule is first built.

### 4.4 Start, done and clock

- CVA6 is held in reset (`rst_ni` low) until `isa_start`; `boot_addr_i` =
  `prog_addr`. Reset also leaves the caches clean, so each run starts from
  DDR contents.
- The D$ is write-through, so DDR is current when the program ends; no flush
  step is needed before `isa_done`.
- A write to the DONE register ends the program: CVA6 is put back into reset,
  a non-zero code sets `isa_error` / `isa_error_code`, then `isa_done`.
- `hart_id_i = 0`; `irq_i`, `ipi_i`, `time_irq_i`, `debug_req_i` tied low;
  `cvxif_resp_i` tied off; `noc_req_o` / `noc_resp_i` is the AXI side.
- Clock risk: 100 MHz on the -1 XC7Z020 is not assured for CVA6. If synthesis
  misses it, CVA6 runs in its own slower clock domain with the CDC at the MMIO
  boundary (and the AXI side), leaving `ctl_start` in the 100 MHz domain.

### 4.5 As implemented (#174)

Implemented on `npu/issue174-a`: `make -C src/test lint sim` passes with the
default `CUSTOM` (all `tb_scalar_core` cycle counts unchanged), the new
`tb_cva6_npu_mmio` passes under iverilog, and `make lint-cva6` elaborates
`scalar_core` with `SCALAR_IMPL="CVA6"` (225 modules) clean under `-Wall`.
`make sim-cva6` (Verilator generating C++, WSL g++ compiling it) runs the
seven ISA test programs translated to RISC-V; all final memory images equal
`isa_sim.py`. Nothing has been synthesized.

Deviations from sections 4.1 to 4.4:

- `axi_mux` (CVA6 DDR path + DMA) sits before `axi_id_serialize`, so the
  ID-less `m_axi` sees a constant ID and no response-ID rebuild is needed.
- `axi_demux` has a third port: unmapped addresses get DECERR from
  `axi_err_slv` instead of reaching DDR.
- `AxiAddrWidth` stays 64: with 32 the data cache silently truncates its
  34-bit physical address. The top drops the upper bits after the decode.
- MMIO faults (bad shape, misaligned address) are latched; DOORBELL still
  answers OKAY so the program does not trap, FENCE answers once a fault is
  latched, and the fault ends the program.
- The wrapper selects CVA6 under `` `ifdef NPU_SCALAR_CVA6 `` (defined by
  `cva6.f`), so the default iverilog / Verilator flow never needs the CVA6
  sources.
- `cva6_dma.sv` also defines `cva6_sync_fifo`, so `cva6/` does not depend on
  `custom/`.
- Provisional values: `isa_instructions` counts NPU jobs issued, `isa_pc`
  reports the boot address (section 9).
- Windows checkouts need `git -c core.longpaths=true` because of the nested
  submodule path length.

## 5. MMIO register draft

The authoritative definition lives in `src/vendors/risc_v_npu/spec/abi.py`;
the generated `RvNpuMmio.h` and the tests derive from it, and tests parse
the RTL to check it. MMIO fault codes 0x11–0x14 are the `ERR_*` values of
`npu_pynq/spec/isa.py`; DONE code 0x20 (trap) is used by `crt0.S`. Offsets are relative to the
MMIO base (open decision).

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x00 | `SHAPE_MN` | RW | M `[15:0]`, N `[31:16]` |
| 0x04 | `SHAPE_K` | RW | K `[15:0]` |
| 0x08 | `ADDR_A` | RW | A operand address |
| 0x0C | `ADDR_B` | RW | B operand address |
| 0x10 | `ADDR_C` | RW | C result address |
| 0x14 | `DOORBELL` | W | issue one GEMM job from the registers above; the AXI write response is delayed while the job queue is full |
| 0x18 | `STATUS` | R | busy, queue level, sticky error and error code |
| 0x1C | `FENCE` | R | the read returns only once every issued job has drained |
| 0x20 | `DONE` | W | write an error code (0 = success) to end the program |
| 0x24 | `DATA_BASE` | R | the CSR `data_base` of this run (added in #174) |
| 0x28 | `PROG_LEN` | R | the CSR `prog_len` of this run (added in #174) |

## 6. Software

### 6.1 From a fixed schedule to a program

| | A (today) | B (this plan) |
|---|---|---|
| Back end emits | a fixed schedule: 64-bit ISA words (`SHAPE`, `ADDR_*`, `GEMM`, `REPEAT`, `INCR`, `END`) | a program: RISC-V machine code in an ELF |
| Scalar core | replays the schedule, no decisions | runs the program: loops, branches, scalar arithmetic |
| Compact form of many jobs | `REPEAT` / `INCR` | ordinary loops |

The NPU-specific decisions (which contractions run on the NPU, tiling, job
order, weight layout) stay in the back end for both. The RISC-V compiler
optimizes only the control code (register allocation, instruction
selection, loops); it does not know the NPU.

### 6.2 A separate device: vendor `risc_v_npu`, delegate `risc_v_npu`

B is a new device, not an option inside `npu_pynq`. Under the delegate
framework that `dev` adopted in #178 (plan 2026-1113-001,
`docs/rules/lib/export_stages.md`), a new device is a match predicate the
middle end links plus a delegate in the back end. So:

- A's path, `src/vendors/npu_pynq/` and `src/export/backend/delegates/npu/`,
  is not changed at all.
- B lives in `src/vendors/risc_v_npu/` (spec, compiler dialect, CVA6
  runtime, ARM driver) and `src/export/backend/delegates/risc_v_npu/`.
- A config selects the device:

```yaml
devices: [cpu, npu]          # A: custom scalar core
devices: [cpu, risc_v_npu]   # B: CVA6
```

Naming: `risc_v_npu`, not `risc-v`. `src/vendors/` folders are Python
packages (`src.vendors.npu_pynq`), so a name with `-` cannot be imported.
Both dialects are linked into the same `pynqo-backend`, so B's names must
not collide with A's: dialect `rvnpu` (A: `npu`), its own C++ namespace and op
names, driver library `librv_npu_rt` (A: `libnpu_rt`).

Shared matrix-core knowledge is copied, not imported (confirmed 2026-11-14, also for the claim predicate).
The matrix core is the same, so `layout.py`, `quant.py`, `board.py`,
`Claim.cpp` and `Pack.cpp` start as copies of the `npu_pynq` files. This keeps
the two devices independent, as the hardware implementations are
(section 1), at the cost of updating both when the matrix core's contract
changes. Alternative, not chosen: `risc_v_npu` imports `npu_pynq/spec/` and
reuses its passes.

How `risc_v_npu` plugs into the delegate framework (aligned after #178):

| Stage | `npu` (A) | `risc_v_npu` (B) |
|---|---|---|
| Middle end: assignment | `npu::isClaimable` (NpuMatch) in `AssignDevices`' `kMatchers` | `rvnpu::isClaimable` (RvNpuMatch, a copy) in the same table |
| Middle end: partitioning | `ir/delegates/npu.mlir`, one `@npu_<id>` each | `ir/delegates/risc_v_npu.mlir`, one `@risc_v_npu_<id>` each |
| CPU graph | bodiless `@__delegate_<id>` declaration, lowered to `__delegate__(id, in, out)` | same; the CPU graph cannot tell the two apart |
| Back end | `compile(npu.mlir)`: claim, Pack, Encode → `programs.*` | `compile(risc_v_npu.mlir)`: claim, Pack, LowerToRiscv, rv_codegen → `rv_programs.*` |
| Delegate output | `M×N` INT32; `libnpu_rt` sums the K-chunk planes | `M×N` INT32; `librv_npu_rt` sums the K-chunk planes on ARM (wrapping, as `libnpu_rt`) |

The K-plane sum stays on ARM in both devices, so the two produce
bit-identical outputs and the CVA6 program only issues jobs. Summing on CVA6
(a RISC-V loop over the C planes) or accumulating across K chunks in the
matrix core are later options, each its own A/B.

Lowering in the `risc_v_npu` dialect:

```
rvnpu.matmul                       (Claim.cpp, copied: INT8 contractions on constant weights)
  │ LowerToRiscv: tile loops as scf.for; each job a run of volatile stores to
  │               the MMIO registers (SHAPE, ADDR_A/B/C, DOORBELL), then a FENCE
  │               load and a DONE store
  ▼
scf / arith / func / memref + volatile llvm.load / llvm.store
  │ standard conversions to the LLVM dialect, mlir-translate
  ▼
LLVM IR
  │ LLVM RISC-V back end: riscv32, -march=rv32imc, -mabi=ilp32, -O2
  ▼
object
  │ lld with crt0.S (assembled by LLVM MC) and link.ld
  ▼
ELF → flat binary loaded at prog_addr
```

- No C runtime: `pynqo-backend` links the LLVM libraries but not Clang, so
  it cannot compile C. The pass emits the MMIO accesses directly, which is
  what an inlined `npu_gemm()` would compile to.
- RV32IMC, not RV32I: our `cv32a60x`-derived configuration has M and C
  (`RVB` / `RVZCB` may be removed at the synthesis gate, so the compiler must
  not use them).
- LLVM rather than GCC: `pynqo-backend` already builds against LLVM 22. The
  build must include the `RISCV` target (`LLVM_TARGETS_TO_BUILD`); not yet
  checked.
- Addresses are relative to `data_base`, as in A; the program reads
  `DATA_BASE` (0x24) at start. B segment relocation follows the scheme of
  `npu_pynq/custom_dialect/lib/Encode.cpp`.
- Granularity (first step): one binary per delegate, the unit the partition
  already outlines. Non-GEMM operations (bias, requantize, pooling, softmax)
  stay on the ARM CPU. Whole-model control flow or non-GEMM work on CVA6 is a
  later step (section 9).

### 6.3 Layout: files to add and change

Legend: `+` new, `M` modified, `=` listed for reference, unchanged.

Vendor `risc_v_npu` (new)

```
src/vendors/risc_v_npu/                   +
├── __init__.py                           +
├── spec/                                 +  the hardware as data
│   ├── abi.py                            +  CSR map (with the CVA6 capability bit), MMIO register
│   │                                        table, address map
│   ├── layout.py                         +  B tiles in DDR (copied from npu_pynq)
│   ├── quant.py                          +  accepted quantization schemes (copied)
│   └── board.py                          +  the board's CPU (copied)
├── custom_dialect/                       +  compile time (C++)
│   ├── include/RvNpu/
│   │   ├── RvNpuOps.td                   +  rvnpu.matmul
│   │   ├── RvNpuPasses.h                 +
│   │   ├── RvNpuMatch.h                  +  the claim predicate the middle end links (copy of NpuMatch)
│   │   └── RvNpuMmio.h                   +  MMIO offsets, generated from spec/abi.py
│   ├── lib/
│   │   ├── Claim.cpp                     +  copied from npu_pynq, produces rvnpu.matmul
│   │   ├── Pack.cpp                      +  copied from npu_pynq
│   │   └── LowerToRiscv.cpp              +  rvnpu.matmul → scf.for + volatile MMIO stores
│   ├── runtime/                          +  linked into every generated program
│   │   ├── crt0.S                        +  stack, clear .bss, jump to the entry; DONE(code) on a trap
│   │   └── link.ld                       +  program address, stack placement
│   └── CMakeLists.txt                    +
└── driver/                               +  run time (ARM)
    ├── rv_npu_rt/                        +  load the RISC-V binary, set prog_addr / data_base,
    │                                        start, wait for isa_done (no SIM backend: SIM is the
    │                                        npu_pynq ISA model and cannot run RISC-V)
    └── overlay/                          +  manifest and verification of the CVA6 overlay
```

The CVA6 runtime sits under `custom_dialect/` because the back end links it
at compile time; `spec/`, `custom_dialect/` and `driver/` keep the vendor
rule of `docs/rules/filetree.md` (`driver/` never needs the compiler).

Back end

```
src/export/backend/
├── delegates/
│   ├── npu/                              =  A, unchanged
│   └── risc_v_npu/                       +
│       ├── lower.cpp / lower.h           +  the delegate: compile(ir/delegates/risc_v_npu.mlir)
│       └── rv_codegen.cpp / .h           +  LLVM IR → RV32IMC object → link → ELF → binary
├── device_register.cpp                   M  devices [cpu, risc_v_npu]
├── package.cpp / .h                      M  only if the package cannot already carry a
│                                            delegate-specific artifact (to check)
└── CMakeLists.txt                        M  the new delegate and dialect; LLVM RISCV target and lld
```

Middle end

```
src/export/middle-end/
├── linalg/transforms/AssignDevices.cpp   M  kMatchers += {"risc_v_npu", rvnpu::isClaimable}
└── CMakeLists.txt                        M  link RvNpuMatch
```

Run time

```
src/runtime/                              M  __delegate__ dispatches to libnpu_rt or librv_npu_rt
                                             by the package's device (to check how it binds today)
src/vendors/npu_pynq/                     =  A, unchanged
```

Hardware follow-ups (not done in #174)

```
src/hw/rtl/npu_accelerator/
├── npu_accelerator.sv                    M  pass SCALAR_IMPL down
└── csr/csr.sv                            M  CVA6 bit in ABI_CAPABILITIES
src/hw/vivado_tcl/npu_accelerator/
└── build_overlay.tcl                     M  read cva6.f, define NPU_SCALAR_CVA6, pass SCALAR_IMPL
```

Tests and A/B

```
src/test/vectors/risc_v_npu/              +
├── rv32asm.py                            +  small RV32IMC assembler in Python (no external toolchain)
└── isa_to_rv.py                          +  existing ISA test programs → RISC-V machine code
                                             (REPEAT / INCR become loops)
src/hw/tb/npu_accelerator/
└── tb_scalar_core_cva6.sv                +  CVA6 system simulation (Verilator); final memory image
src/test/Makefile                         M  Verilator rule for the CVA6 testbench

exp/scalar_core_compare/                  +
├── README.md                             +  A/B report: correctness, cycles per workload, program size, area
└── run.py                                +  reruns A and B
```

Rules and documents

```
docs/rules/filetree.md                    M  src/vendors/risc_v_npu/, delegates/risc_v_npu/,
                                             src/test/vectors/risc_v_npu/, exp/scalar_core_compare/
```

Summary

| Area | New | Modified |
|---|---|---|
| Vendor `risc_v_npu` | `spec/` (4 files); `custom_dialect/`: `RvNpuOps.td`, `RvNpuPasses.h`, `RvNpuMmio.h`, `Claim.cpp`, `Pack.cpp`, `LowerToRiscv.cpp`, `runtime/`; `driver/`: `rv_npu_rt/`, `overlay/` | — |
| Back end | `delegates/risc_v_npu/` (`lower.*`, `rv_codegen.*`) | `device_register.cpp`, `CMakeLists.txt`, `package.*` if needed |
| Run time | — | `src/runtime/` dispatch |
| A (`npu_pynq`, `delegates/npu`) | — | — (unchanged) |
| Hardware follow-ups | — | `npu_accelerator.sv`, `csr.sv`, `build_overlay.tcl` |
| Tests and A/B | `rv32asm.py`, `isa_to_rv.py`, `tb_scalar_core_cva6.sv`, `exp/scalar_core_compare/` | `Makefile` |

### 6.4 Tests and A/B

- Correctness: the existing ISA test programs (`generate_isa_vectors.py`)
  are translated to equivalent RISC-V machine code by `isa_to_rv.py` (using
  the in-repository `rv32asm.py`, so the tests need no RISC-V toolchain) and
  run on B. A runs the original programs. Both final memory images must equal
  `isa_sim.py`, which stays the golden model unchanged.
- Then the back-end path: the same GEMMs taken through `--npu-claim` and
  `LowerToRiscv`, checked the same way.
- The A/B report goes to `exp/scalar_core_compare/`: per workload cycles
  (`isa_cycles`), area (Yosys estimate, then Vivado), and program size
  (ISA words versus ELF bytes).

## 7. Repository, rule and tool changes

| Area | Change |
|---|---|
| `docs/rules/filetree.md` | add `scalar_core/{custom,cva6}/`; `cva6/upstream/` as a pinned, unmodified third-party submodule (done in #174); `src/vendors/risc_v_npu/`, `src/export/backend/delegates/risc_v_npu/`, `src/test/vectors/risc_v_npu/`, `exp/scalar_core_compare/` (with the software) |
| `src/test/Makefile` | CVA6-only files left out of the default `RTL`, `lint-cva6` target (done in #174); still to add: a Verilator simulation rule for the CVA6 testbench (needs a C++ compiler) and building the test programs before it |
| CI (`ci.yml`) | checkout with the CVA6 submodule and its three nested submodules for the CVA6 jobs; a Clang/LLVM with the RISC-V target; check the Verilator version (CI installs it by apt on `ubuntu-latest`; CVA6 needs a recent 5.x) |
| `build_overlay.tcl` | read `cva6.f`, define `NPU_SCALAR_CVA6`, pass `SCALAR_IMPL` (needs the parameter on `npu_accelerator`) |
| ABI | one capability bit reporting the CVA6 scalar core (minor ABI bump) |

The CSR ABI otherwise stays: `isa_start`, `prog_addr`, `prog_len`,
`data_base`, `isa_done`, `isa_error`, `isa_cycles` keep their meaning; on B
`prog_addr` points at the delegate's RISC-V binary instead of ISA words.

Tools in this environment: Verilator (Windows, msys) generates C++ and WSL
Ubuntu's g++ compiles it (`make sim-cva6`). WSL also has LLVM 22.1.8 at
`/usr/lib/llvm-22` with MLIR, clang and the RISCV target, plus `ld.lld-22`
(no lld libraries, so linking calls the executable, as the CPU path does).
There is no Yosys, so area, Fmax and power are not measured.

## 8. Phases

| Phase | Work | Exit criterion |
|---|---|---|
| P0 | rule updates, submodule at v5.3.0 (done, #174); CI toolchain (Clang/LLVM RISC-V, Verilator) | CI green with the submodule excluded from the default flow |
| P1 | move A into `custom/` with `custom_` prefixes; wrapper with `SCALAR_IMPL` (done, #174) | `make -C src/test lint sim` passes unchanged; A baseline recorded (`tb_scalar_core`: 7 programs, e.g. `gemm_33x20x320` 18 jobs / 6629 cycles) |
| P2 | synthesis-only check of `cv32a60x` (and `cv32a65x`) on XC7Z020: LUT / FF / BRAM, Fmax, via `ic synth` (Yosys estimate) and Vivado | go / no-go gate before more integration work (the RTL of P3 was written ahead of it) |
| P3 | hardware: config pkg, MMIO, top, DMA copy (done and lint-clean, #174); `crt0.S` / `link.ld`, `rv32asm.py` / `isa_to_rv.py`, Verilator system testbench | final memory image equals `isa_sim.py` on the existing vector set |
| P4 | vendor `risc_v_npu` and delegate `risc_v_npu` (section 6.3), runtime dispatch | back-end-generated ELF passes the same checks |
| P5 | A/B report in `exp/scalar_core_compare/`; Vivado timing and power; board run | table filled for A and B (B' if P2 allowed) |

## 9. Open decisions

1. MMIO base address and the size of the cached DDR region (#174 uses
   `0x4000_0000`, 4 KB, and DDR `0x0000_0000`–`0x1FFF_FFFF`).
2. Meaning of `isa_instructions` on B: CPU retired instructions or NPU jobs
   (provisionally jobs); and `isa_pc` (provisionally the boot address).
3. Issue mapping: #174 covers the hardware; the software (sections 6.2 to
   6.4) as the same issue or a new sub-issue under #166.
4. Granularity after the first step: one ELF per delegate (first step), or
   one program for the whole model so CVA6 drives the control flow between
   layers.
5. Where non-GEMM operations run: on the ARM CPU (first step), on CVA6, or on
   the vector core (#169).
6. Resolved: the LLVM 22 that `pynqo-backend` builds against has the RISCV
   target (WSL `/usr/lib/llvm-22`).
7. Where CI gets Verilator: the apt package (if recent enough) or a pinned
   build.
8. Resolved: the device is named `risc_v_npu` (`risc-v` is not a valid
   Python package name).
9. Resolved (2026-11-14): matrix-core knowledge and the claim predicate are
   copied from `npu_pynq`; the K-plane sum is done on ARM by `librv_npu_rt`.
10. How `src/runtime/` binds `__delegate__` to a driver today, and so what
    the dispatch to `librv_npu_rt` needs.
