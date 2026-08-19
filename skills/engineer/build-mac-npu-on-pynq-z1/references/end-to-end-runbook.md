# End-to-End MAC NPU Runbook

## Contents

1. Working assumptions
2. Phase 0: inspect and resume
3. Phase 1: arithmetic model
4. Phase 2: pure RTL
5. Phase 3: AXI4-Lite peripheral
6. Phase 4: software driver
7. Phase 5: Vivado overlay
8. Phase 6: board deployment
9. Completion audit
10. Next architecture milestone

## Working assumptions

Run commands from the repository root:

```text
C:\Users\User\Desktop\agent_workspace\npu\npu_repo_in_pynq
```

Canonical paths:

| Concern | Path |
| --- | --- |
| Pure MAC RTL | `src/rtl/mac_npu/mac_unit.sv` |
| AXI wrapper | `src/rtl/mac_npu/mac_axi_lite.sv` |
| Python golden model | `sw/mac_npu/mac_reference.py` |
| PYNQ driver | `sw/mac_npu/mac_mmio.py` |
| RTL testbenches | `src/tb/mac_npu/tb_mac_unit.sv`, `tb_mac_axi_lite.sv` |
| Python tests | `sw/mac_npu/test_mac_reference.py`, `test_mac_mmio.py` |
| Vivado Tcl | `src/vivado_tcl/mac_npu/build_overlay.tcl` |
| Generated overlay | `mount/mac_npu/overlay/mac_npu.bit`, `mac_npu.hwh` |
| Board smoke test | `sw/mac_npu/hardware_smoke_test.py` |
| Deploy entry point | `sw/mac_npu/deploy_and_test.ps1` |

Do not edit files under `results/`; they are generated evidence and may be
deleted/rebuilt. Do not put credentials in source, config, logs, or state.

## Phase 0: inspect and resume

1. Read this skill's `STATE.md` and the applicable references.
2. Inspect `git status --short`; preserve unrelated user changes.
3. Confirm the target files exist with `rg --files mount/mac_npu`.
4. Inspect `configs/pynq-sync.json` for the authoritative board host, user, and
   remote root. Do not invent a different address.
5. Check tools:

```powershell
Get-Command python,ssh,scp -ErrorAction Stop
Get-Item C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat
```

Resume at the first phase whose gate lacks current evidence. Rebuild downstream
artifacts whenever an upstream contract or RTL file changes.

## Phase 1: arithmetic model

Lock these values before implementation:

```text
a, b:              signed INT8
accumulator:       signed INT32
operation:         accumulator + a*b
overflow:          two's-complement wrap modulo 2^32
priority:          reset > clear > valid/start
accepted rate:     one pair per clock in the pure core
```

Update these together if the contract changes:

- `configs/mac_mvp.json`
- `docs/spec.md`
- `mac_reference.py`
- RTL and both test layers
- skill architecture/register references

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s sw/mac_npu -p 'test_*.py' -v
```

Gate: directed signed/boundary tests, deterministic random vectors, wrap
behavior, MMIO encoding, timeout, and accumulation tests all pass.

## Phase 2: pure RTL

Implement arithmetic only in `mac_unit.sv`; do not place AXI logic there.
Required behavior at each rising edge:

1. `rst_n == 0`: clear result and result-valid.
2. Else `clear == 1`: clear result and result-valid.
3. Else `valid == 1`: sign-extend the product, accumulate, pulse result-valid.
4. Else: hold result and deassert result-valid.

The testbench must cover reset, clear priority, idle hold, positive/negative
products, endpoints `-128` and `127`, and multiple consecutive accumulations.

Gate: no inferred behavior disagrees with the Python golden model. The current
end-to-end XSIM test also elaborates this core beneath the AXI wrapper.

## Phase 3: AXI4-Lite peripheral

Use the exact register and pulse semantics in
`rtl-and-axi-contract.md`. Keep AXI write address and write data independently
buffered: AXI4-Lite does not guarantee AW and W arrive in the same cycle.

Run:

```powershell
& .\mount\mac_npu\test\run_xsim.ps1
```

Required PASS:

```text
PASS: AXI4-Lite writes a/b/clear/start and reads accumulator
```

Gate: testbench performs genuine AXI handshakes, polls sticky done, reads signed
results, and verifies clear priority. A clean compile without the PASS is not
sufficient.

## Phase 4: software driver

Keep register offsets in `mac_mmio.py` identical to RTL. The driver must:

1. validate `a` and `b` are Python integers in `[-128, 127]`;
2. write their low 8-bit two's-complement encodings;
3. issue a pulse write for clear or start;
4. poll STATUS with a finite timeout;
5. convert the 32-bit result back to a signed Python integer;
6. discover address/range from `Overlay.ip_dict`, not hard-code it in normal
   driver use.

Gate: `test_mac_mmio.py` passes against its fake MMIO model, including timeout
and signed endpoint behavior.

## Phase 5: Vivado overlay

Follow `vivado-overlay-flow.md`. Build with:

```powershell
& .\mount\mac_npu\core\api\build_overlay.ps1
```

Gate:

- synthesis, placement, routing, DRC, and bitstream generation succeed;
- timing has no failing setup endpoints;
- the two overlay files share basename `mac_npu`;
- HWH contains `mac_axi_lite_0` at `0x43C00000..0x43C0FFFF`;
- bitstream and HWH timestamps correspond to the current build.

Record relevant report paths and values in `STATE.md`; do not paste entire
Vivado logs into the skill.

## Phase 6: board deployment

Follow `board-deploy-and-debug.md`. First prove physical reachability; then run:

```powershell
& .\mount\mac_npu\core\api\deploy_and_test.ps1
```

The script syncs the mount tree and invokes the remote Python smoke test.
Do not mark this phase complete from a dry-run sync or local fake-MMIO test.

Gate: capture the board PASS line and retain enough surrounding output to show
that `Overlay` loaded `mac_npu.bit`, found `mac_axi_lite_0`, and completed all
assertions without timeout.

## Completion audit

Before claiming the scalar milestone complete, prove every row:

| Requirement | Authoritative evidence |
| --- | --- |
| Python writes `a` and `b` | Board smoke test reaches positive and negative assertions |
| Python issues clear | Result reads zero immediately after clear |
| Python issues start | STATUS transitions to done and no timeout occurs |
| Hardware accumulates | `2*3` gives `6`, then `-7*6` gives `-36` |
| Signed endpoint handling | `-128*-128` then `127*-128` produces `16384`, then `128` |
| Python reads accumulator | Assertions use the real MMIO RESULT read |
| Correct overlay/IP | `Overlay.ip_dict` contains `mac_axi_lite_0` from matching HWH |

If any evidence is indirect, missing, or simulation-only, the milestone is not
complete.

## Next architecture milestone

Only after board PASS:

1. profile MMIO overhead;
2. define an AXI4-Stream vector protocol;
3. add AXI DMA and contiguous PYNQ buffers;
4. implement dot product, then parallel processing elements and tiled GEMM;
5. stabilize a Python runtime API;
6. consider MLIR/TVM/LLVM lowering last.
