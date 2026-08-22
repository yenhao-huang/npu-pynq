# Verification Matrix

## Evidence layers

| Layer | Command/artifact | What it proves | What it does not prove |
| --- | --- | --- | --- |
| Python golden model | `python -m unittest ...` | signed math, wrap, driver encoding/API | RTL or FPGA behavior |
| XSIM | `test/run_xsim.ps1` | RTL elaboration and AXI transaction behavior | synthesis or board access |
| Vivado implementation | routed reports + `.bit` | synthesizable, routable, timed FPGA design | Python/ARM MMIO behavior |
| HWH inspection | `overlay/mac_npu.hwh` | IP instance/address metadata | bitstream actually loaded |
| Board smoke test | `deploy_and_test.ps1` | complete ARM→AXI→PL→AXI→ARM path | high-throughput performance |

Never promote evidence from a lower row to satisfy a higher row.

## Python tests

Run without creating caches:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s sw/mac_npu -p 'test_*.py' -v
```

Required coverage:

- directed `2*3 + 4*5 = 26` reference case;
- positive×negative and negative×negative;
- `-128`, `127`, zero, and signed 32-bit wrap;
- 256 deterministic random pairs using the fixed seed;
- rejected out-of-range/type inputs and unequal vector lengths;
- fake-MMIO clear/start/accumulate/signed decode;
- finite timeout when done never asserts.

## XSIM

Run:

```powershell
& .\mount\mac_npu\test\run_xsim.ps1
```

`tb_mac_axi_lite.sv` must drive AW/W/B and AR/R handshakes, not reach into DUT
internals. Require this exact PASS marker:

```text
PASS: AXI4-Lite writes a/b/clear/start and reads accumulator
```

The current test completes at 560 ns. Time may change after testbench edits;
the self-checking assertions and PASS are authoritative.

`run_rtl_sim.ps1` uses Icarus Verilog for the isolated MAC core and resolves
`iverilog`/`vvp` from PATH or the standard MSYS2 UCRT64 installation. For
installation and repair, read `installation/icarus-verilog-windows.md`. XSIM
remains the AXI4-Lite wrapper verification route through Vivado 2026.1.

## Vivado implementation

Require all of:

- synthesis 0 errors;
- placement and routing successful;
- no failed/unrouted/partially routed nets;
- routed WNS ≥ 0 and zero setup-failing endpoints;
- pre-bitstream DRC 0 errors;
- `write_bitstream completed successfully`;
- deployable bit/HWH pair exists and HWH address metadata matches the contract.

Warnings must be inspected in context. Do not ignore an unknown critical
warning merely because a bitstream exists.

## Hardware smoke test

Required board assertions:

```text
clear -> 0
2*3 -> 6
accumulate -7*6 -> -36
clear -> 0
-128*-128 -> 16384
accumulate 127*-128 -> 128
```

Require the final PASS marker. A timeout, missing IP, assertion failure, or
overlay load failure means the milestone remains incomplete.

## Revalidation scope

| Changed item | Minimum rerun |
| --- | --- |
| Python formatting/docs only | Python syntax/link checks as applicable |
| Golden model or driver | all Python tests + board smoke test |
| MAC RTL | Python tests, XSIM, Vivado build, board smoke test |
| AXI wrapper/register map | Python tests, XSIM, Vivado build/HWH, board smoke test |
| Vivado Tcl/clock/address | Vivado build, reports/HWH, board smoke test |
| Bitstream or HWH | artifact pairing/HWH checks + board smoke test |
| Board script | Python syntax/tests + board smoke test |
