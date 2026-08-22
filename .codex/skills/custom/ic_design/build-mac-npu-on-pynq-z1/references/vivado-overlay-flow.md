# Vivado Overlay Build and Evidence

## Contents

1. Environment
2. Block design topology
3. Batch build
4. Artifact inspection
5. Timing and DRC evidence
6. Failure recovery

## Environment

The current Windows installation is:

```text
C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat
part: xc7z020clg400-1
license: Vivado Basic supports this device
```

`build_overlay.ps1` is the stable entry point. It calls
`src/vivado_tcl/mac_npu/build_overlay.tcl`; keep GUI-only edits out of the canonical
flow unless they are reproduced in Tcl.

## Block design topology

The Tcl script must produce:

```text
processing_system7_0/M_AXI_GP0
  -> AXI interconnect/protocol converter
  -> mac_axi_lite_0/s_axi

processing_system7_0/FCLK_CLK0
  -> AXI interconnect + mac_axi_lite_0/s_axi_aclk

processor_system_reset/peripheral_aresetn
  -> mac_axi_lite_0/s_axi_aresetn
```

Expose PS DDR and FIXED_IO. Enable `M_AXI_GP0` and `FCLK_CLK0`. The validated
current design records a 50 MHz FCLK in HWH; do not claim 100 MHz unless the PS
property and generated HWH both prove it.

Assign `mac_axi_lite_0/s_axi/reg0`:

```text
base:  0x43C00000
high:  0x43C0FFFF
range: 64 KiB
```

## Batch build

Run:

```powershell
& .\mount\mac_npu\core\api\build_overlay.ps1
```

The Tcl script recreates the project under `results/vivado/mac_npu`, generates
the block design/wrapper, launches `impl_1 -to_step write_bitstream`, waits, and
copies deployable outputs to `mount/mac_npu/overlay`.

Vivado may spawn several child processes for out-of-context synthesis; this is
normal. Do not terminate an existing interactive Vivado session without user
authorization. If sandboxed launch cannot read license/user settings, run the
same narrowly scoped batch command with approved elevated execution rather than
changing license files.

## Artifact inspection

Required files:

```text
mount/mac_npu/overlay/mac_npu.bit
mount/mac_npu/overlay/mac_npu.hwh
```

They must share a basename because `pynq.Overlay` locates the HWH beside the
bitstream.

Inspect address metadata:

```powershell
[xml]$hwh = Get-Content -Raw mount\mac_npu\overlay\mac_npu.hwh
$hwh.SelectSingleNode("//*[@INSTANCE='mac_axi_lite_0' and @BASEVALUE='0x43C00000']")
```

Confirm `HIGHVALUE="0x43C0FFFF"`, AXI protocol `AXI4LITE`, data width 32, and a
clock frequency. Compare hashes/timestamps when uncertain whether outputs came
from the current source.

## Timing and DRC evidence

Inspect:

```text
results/vivado/mac_npu/mac_npu.runs/impl_1/
  mac_npu_bd_wrapper_timing_summary_routed.rpt
  mac_npu_bd_wrapper_drc_routed.rpt
  runme.log
```

Acceptance:

- synthesis and implementation finish without errors;
- `write_bitstream completed successfully` appears near the end of `runme.log`;
- pre-bitstream DRC reports zero errors;
- routed WNS is non-negative and TNS failing endpoints are zero;
- route status has no failed/unrouted/partially routed nets.

The observed successful build had WNS `11.436 ns`; treat that as historical
evidence, not a universal requirement after design changes.

## Failure recovery

| Symptom | Action |
| --- | --- |
| `vivado` not on PATH | Use the absolute 2026.1 path through `build_overlay.ps1` |
| user apps/load_features failure | Check for concurrent Vivado and user-settings access; do not kill GUI automatically |
| license missing only in isolated HOME | Restore normal user profile; isolation hid the valid license |
| module interface not inferred | Check `s_axi_*` names, widths, clock/reset names, and SystemVerilog top |
| address segment differs | Re-run explicit `assign_bd_address -offset ... -range ... -force` |
| HWH missing | Confirm `generate_target all` completed and inspect `hw_handoff` path |
| bitstream missing | Read the end of `impl_1/runme.log`; use final DRC/implementation error |
| timing violation | Inspect failing paths before changing clocks or adding pipelines |

After changing Tcl, regenerate both artifacts; never pair a new bitstream with
an old HWH.
