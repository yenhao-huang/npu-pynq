# PYNQ-Z1 Deployment and Debug

## Contents

1. Authoritative connection settings
2. Physical/network gate
3. Sync and remote execution
4. Expected board sequence
5. Failure decision tree
6. Evidence and resume rules

## Authoritative connection settings

Read `configs/pynq-sync.json`. The current values are:

```text
host:        192.168.2.99
user:        xilinx
remote root: /home/xilinx/jupyter_notebooks/pynq_z1_repo
local root:  mount
```

Do not store a password. Use existing OpenSSH configuration or interactive
authentication. Do not delete remote files; the sync controller is upload-only.

## Physical/network gate

Before retrying SSH repeatedly, prove all layers:

1. PYNQ-Z1 power LED is on and Linux has booted from the PYNQ image.
2. Ethernet cable is connected and Windows reports the physical adapter `Up`,
   not `Media disconnected`.
3. PC adapter has a compatible static address such as `192.168.2.1/24`.
4. `arp -a` or ping/SSH activity can resolve/reach `192.168.2.99`.
5. TCP port 22 accepts a connection.

Useful read-only checks:

```powershell
Get-NetAdapter
Get-NetIPAddress -AddressFamily IPv4
arp -a
Test-NetConnection 192.168.2.99 -Port 22
```

If there is no active physical adapter and no `192.168.2.x` local address,
stop network commands: this is a physical/external-state blocker, not an SSH
credential or Python problem. USB/JTAG absence reinforces that the board is not
connected but JTAG is not required for PYNQ Overlay loading.

Do not change adapter addresses without user authorization; it affects host
network configuration.

## Sync and remote execution

Preview uploads without network mutation:

```powershell
& .\core\service\pynq_sync_controller.ps1 -Once -DryRun
```

Deploy and test:

```powershell
& .\mount\mac_npu\core\api\deploy_and_test.ps1
```

That script:

1. uploads changed files under `mount/` using the sync controller;
2. updates the local SHA-256 manifest only after successful transfer;
3. enters `/home/xilinx/jupyter_notebooks/pynq_z1_repo/mac_npu` over SSH;
4. runs `python3 hardware_smoke_test.py --bitfile overlay/mac_npu.bit`.

If diagnosing manually, first inspect the remote runtime:

```bash
python3 -c "import pynq; print(pynq.__version__)"
ls -l overlay/mac_npu.bit overlay/mac_npu.hwh
```

## Expected board sequence

`hardware_smoke_test.py` loads the overlay and uses `mac_mmio.py`:

```text
Overlay(mac_npu.bit)
verify mac_axi_lite_0 in ip_dict
construct MMIO from HWH phys_addr + addr_range
clear; assert result == 0
MAC(2, 3); assert result == 6
MAC(-7, 6); assert accumulated result == -36
clear
MAC(-128, -128); assert 16384
MAC(127, -128); assert accumulated result == 128
print PASS
```

This tests real ARM-to-PL MMIO, not a software substitute.

## Failure decision tree

| Failure | Likely layer | Next evidence/action |
| --- | --- | --- |
| Ethernet `Media disconnected` | physical | power board, connect cable, inspect link LEDs |
| No `192.168.2.x` PC address | host network | configure correct static subnet with user approval |
| Port 22 timeout | routing/boot | verify subnet, board boot, target IP, cable |
| Password/host-key error | SSH | use intended credentials/config; do not embed secrets |
| Sync fails mid-transfer | transport | rerun once connectivity is stable; manifest preserves successful-state semantics |
| `ModuleNotFoundError: pynq` | board image/runtime | verify command is running on PYNQ Linux with a PYNQ image |
| HWH parser/IP missing | artifact pairing | confirm matching basename, inspect `Overlay.ip_dict` and HWH instance |
| overlay download fails | bitstream/platform | verify PYNQ-Z1 target part and inspect kernel/FPGA manager errors |
| MMIO timeout waiting done | RTL/control | read STATUS/RESULT manually, re-run XSIM, inspect start/done timing |
| positive works, negative fails | signed encoding | inspect low 8-bit writes and signed 32-bit result conversion |
| result always zero | address/start/clock/reset | confirm HWH base, CONTROL write, FCLK, reset, and actual loaded overlay |
| stale result | sticky done/control | ensure start clears stale done before polling |

## Evidence and resume rules

Capture:

- connection target and PYNQ version;
- bit/HWH filenames and optionally hashes;
- `ip_dict` evidence for `mac_axi_lite_0`;
- full smoke-test PASS line;
- exact failure output if blocked.

A dry-run, successful upload, local XSIM PASS, or fake-MMIO test cannot replace
the board PASS. If the board is physically absent, preserve artifacts and resume
from this phase when external state changes; do not rebuild stable RTL merely to
appear active.
