# PYNQ Development State

Run ID: phase1b-dma-vertical-slice-20260822
Instance: `.codex/skills/custom/ic_design/pynq-dev`
Started: 2026-08-22T09:56:01+08:00
Scope: Integrate the ABI v1 AXI4-Lite control plane, AXI4-Stream DMA data
plane, reproducible PYNQ-Z1 Vivado Tcl, host runtime, and physical-board matrix
vertical slice for Issue #4.
OpenSpec change: `implement-phase1b-dma-vertical-slice`

Last updated: 2026-08-22T22:54:00+08:00

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | Issue `#4` objective, scope, acceptance criteria, and dependency on Issue `#3` were read from GitHub. The issue is assigned to `yenhao-huang`; claim comment records agent `a`. Branch `npu/issue4-a` and dedicated worktree were created from `dev`, then fast-forwarded to Phase 1A commit `8817843`. | Scope includes RTL/interface, testbench, constraints/Tcl, PYNQ runtime, overlay metadata, DMA, and physical-board evidence. Numeric arithmetic remains the Phase 0/1A contract. No exporter, ResNet, Transformer, or `docs/human/` changes. |
| 1. Read Context and Rules | completed | Repository rules, `pynq-dev` references, Phase 0 ABI/model, Phase 1A RTL, test Makefile, and relevant prior PYNQ AXI/Tcl/board references were read. | Required gates: Python/runtime tests, RTL lint/simulation, strict OpenSpec, Vivado build/utilization/timing/address inspection, matching bit/HWH, and board PASS. Lower-layer evidence cannot replace board evidence. |
| 2. Prepare OpenSpec Change | completed | OpenSpec CLI 1.10.0 created `implement-phase1b-dma-vertical-slice`; proposal, six capability delta specs, design, and 18 tasks are defined. Strict validation passes. | Frozen contracts cover 2x2xK<=256 physical jobs, two 8-bit input frames, 32-bit output, AXI-Lite ABI, DMA sequencing, fixed addresses, 100-MHz timing gate, HWH provenance, exact DSP mapping, PYNQ 3.1 completion/recovery compatibility, and exact board PASS. |
| 3. Implement | completed | ABI v1 AXI-Lite registers, bounded stream controller, 2x2 systolic integration, runtime, provenance verifier, board smoke entry point, repository-relative Vivado flow, and seven self-checking RTL testbenches are implemented. | Controller LUTRAM reads and PE products are registered with explicit drain cycles. A source-controlled PE synthesis gate confirms exactly one DSP48 per PE. Generated artifacts and build products remain ignored. |
| 4. Validate | in_progress | `python -m unittest discover -s src/test/tests -v`: 58/58 PASS. Seven direct `iverilog -Wall` simulations PASS. Strict OpenSpec validation PASS. PE OOC synthesis: exactly one DSP48, 0 warnings. Clean Vivado 2026.1 production build from current PR head `7dd50b3f4019e9bdbe52733e1465d96cb1e69fa6`: BIT/HWH metadata/provenance PASS; WNS 0.345 ns; 0 setup-failing paths; 7,697/7,697 routable nets fully routed; 0 routing errors; 0 DRC errors. Manifest records BIT SHA-256 `275c837a2db064e9dc5fc806cabf7acd5f6a33e09b452fe6579ad884c64ab53a` and HWH SHA-256 `1dd6388726c01e88bb9db2f6608d29bd5f5d651b4158c2ad62063085d8139064`. Board `192.168.2.99` passes ICMP/TCP22 and runs PynqLinux 3.0, Python 3.10.4, PYNQ 3.1.1, and NumPy 1.21.5. Authorized upload to `/home/xilinx/jupyter_notebooks/npu_matrix` PASS; remote hashes and provenance match. Root staged checks PASS: overlay/IP discovery in 12.222 s, ABI/runtime limits 2x2x256, and aligned XRT buffers. A one-instance diagnostic compatibility patch completed the real DMA/MMIO/PL transaction with C=`[[636,-891],[-19,29]]`, DONE status `2`, error `0`, and stable cycles `72683`. | The unmodified formal smoke remains FAILED. PYNQ 3.1.1 leaves `channel.transferred=0` until `channel.wait()` runs, so runtime raises `DMAError: A MM2S transferred 0 bytes, expected 4` after polling idle. Recovery then hangs in `DMAChannel.stop()` and masks the original error. Diagnostic success is not the required exact formal PASS; task 5.2 stays incomplete until runtime compatibility and bounded recovery are fixed, tested, deployed, and the unmodified smoke command passes. GNU Make/Verilator and current-head CI also remain unresolved. |
| 5. Handoff | in_progress | PR #17 history is consolidated into one PR #15-compliant `feat` commit atop the latest `origin/dev`. A clean current-head overlay now exists outside Git and the board is reachable. Draft PR `#17` targets `dev`; Issue #4 evidence comment `5378326981` records the earlier completed and blocked gates. | Explicit upload approval, physical-board PASS, and GitHub Actions for the rewritten head remain required. Keep the PR draft; do not merge or close Issue #4. Never commit bitstreams, HWH, reports, caches, or generated Vivado output. |

## Current compatibility update

- PYNQ 3.1 post-idle DMA bookkeeping and active-channel recovery guards are
  implemented with regression coverage.
- `python -m unittest discover -s src/test/tests -v`: 59/59 PASS.
- Strict OpenSpec validation: PASS with 17/18 tasks complete.
- Repository `make -C src/test lint sim`: BLOCKED because GNU Make and Verilator
  are unavailable on this host; task 6.2 remains incomplete.
- Clean Vivado 2026.1 build from source commit
  `7d3018fd143087237e983f0b981e038962f00203`: PASS with WNS 0.345 ns, zero
  setup-failing paths, 7,697/7,697 fully routed nets, zero routing errors, and
  zero DRC errors.
- Current-source BIT SHA-256
  `d03c3c3983d0c49a3d5fc0af026d6931a53e4f4bdbf40914123b1981e9bd1df4`
  and HWH SHA-256
  `ade8502a905d4ec2005a05f3ea3a27ba9acaad4e702c6ea4bd2aa7b3d1c2142a`
  match on host and board.
- Unmodified formal board smoke: `PASS: NPU DMA matrix vertical slice`; result
  `[[636,-891],[-19,29]]`, status 2, error 0, stable cycles 93983, accelerator
  `0x43C00000`, DMA `0x40400000`, and limits 2x2x256. Task 5.2 is complete.
