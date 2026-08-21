# CI and CD Rules

The pipeline is divided by what each runner can objectively verify:

| Stage | Tool | Runner |
| --- | --- | --- |
| Lint | Verilator or equivalent open-source tool | GitHub-hosted |
| Simulation | Icarus, Verilator, cocotb, Python | GitHub-hosted |
| Synthesis, implementation, timing, bitstream | Vivado | trusted self-hosted |
| Board validation | PYNQ runtime and physical board | trusted self-hosted or manual |

Rules:

- `.github/workflows/ci.yml` runs open-source checks for pull requests and
  integration branches.
- Vivado commands must not be added to a GitHub-hosted job. A Vivado job must
  declare a trusted self-hosted runner with the required licence and tool.
- Do not enable a privileged self-hosted runner for untrusted fork pull
  requests.
- Never claim synthesis, timing closure, resource utilization, or board success
  without the corresponding Vivado report or observed board output.
- Branch build artifacts are uploaded as temporary CI artifacts. Release
  bitstreams are attached to a tag on `main`; they are never committed.
- Without a configured self-hosted runner, synthesis and board validation stay
  manual and must be reported as not run or blocked.
