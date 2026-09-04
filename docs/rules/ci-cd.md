# CI and CD Rules

The pipeline is divided by what each runner can objectively verify:

| Stage | Tool | Runner |
| --- | --- | --- |
| Lint | Verilator or equivalent open-source tool | GitHub-hosted |
| Simulation | Icarus, Verilator, cocotb, Python | GitHub-hosted |
| Synthesis, implementation, timing, bitstream | Vivado | `self-hosted, vivado` |
| Board validation | PYNQ runtime and physical board | `self-hosted, pynq-z1` |

Rules:

- `.github/workflows/ci.yml` runs open-source checks for pull requests and
  integration branches.
- `.github/workflows/cd.yml` is the production delivery workflow. It is
  triggered only by a published GitHub Release, not by a branch or tag push.
- `.github/cd/` contains automated deployment-and-acceptance scripts. These
  scripts may validate inputs, execute board tests non-interactively, and
  collect evidence. Example-local deployment wrappers only transfer files for
  a later human-run notebook or CLI validation.
- Production CD accepts only a non-draft, non-prerelease tag matching
  `vMAJOR.MINOR.PATCH`. The resolved release commit must be contained in
  `origin/main`.
- The Vivado job must build from the validated release tag, verify the BIT,
  HWH, provenance manifest, and implementation evidence, then attach the
  deterministic overlay files and standalone example archive to that Release.
- Board deployment must use the protected `pynq-z1-production` environment.
  Board host, user, remote root, SSH configuration, and credentials come from
  runner or environment configuration and must never be embedded in source.
- Every deployment uses a run-specific staging directory and immutable
  versioned destination. Update the board's `current` symlink only after the
  normal, non-aligned, and repeated matrix cases pass; retain the JSON evidence
  in both the workflow run and GitHub Release.
- Vivado commands must not be added to a GitHub-hosted job. A Vivado job must
  declare a trusted self-hosted runner with the required licence and tool.
- Do not enable a privileged self-hosted runner for untrusted fork pull
  requests.
- Never claim synthesis, timing closure, resource utilization, or board success
  without the corresponding Vivado report or observed board output.
- Branch build artifacts are uploaded as temporary CI artifacts. Release
  bitstreams are attached to the published Release whose commit is on `main`;
  they are never committed.
- Without a configured self-hosted runner, synthesis and board validation stay
  blocked and must be reported as not run. Publishing a Release does not make a
  missing trusted runner or physical board pass implicitly.
