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
  integration branches, and carries the `release-readiness` gate for pull
  requests into `main`.
- `.github/workflows/cd.yml` is the production delivery workflow. It is
  triggered by a push to `main` or an explicit manual dispatch, never by a
  published Release, a tag push, or any other branch. The GitHub Release is the
  last step of a passing run, not its trigger.
- The release version is a human decision recorded on `main` as
  `changelog/vMAJOR.MINOR.PATCH.md`. `.github/cd/resolve_release_version.py`
  reads the highest declared version; when that tag already exists the run
  publishes nothing and succeeds. A pull request into `main` that declares no
  new version cannot merge.
- Publication depends on every validation job. Only the publishing job may hold
  `contents: write`; it creates the annotated tag on the validated commit and
  refuses a tag that already exists or a commit no longer contained in
  `origin/main`. A failed run leaves no tag and no Release. Tags are never
  moved or deleted; a corrected release is a new version.
- `.github/cd/` contains automated deployment-and-acceptance scripts. These
  scripts may validate inputs, execute board tests non-interactively, and
  collect evidence. Example-local deployment wrappers only transfer files for
  a later human-run notebook or CLI validation.
- The Vivado job must build from the validated `main` commit, verify the BIT,
  HWH, provenance manifest, and implementation evidence, and assemble the
  standalone matrix and ResNet-18 packages with a checksum file. Nothing is
  attached to a Release before the board has accepted those same packages.
- The ResNet-18 package is what the board runs and what the Release publishes.
  The board proves the extracted tree against the published archive digest and
  the package manifest before programming the overlay, runs the acceptance
  corpus, then classifies the pinned photograph, and promotes the deployment
  directory only after readable evidence exists.
- Board deployment must use the protected `pynq-z1-production` environment, and
  publication must use the `pynq-z1-release` environment whose reviewers own
  the release decision. Board host, user, remote root, SSH configuration, and
  credentials come from runner or environment configuration and must never be
  embedded in source.
- Every deployment uses a run-specific staging directory and immutable
  versioned destination. Update the board's active selection only after the
  required cases pass; retain the JSON evidence in both the workflow run and
  GitHub Release.
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
  blocked and must be reported as not run. A push to `main` does not make a
  missing trusted runner or physical board pass implicitly, and without them no
  Release is published at all.

## Repository configuration this workflow assumes

These are GitHub settings, not files, and a person applies them:

- Branch protection on `main` requiring the `lint-and-simulate` and
  `release-readiness` checks, and prohibiting direct pushes.
- A `pynq-z1-production` environment scoped to the board runner.
- A `pynq-z1-release` environment with required reviewers, which is where the
  publication decision is taken.
