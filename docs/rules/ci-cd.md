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
- `.github/workflows/cd.yml` is the production delivery workflow. A published
  **prerelease** triggers it; a stable Release is what it produces. Publication
  comes first and validation follows, but the stable Release exists only once
  the board has accepted the packages attached to it.
- Publish the release candidate as a prerelease on a `main` commit. CD refuses a
  draft, refuses a Release that is already stable, and refuses a tag that does
  not match `vMAJOR.MINOR.PATCH`. A stable Release must never be published by
  hand: it would carry no validation evidence, and CD will not retrofit any.
- The prerelease tag must equal the version `main` declares as
  `changelog/vMAJOR.MINOR.PATCH.md`, which `.github/cd/resolve_release_version.py`
  reports. A mismatch fails before any privileged work, so the package, the
  changelog and the Release cannot drift apart. A pull request into `main` that
  declares no new version cannot merge.
- The promotion job depends on every validation job. Only it may hold
  `contents: write`; it attaches the packages, checksums and board evidence to
  the prerelease and only then clears the prerelease flag. A failed run leaves
  the candidate as a prerelease, which is the signal that it was not accepted.
  Tags are never moved or deleted; a corrected release is a new version.
- Converting a prerelease to a stable Release raises the `released` event, not
  `published`, so the promotion at the end of a run cannot retrigger CD.
- `.github/cd/` contains automated deployment-and-acceptance scripts. These
  scripts may validate inputs, execute board tests non-interactively, and
  collect evidence. Example-local deployment wrappers only transfer files for
  a later human-run notebook or CLI validation.
- The Vivado job must build from the validated release tag, verify the BIT,
  HWH, provenance manifest, and implementation evidence, and assemble the
  standalone matrix and ResNet-18 packages with a checksum file.
- The ResNet-18 package is what the board runs and what the Release publishes.
  The board proves the extracted tree against the published archive digest and
  the package manifest before programming the overlay, runs the acceptance
  corpus, then classifies the pinned photograph, and promotes the deployment
  directory only after readable evidence exists.
- Board deployment must use the protected `pynq-z1-production` environment, and
  promotion must use the `pynq-z1-release` environment whose reviewers own the
  decision to make the candidate stable. Board host, user, remote root, SSH
  configuration, and credentials come from runner or environment configuration
  and must never be embedded in source.
- Every deployment uses a run-specific staging directory and immutable
  versioned destination. Update the board's active selection only after the
  required cases pass; retain the JSON evidence in both the workflow run and
  the Release.
- Vivado commands must not be added to a GitHub-hosted job. A Vivado job must
  declare a trusted self-hosted runner with the required licence and tool.
- Do not enable a privileged self-hosted runner for untrusted fork pull
  requests.
- Never claim synthesis, timing closure, resource utilization, or board success
  without the corresponding Vivado report or observed board output.
- Branch build artifacts are uploaded as temporary CI artifacts. Release
  bitstreams are attached to the Release whose commit is on `main`; they are
  never committed.
- Without a configured self-hosted runner, synthesis and board validation stay
  blocked and must be reported as not run. Publishing a prerelease does not make
  a missing trusted runner or physical board pass implicitly, and without them
  the candidate stays a prerelease.

## Recovering a failed run

A failed run leaves the candidate as a prerelease with whatever evidence the run
produced retained on the run itself. Fix the cause on `dev`, promote again, and
publish a prerelease for the next version; the failed candidate stays a
prerelease as the record that it was not accepted. Do not edit a prerelease to
stable by hand to "finish" a run.

## Repository configuration this workflow assumes

These are GitHub settings, not files, and a person applies them:

- Branch protection on `main` requiring the `lint-and-simulate` and
  `release-readiness` checks, and prohibiting direct pushes.
- A `pynq-z1-production` environment scoped to the board runner.
- A `pynq-z1-release` environment with required reviewers, which is where the
  promotion decision is taken.
