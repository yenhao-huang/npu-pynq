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
- `.github/workflows/cd.yml` is the production delivery workflow. It runs on a
  push to a `release/vMAJOR.MINOR.PATCH` branch cut from `dev`. A push event uses
  the workflow file from the branch it ran on, so a release validates the
  delivery path as well as itself, before anything reaches `main`.
- The release branch name must equal the version the source declares as
  `changelog/vMAJOR.MINOR.PATCH.md`, which `.github/cd/resolve_release_version.py`
  reports, and that version must not already be tagged. A pull request into
  `main` that declares no new version cannot merge.
- A passing CD run publishes a **draft** Release holding the standalone
  ResNet-18 package, the standalone matrix package, the deterministic overlay
  artifacts, one checksum file covering them, and the board evidence from that
  same run. A draft has no tag and is not public. A failed run publishes
  nothing, and rerunning recreates the draft.
- CD never creates a tag. `.github/workflows/release-publish.yml` publishes the
  draft, and it is the only thing that does. It is dispatched by hand, builds
  nothing, touches no board, and refuses a draft that is not a draft, is missing
  a validated asset, targets a commit not contained in `main`, or whose tag
  already exists.
- Merge the promotion pull request with a merge commit. A squash produces a new
  commit, the validated commit is then absent from `main`, and publishing
  refuses it.
- Tags are never moved or deleted. A corrected release is a new version.
- `.github/cd/` contains automated deployment-and-acceptance scripts. These
  scripts may validate inputs, execute board tests non-interactively, and
  collect evidence. Example-local deployment wrappers only transfer files for
  a later human-run notebook or CLI validation.
- The Vivado job must build from the release branch commit, verify the BIT, HWH,
  provenance manifest, and implementation evidence, and assemble the standalone
  matrix and ResNet-18 packages with a checksum file.
- The ResNet-18 package is what the board runs and what the Release publishes.
  The board proves the extracted tree against the published archive digest and
  the package manifest before programming the overlay, runs the acceptance
  corpus, then classifies the pinned photograph, and promotes the deployment
  directory only after readable evidence exists.
- Board deployment must use the protected `pynq-z1-production` environment, and
  publishing must use the `pynq-z1-release` environment whose reviewers own the
  release decision. Board host, user, remote root, SSH configuration, and
  credentials come from runner or environment configuration and must never be
  embedded in source.
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
  blocked and must be reported as not run. Pushing a release branch does not make
  a missing trusted runner or physical board pass implicitly, and without them no
  draft is produced at all.

## The four states of a release

`.codex/skills/deploy/release-npu-pynq/` is the procedure. The states are:

1. `release/vMAJOR.MINOR.PATCH` cut from `dev` and pushed.
2. A draft Release, produced by a passing CD run. Validated, untagged, private.
3. A promotion pull request merged into `main` with a merge commit.
4. A published Release, created by `release-publish.yml` once the validated
   commit is contained in `main`.

## Recovering a failed run

Nothing is tagged or public, so recovery is ordinary work: fix the cause on
`dev`, rebuild the release branch from the new `dev`, and push it again. The run
recreates the draft. Do not publish a draft by hand to finish a run.

## Repository configuration this workflow assumes

These are GitHub settings, not files, and a person applies them:

- Branch protection on `main` requiring the `lint-and-simulate` and
  `release-readiness` checks, and prohibiting direct pushes.
- A `pynq-z1-production` environment scoped to the board runner.
- A `pynq-z1-release` environment with required reviewers, which is where the
  release decision is taken.
