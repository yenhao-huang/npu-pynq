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
- `.github/workflows/release.yml` is the production delivery workflow. It runs
  on a push to a `release/vMAJOR.MINOR.PATCH` branch cut from `dev`, unless the
  push only changes documentation (`*.md`, `docs/`, `.codex/`); a README fix
  on a release branch does not start the run again. A push
  event uses the workflow file from the branch it ran on, so a release validates
  the delivery path as well as itself, before anything reaches `main`:

  ```text
  check ─┬─ export (model)
         └─ build-bitstream
                  └─ test-on-board (model)
                           └─ draft-release
  ```

- The release branch name must equal the version the source declares as
  `changelog/vMAJOR.MINOR.PATCH.md`, which `.github/release/version.py`
  reports, and that version must not already be tagged. `check` enforces both,
  and checks the runner's tools and the board, before anything long starts. A
  pull request into `main` that declares no new version cannot merge.
- The models a release ships are the entries of `.github/release/models.json`:
  how to fetch the model, its export config, the files `src/deploy/deploy.sh`
  adds, and the example guide's run command. A release model goes the guide's
  way and no other: `export` fetches and exports it on a hosted machine and runs
  the command on the host simulator; `test-on-board` packages it with
  `deploy.sh` (`ARCHIVE=` writes the archive instead of sending it), and
  `.github/release/board_check.sh` runs that archive's command on the board,
  detached and polled so a dropped ssh session does not fail the run. The board
  must agree with the host simulator: the same top-1 class for a classifier,
  the same token IDs for an LLM, with NPU jobs counted.
- The archive the board ran is the release asset. A passing run publishes a
  **draft** Release holding `pynqo-<model>-<tag>.tar.gz` and `board-<model>.json`
  for each model, `npu_matrix.bit/.hwh/.manifest.json`, `bitstream-report.txt`,
  and one `SHA256SUMS` covering them. A draft has no tag and is not public. A
  failed run publishes nothing, and rerunning recreates the draft.
- The release never creates a tag. `.github/workflows/publish.yml` publishes
  the draft, and it is the only thing that does. It is dispatched by hand,
  builds nothing, touches no board, and refuses a draft that is not a draft, is
  missing a validated asset, targets a commit not contained in `main`, or whose
  tag already exists.
- Merge the promotion pull request with a merge commit. A squash produces a new
  commit, the validated commit is then absent from `main`, and publishing
  refuses it.
- Tags are never moved or deleted. A corrected release is a new version.
- `build-bitstream` builds from the release commit and verifies the BIT, HWH,
  provenance manifest and implementation evidence. Every checkout after `check`
  names the release commit, never the release tag. It reuses the
  `bitstream-<tag>` artifact of the newest run on the same release branch whose
  `build-bitstream` succeeded, but only if it verifies, its commit is an
  ancestor of the release commit, and `src/hw` and
  `src/vendors/npu_pynq/driver/overlay/` are identical between the two.
  Otherwise Vivado runs.
- Board deployment must use the protected `pynq-z1-production` environment, and
  publishing must use the `pynq-z1-release` environment whose reviewers own the
  release decision. Board host and user (`PYNQ_BOARD_HOST`, `PYNQ_BOARD_USER`),
  SSH configuration, and credentials come from runner or environment
  configuration and must never be embedded in source.
- A release model replaces `~/jupyter_notebooks/<model>` on the board, the same
  place the example guide deploys to. Its board output is kept in both the
  workflow run and the Release.
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
- Bring the runners online before pushing a release branch. A job whose labels no
  online runner offers stays queued for about a day and then expires. The
  `release-npu-pynq` skill under `.codex/skills/deploy/` starts this host's
  runner services and confirms their labels; registering a runner is a person's
  one-time setup.

## The four states of a release

`.codex/skills/deploy/release-npu-pynq/` is the procedure. The states are:

1. `release/vMAJOR.MINOR.PATCH` cut from `dev` and pushed.
2. A draft Release, produced by a passing `release.yml` run. Validated, untagged, private.
3. A promotion pull request merged into `main` with a merge commit.
4. A published Release, created by `publish.yml` once the validated
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
