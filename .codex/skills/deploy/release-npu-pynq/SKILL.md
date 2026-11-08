---
name: release-npu-pynq
description: Take a validated `dev` to a published GitHub Release for this repository. Use when the user asks to cut, prepare, run, or publish a release, to promote `dev` to `main`, to start CD, to bring the self-hosted runners online, or asks what state a release is in or why a CD job is queued. Covers the runners, the release branch, the release.yml validation run, the draft Release, the promotion pull request, and the publish step.
---

# Release npu-pynq

One release moves through four states. Each one is visible in the repository, so
you can always answer "where is this release" by looking, not by remembering.

```text
runners online  ──▶  release/vX.Y.Z  ──▶  draft Release  ──▶  main  ──▶  published
   (step 2)           (branch)          (release.yml)      (PR)     (publish.yml)
```

Nothing is tagged until the last step, and the tag names a commit that is both
validated by hardware and merged into `main`.

## Rules that decide everything else

- The version is declared in the repository as `changelog/vMAJOR.MINOR.PATCH.md`,
  not chosen in a workflow input. The release branch name must match it.
- `dev` → `main` is a person's decision. Prepare the pull request; never merge it.
- Merge the promotion pull request **with a merge commit**. A squash creates a
  new commit, the validated commit is then absent from `main`, and publishing
  refuses it.
- A tag is never moved or deleted. A corrected release is a new version.
- Never publish a Release by hand, and never clear a draft by hand. The publish
  workflow is what checks that the draft was validated and merged.
- Bring the runners online before pushing a release branch. That is this skill's
  job, not something to hand back to the user.

## 1. Decide the version

Read `changelog/` on `dev`. The release is the highest `vMAJOR.MINOR.PATCH.md`
present. Confirm the exact resolution:

```bash
python3 .github/release/version.py
```

If the version needs to change, rename the changelog file; nothing else selects
the version. Default to the next patch unless the release decision says
otherwise. When `main` holds commits `dev` lacks (an earlier promotion), merge
`origin/main` into the release branch first and keep the top-level
`changelog/` files.

Before the final CD run, update the README Quick Start download links and its
`scp` and `tar` commands to `pynqo-resnet18-vX.Y.Z.tar.gz` and
`pynqo-smollm-vX.Y.Z.tar.gz` for this release. Include
that change in the release source before validation, preferably on `dev` before
cutting the release branch. The new download URL becomes live when the release
is published. GitHub's `/releases/latest/download/` URL still requires an exact
asset filename; `<latest>` is a version placeholder, not a literal filename.

## 2. Bring the runners online

Do this yourself, before pushing. Three jobs need this host's runners, and a
release branch pushed without them leaves a job queued for a day:

| Job | Labels |
| --- | --- |
| `check` | `self-hosted`, `vivado`, `pynq-z1` |
| `build-bitstream` | `self-hosted`, `vivado` |
| `test-on-board` | `self-hosted`, `pynq-z1` |

Read `references/rules/env.md`, then:

```powershell
& .codex/skills/deploy/release-npu-pynq/references/scripts/start_cd_runners.ps1 -WhatIf
& .codex/skills/deploy/release-npu-pynq/references/scripts/start_cd_runners.ps1
```

The script starts every `actions.runner.*` service on this host and waits for
the repository to report them online with both labels. It starts services only;
registering a runner is a person's one-time setup, described in
`references/runners.md`.

The script handles every shape a runner can take here, in this order: the
logon scheduled task `GitHub Actions Runner` (how this host is set up), an
enabled `actions.runner.*` service, or a bare `run.cmd` launched in its own
window. A bare `run.cmd` is online only while that window stays open.

If no task exists yet, `references/scripts/register_runner_task.ps1` is the
one-time setup. It is the right choice when the Windows account has no password,
which rules out a service, and it gives the jobs the user's own PATH, Vivado
licence and SSH keys. `references/runners.md` explains why, and lists what each
setup symptom on this host turned out to mean.

Stop and report if the script exits non-zero:

- Nothing found — no runner is installed on this host. `references/runners.md`
  has the one-time registration.
- Runners never report online — the runner started but cannot reach GitHub.
- A label is missing — a runner exists but does not offer `vivado` or `pynq-z1`.
  Add it on the repository's Runners page; re-registration is not needed.

The board jobs run `bash` steps, which on this Windows host is Git Bash: it
needs `ssh`, `scp` and `tar`, and the same SSH key as the Windows account.

Also confirm the two environments exist, because a job naming a missing
environment fails immediately: `pynq-z1-production` for the board jobs, and
`pynq-z1-release` with required reviewers for publishing. They are repository
settings and a person creates them. `PYNQ_BOARD_HOST` and `PYNQ_BOARD_USER`
are optional variables there (default `xilinx@192.168.2.99`).

## 3. Cut the release branch

```bash
git fetch origin dev
git checkout -b release/vX.Y.Z origin/dev
git push -u origin release/vX.Y.Z
```

The push starts `release.yml`. Because a push event uses the workflow file from
the branch it ran on, this validates the workflow itself as well as the
release; `main` does not have to be touched first.

## 4. Watch the release run

```text
check ─┬─ export (resnet18 | smollm)
       └─ build-bitstream
                └─ test-on-board (resnet18 | smollm)
                         └─ draft-release
```

- `check` (about a minute): the version, the runner's tools, the board.
- `export`, one job per entry of `.github/release/models.json`, on a hosted
  machine: the model guide's fetch and export, then its run command on the host
  simulator. That host result is the board's reference.
- `build-bitstream`: Vivado, about an hour, at the same time as `export`.
- `test-on-board`, one model at a time: `src/deploy/deploy.sh` with `ARCHIVE=`
  writes `pynqo-<model>-vX.Y.Z.tar.gz`, `.github/release/board_check.sh` sends
  it to `~/jupyter_notebooks/<model>` and runs the guide's command there, and
  the result must match the host: the same top-1 class for ResNet-18, the same
  token IDs for SmolLM. ResNet-18 takes about half an hour on the board.
- `draft-release`: the **draft Release** with the two archives, the board
  results, `npu_matrix.bit/.hwh/.manifest.json`, `bitstream-report.txt` and one
  `SHA256SUMS`. A draft has no tag and is not public.

A run never redoes Vivado that already succeeded for this release: if an earlier
run on the same `release/vX.Y.Z` branch passed `build-bitstream` and `src/hw` is
unchanged since its commit, that bitstream is reused (the job shows a
`bitstream reused` notice). So after a board or packaging fix, push the fix to
the release branch and expect the run to reach the board in minutes.

```bash
gh run list --workflow release.yml --branch release/vX.Y.Z
gh run watch <run-id>
```

A job that sits at `queued` with `runner=-` has no runner offering its labels,
whatever the log says above that line: `Evaluating: success() / Result: true`
means only that the job is eligible to run. A started job prints `Runner name:`.
Go back to step 2, then rerun:

```bash
gh run rerun <run-id> --failed
```

Read `docs/rules/ci-cd.md` before diagnosing anything else. Common causes:

- `check` rejects the branch — the branch name and the declared version
  disagree, or the board is unreachable or lacks the sudo rule
  (`references/runners.md`).
- `export` fails — a pinned download changed or is unreachable, or the model no
  longer runs on the host simulator. The release must not ship in either case.
- `test-on-board` fails at the comparison — the board and the host simulator
  disagree; that is a hardware or runtime defect, not a flaky step.

There is no separate rehearsal: the release run is the validation. When a run
is certain to fail (a defect is already known), cancel it rather than let it
finish: the concurrency group queues a new run behind it instead of replacing
it.

Fix on `dev`, then bring the release branch up to the new `dev` and push again.
Prefer a fast-forward (or a merge) over a force-push: an earlier bitstream is
reused only from an ancestor of the new release commit, so a rewritten branch
pays for Vivado again. The run recreates the draft, so a retry is safe.

## 5. Prepare the promotion pull request

Only after the release run is green. Base `main`, head `release/vX.Y.Z`. The
body states the integrated `dev` commit, the issues and pull requests in the
upload, the release run URL, and what is still blocked.

Update `changelog/vX.Y.Z.md` first: set the change-before commit to `main`'s
current head, the change-after commit to this integration, and add one row per
issue and pull request, following `docs/rules/git/changelog.md`.

Stop here and hand the pull request to a person.

## 6. Publish

After the pull request is merged with a merge commit:

```bash
gh workflow run publish.yml -f release_tag=vX.Y.Z
```

It refuses a draft that is missing assets, a commit that is not contained in
`main`, and a tag that already exists. It builds nothing and touches no board.
Approving the `pynq-z1-release` environment is the release decision.

After publication, verify that the README Quick Start link resolves to the
published `pynqo-resnet18-vX.Y.Z.tar.gz` and `pynqo-smollm-vX.Y.Z.tar.gz`
assets and that its `scp` and `tar` commands use the same filenames. If the README update was missed before CD,
prepare a documentation-only follow-up PR to `main`; do not push a docs-only
commit to the release branch just to fix the link, because that starts CD again.

## Reporting state

Say which of the four states the release is in and what the next action is, for
example: "v1.0.8 is validated; the draft holds the archives and board results;
the promotion pull request is open and waiting on you." Never describe
synthesis, timing or board results that no run produced.
