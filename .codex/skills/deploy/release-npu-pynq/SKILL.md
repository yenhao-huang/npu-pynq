---
name: release-npu-pynq
description: Take a validated `dev` to a published GitHub Release for this repository. Use when the user asks to cut, prepare, run, or publish a release, to promote `dev` to `main`, to start CD, to bring the self-hosted runners online, or asks what state a release is in or why a CD job is queued. Covers the runners, the release branch, the CD validation run, the draft Release, the promotion pull request, and the publish step.
---

# Release npu-pynq

One release moves through four states. Each one is visible in the repository, so
you can always answer "where is this release" by looking, not by remembering.

```text
runners online  ──▶  release/vX.Y.Z  ──▶  draft Release  ──▶  main  ──▶  published
   (step 2)           (branch)            (cd.yml)          (PR)    (release-publish.yml)
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
python3 .github/cd/resolve_release_version.py
```

If the version needs to change, rename the changelog file; nothing else selects
the version. Default to the next patch unless the release decision says
otherwise.

## 2. Bring the runners online

Do this yourself, before pushing. Two CD jobs need this host's runners, and a
release branch pushed without them leaves a job queued for a day:

| Job | Labels |
| --- | --- |
| `build-overlay` | `self-hosted`, `vivado` |
| `board-validation` | `self-hosted`, `pynq-z1` |

Read `references/rules/env.md`, then:

```powershell
& .codex/skills/deploy/release-npu-pynq/references/scripts/start_cd_runners.ps1 -WhatIf
& .codex/skills/deploy/release-npu-pynq/references/scripts/start_cd_runners.ps1
```

The script starts every `actions.runner.*` service on this host and waits for
the repository to report them online with both labels. It starts services only;
registering a runner is a person's one-time setup, described in
`references/runners.md`.

Stop and report if the script exits non-zero:

- No service exists — no runner is installed on this host, or it is configured
  to run interactively and needs a person to launch its `run.cmd`.
- Runners never report online — the service is running but cannot reach GitHub.
- A label is missing — a runner exists but does not offer `vivado` or `pynq-z1`.

Do not push the release branch until the runners are online. Pushing is what
starts the multi-hour privileged run, and a queued run wastes a day before
anyone notices.

Also confirm the two environments exist, because a job naming a missing
environment fails immediately: `pynq-z1-production` for the board job, and
`pynq-z1-release` with required reviewers for publishing. They are repository
settings and a person creates them.

## 3. Cut the release branch

```bash
git fetch origin dev
git checkout -b release/vX.Y.Z origin/dev
git push -u origin release/vX.Y.Z
```

The push starts `cd.yml`. Because a push event uses the workflow file from the
branch it ran on, this validates the workflow itself as well as the release;
`main` does not have to be touched first.

## 4. Watch the CD run

Its jobs, in order: `validate-source`, `host-checks`, `build-overlay` (Vivado,
self-hosted), `board-validation` (physical PYNQ-Z1), `publish-draft`.

```bash
gh run list --workflow cd.yml --branch release/vX.Y.Z
gh run watch <run-id>
```

A passing run leaves a **draft Release** holding the ResNet-18 package, the
matrix package, the overlay artifacts, the checksums and the board evidence. A
draft has no tag and is not public.

A job that sits at `queued` with `runner=-` has no runner offering its labels,
whatever the log says above that line: `Evaluating: success() / Result: true`
means only that the job is eligible to run. A started job prints `Runner name:`.
Go back to step 2, then rerun:

```bash
gh run rerun <run-id> --failed
```

Read `docs/rules/ci-cd.md` before diagnosing anything else. Common causes:

- Packaging names a missing file — the untracked ResNet-18 model workspace or
  acceptance bundle is absent from the Vivado runner.
- `board-validation` fails immediately — the `pynq-z1-production` environment
  does not exist.
- `validate-source` rejects the branch — the branch name and the declared
  version disagree.

Fix on `dev`, then rebuild the release branch from the new `dev` and push again.
The run recreates the draft, so a retry is safe.

## 5. Prepare the promotion pull request

Only after the CD run is green. Base `main`, head `release/vX.Y.Z`. The body
states the integrated `dev` commit, the issues and pull requests in the upload,
the CD run URL, and what is still blocked.

Update `changelog/vX.Y.Z.md` first: set the change-before commit to `main`'s
current head, the change-after commit to this integration, and add one row per
issue and pull request, following `docs/rules/git/changelog.md`.

Stop here and hand the pull request to a person.

## 6. Publish

After the pull request is merged with a merge commit:

```bash
gh workflow run release-publish.yml -f release_tag=vX.Y.Z
```

It refuses a draft that is missing assets, a commit that is not contained in
`main`, and a tag that already exists. It builds nothing and touches no board.
Approving the `pynq-z1-release` environment is the release decision.

## Reporting state

Say which of the four states the release is in and what the next action is, for
example: "v0.1.5 is validated; the draft holds the packages and board evidence;
the promotion pull request is open and waiting on you." Never describe
synthesis, timing or board results that no run produced.
