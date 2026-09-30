# Self-hosted runners for continuous deployment

Two CD jobs cannot run on GitHub's hosted machines:

| Job | Labels | What it needs |
| --- | --- | --- |
| `build-overlay` | `self-hosted`, `vivado` | A licensed Vivado install |
| `board-validation` | `self-hosted`, `pynq-z1` | A physical PYNQ-Z1 on the direct link |

Both may live on the same Windows host, registered as one runner carrying all
three labels, or as two runners.

## What a queued job means

A job that stays `queued` with `runner=-` has no runner offering its labels. The
log line `Waiting for a runner to pick up this job...` is that state, and
`Evaluating: success() / Result: true` above it only means the job is eligible —
not that it started. A run that has genuinely started prints a `Runner name:`
line.

GitHub expires a queued job after about 24 hours. Nothing is tagged or published
in the meantime, so a queued run is safe to leave and safe to rerun.

## Checking state

```powershell
gh api repos/:owner/:repo/actions/runners `
  --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
```

`status` must be `online`. A runner that is `offline`, or `online` without the
label a job requests, will not pick that job up.

Without repository administration access this call returns 403. Fall back to
`Get-Service -Name 'actions.runner.*'` on the host, which shows whether the
service is running locally even when the API is unreadable.

## Starting them

`references/scripts/start_cd_runners.ps1` starts every runner service on this
host and waits for the API to report them online. Run it before pushing a
release branch:

```powershell
& .codex/skills/deploy/release-npu-pynq/references/scripts/start_cd_runners.ps1
```

Pass `-WhatIf` first to see which services it would start. The script starts
services only; it never registers, reconfigures, or removes a runner, and it
never handles a registration token.

A runner configured to run interactively rather than as a service has no
service entry. The script reports that and names the `run.cmd` it found, which a
person launches in its own window; it stays online only while that window is
open.

## When no runner exists yet

Registering one is a person's setup task, done once per host:

1. Repository → Settings → Actions → Runners → New self-hosted runner.
2. Follow the shown download and `config.cmd` steps on the Vivado host.
3. Add the labels the jobs request: `vivado`, and `pynq-z1` if the same host
   also reaches the board.
4. Install it as a service (`svc.cmd install` then `svc.cmd start`) so it
   survives a reboot and this skill can start it.

Do not register a runner on an untrusted machine, and do not enable a
privileged runner for fork pull requests. See `docs/rules/ci-cd.md`.
