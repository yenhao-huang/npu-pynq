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

## Interactive runners

A runner configured with `run.cmd` rather than as a service has no service
entry. `Get-Service -Name 'actions.runner.*'` returns nothing, which looks the
same as no runner at all. Find it:

```powershell
Get-ChildItem -Path C:\, $HOME -Filter run.cmd -Recurse -Depth 4 `
    -ErrorAction SilentlyContinue |
  Where-Object { $_.DirectoryName -match 'actions-runner' } |
  Select-Object -ExpandProperty FullName
```

Then start it in its own window:

```powershell
cd <the folder printed above>
.\run.cmd
```

It prints `Listening for Jobs` and stays online only while that window is open.
`start_cd_runners.ps1` does this discovery and launch for you.

### Prefer a service

An interactive runner has to be started by hand after every reboot, and a closed
window silently takes CD offline. Convert it once:

```powershell
cd <runner folder>
.\svc.cmd install
.\svc.cmd start
```

### A runner under System32 should be relocated

`C:\Windows\System32\actions-runner` works, but the runner writes its
`_work/` tree beside itself, and a Vivado build puts gigabytes of output there —
inside a protected system directory that needs elevation for ordinary file
operations. Move it to something like `C:\actions-runner`: remove the old
registration (`.\config.cmd remove`), then register again from the new folder.
That is a person's setup task, not something this skill does.

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
