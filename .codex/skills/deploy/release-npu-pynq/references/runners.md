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

## How this host's runner is set up

On this repository's Windows host the runner is:

| Item | Value |
| --- | --- |
| Folder | `C:\actions-runner` |
| Registration name | `DESKTOP-54U632L` |
| Labels | `self-hosted`, `Windows`, `X64`, `vivado`, `pynq-z1` |
| Started by | Scheduled task `GitHub Actions Runner`, at logon, as the signed-in user |
| Model workspace for CD | `C:\npu-assets\resnet18\model` (or the `RESNET18_MODEL_DIR` repository variable) |

It runs as a logon task rather than a service on purpose. See "Why a logon
task" below before changing that.

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

### Why a logon task, not a service

A service would start without anyone signing in, but two things rule it out
here:

- **The Windows account has no password.** Windows refuses to let a service
  log on as a passwordless account, by default security policy. Leaving the
  password field empty in the service's Log On tab fails. Relaxing that policy
  weakens the whole machine; do not.
- **The jobs need the user's environment.** A service falls back to
  `NT AUTHORITY\NetworkService`, whose PATH does not find `vivado`, `python` or
  the Microsoft Store `pwsh`, and which has neither the Vivado licence nor the
  SSH keys that reach the PYNQ-Z1.

A logon scheduled task avoids both: it runs `run.cmd` as the signed-in user, needs
no password, and survives reboots. Set it up once, from an elevated PowerShell:

```powershell
& .codex/skills/deploy/release-npu-pynq/references/scripts/register_runner_task.ps1 `
    -RunnerRoot C:\actions-runner -WhatIf
& .codex/skills/deploy/release-npu-pynq/references/scripts/register_runner_task.ps1 `
    -RunnerRoot C:\actions-runner
```

It stops and disables any `actions.runner.*` service first, because two
listeners for one registration fight over the session. The cost is that the
runner is online only while the user is signed in; the machine must not sit at
the lock screen after a reboot with nobody logged on.

If the account does have a password and a service is wanted instead, create it
from the service's properties in `services.msc` (Log On tab, "This account",
`.\<user>`), not with `sc.exe`: the GUI also grants "Log on as a service",
`sc.exe` does not.

### Creating the service by hand

`config.cmd` sometimes stops right after `Granting file permissions` without
installing the service or printing why. The service it would create is just
`bin\RunnerService.exe` under a fixed name, so it can be made directly:

```powershell
cd C:\actions-runner
$agent = (Get-Content .\.runner -Raw | ConvertFrom-Json).agentName
$svc   = "actions.runner.yenhao-huang-npu-pynq.$agent"
sc.exe create $svc binPath= "`"C:\actions-runner\bin\RunnerService.exe`"" start= auto obj= "NT AUTHORITY\NetworkService"
Set-Content -Path .\.service -Value $svc -NoNewline   # lets config.cmd remove find it
sc.exe start $svc
```

Write `sc.exe`, never `sc`: in PowerShell `sc` is `Set-Content`. And `binPath= `
needs the space after the equals sign.

### A runner under C:\Windows fails as a service

This host's runner was first unpacked into `C:\Windows\System32\actions-runner`.
Run by hand it worked; as a service it failed with **error 1053** ("the service
did not respond in time"). Moving it fixed that immediately: the folder and its
registration copy across unchanged, and no re-registration is needed.

```powershell
Copy-Item C:\Windows\System32\actions-runner C:\actions-runner -Recurse
```

Do not leave both copies startable. They share one registration, and two
listeners for it fight over the session. Delete the old copy once the new one
is online.

### A runner under System32 should be relocated

`C:\Windows\System32\actions-runner` works, but the runner writes its
`_work/` tree beside itself, and a Vivado build puts gigabytes of output there —
inside a protected system directory that needs elevation for ordinary file
operations. Move it to something like `C:\actions-runner`: remove the old
registration (`.\config.cmd remove`), then register again from the new folder.
That is a person's setup task, not something this skill does.

## Symptoms seen on this host, and what they meant

| Symptom | Meaning |
| --- | --- |
| Job `queued`, log says `Waiting for a runner` | No online runner offers every requested label |
| `Get-Service 'actions.runner.*'` is empty | The runner was configured without the service option, not missing |
| `config.cmd` says "already configured" but the Runners page lists none | A stale local registration; `config.cmd remove --local`, then register again |
| `config.cmd remove` asks for a token | That is the removal token, from the runner's Remove button; not needed for a stale one |
| Service start: error 1053 | The runner lives under `C:\Windows`; move it |
| Checkout: "A branch or tag with the name 'vX.Y.Z' could not be found" | The workflow asked for a tag; CD never creates one (fixed: it checks out the commit) |

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
