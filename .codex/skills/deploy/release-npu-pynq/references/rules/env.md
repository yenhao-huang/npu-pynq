# Environment Rules

Primary language: PowerShell
Runtime version: Windows PowerShell 5.1 or PowerShell 7+
Package manager: none
Frameworks: none
Service manager: Windows Services (`Get-Service`, `Start-Service`)
Required services: two GitHub Actions self-hosted runners; `gh` authenticated
against this repository

- The release runs on the Windows development host. It is the machine that has
  Vivado and the direct Ethernet link to the PYNQ-Z1.
- Runner services are named `actions.runner.<owner>-<repo>.<runner-name>`.
  Discover them with `Get-Service -Name 'actions.runner.*'`; never hardcode a
  runner name.
- The Vivado runner must carry the labels `self-hosted` and `vivado`. The board
  runner must carry `self-hosted` and `pynq-z1`. One host may provide both.
- A runner registration token is short-lived and host-specific. Never store one
  in this repository, and never echo one.
- `gh auth status` must show a token with `repo` scope. Reading runner state
  needs administration access to the repository.
- The PYNQ-Z1 is reached over the direct Ethernet link, commonly
  `xilinx@192.168.2.99`. Board credentials come from runner or environment
  configuration, never from this repository.
