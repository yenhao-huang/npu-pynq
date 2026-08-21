# Environment Rules

Primary language: PowerShell
Runtime version: Windows PowerShell 5.1 or PowerShell 7+
Package manager: none
Frameworks: none
Service manager: none
Required services: OpenSSH `ssh` and `scp`; reachable PYNQ Linux SSH server

- Default SSH host: `pynq_board`.
- Default SSH user: `xilinx`.
- Default remote root: `/home/xilinx/jupyter_notebooks`.
- If the hostname is not configured, accept an explicit board IP such as
  `192.168.2.99`.
- Use existing SSH configuration and interactive authentication. Never store a
  password or private key in this skill.
- Require the computer and board to be on the same reachable network before a
  live transfer.
