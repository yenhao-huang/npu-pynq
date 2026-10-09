# Environment Rules

Primary language: bash (macOS system bash 3.2 compatible)
Runtime version: macOS 13+
Package manager: Homebrew for `gh` and LLVM 22; the repository venv for Python
Required services: OpenSSH `ssh`/`scp`; `gh` authenticated to the repository

- Board: `xilinx@192.168.2.99` by default; override with `--host`/`--user` or
  `PYNQ_BOARD_HOST`/`PYNQ_BOARD_USER`.
- The board is reached over a direct Ethernet cable. The Mac's adapter needs a
  manual IPv4 address on the board's /24 (for example `192.168.2.1`).
- SSH must work without a password (`ssh-copy-id xilinx@192.168.2.99` once).
- The board must allow `sudo -n /usr/local/share/pynq-venv/bin/python3` and
  `sudo -n chmod`, as the CD board steps require.
- `build/deploy/` holds staged overlays, packages and fetched results; it is
  ignored by Git.
