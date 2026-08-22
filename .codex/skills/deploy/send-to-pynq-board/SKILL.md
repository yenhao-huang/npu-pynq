---
name: send-to-pynq-board
description: Upload Vivado/PYNQ overlay files and optional notebooks from Windows to the SSH host `pynq_board` or a supplied PYNQ IP address. Use when the user asks to send, copy, transfer, deploy, or install `.bit`, `.hwh`, `.tcl`, `.ipynb`, or related files onto a PYNQ board, especially when the bitstream and hardware handoff need matching remote basenames.
---

# Send to PYNQ Board

Transfer a PYNQ overlay through OpenSSH without storing passwords. Use the
bundled PowerShell script for deterministic naming and validation.

## Workflow

1. Find candidate files with `rg --files -g '*.bit' -g '*.hwh' -g '*.ipynb'`.
2. Select the matching `.bit` and `.hwh` from the same Vivado design. If the
   choice is ambiguous, show candidates and ask the user which pair to use.
3. Choose an overlay name. Strip a trailing `_wrapper` from the bitstream name
   by default so the remote `.bit` and `.hwh` share one basename.
4. Read `references/rules/env.md`, then run
   `references/scripts/send_to_pynq_board.ps1` with `-DryRun` first.
5. Review the exact local sources, remote host, remote directory, and renamed
   destinations printed by the dry run.
6. Run the same command without `-DryRun`. SSH may interactively request host
   trust or the board password; never capture, save, or echo that password.
7. Verify the remote listing printed by the script. Report the uploaded paths.

## Command

```powershell
& npu_repo_in_pynq/.codex/skills/deploy/send-to-pynq-board/references/scripts/send_to_pynq_board.ps1 `
  -BitPath <design.bit> `
  -HwhPath <design.hwh> `
  -OverlayName <overlay-name> `
  -BoardHost pynq_board `
  -DryRun
```

Remove `-DryRun` only after checking the plan. If `pynq_board` is not a
configured SSH hostname, pass the board IP, commonly `-BoardHost 192.168.2.99`.

Use `-ExtraPath <file1>,<file2>` for notebooks or Tcl files. The default remote
directory is `/home/xilinx/jupyter_notebooks/<overlay-name>`.

## Guardrails

- Do not upload unless the user asks for a transfer; inspection requests are
  read-only.
- Do not guess between multiple overlay pairs.
- Require `.bit` and `.hwh` and normalize both remote basenames.
- Do not put credentials in commands, files, logs, or skill state.
- Do not delete or overwrite unrelated remote files. The two normalized overlay
  files may replace files with those exact names after the dry run makes this
  visible.
- Stop on SSH, SCP, validation, or remote verification failure.
