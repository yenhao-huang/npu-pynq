# Validation Gates

Choose gates from affected behavior, not merely changed file extensions. Record
the exact command, exit status, useful report path, and skipped/blocked rationale.

## Baseline

- Inspect `git status --short` before and after work; distinguish pre-existing
  user changes from current changes.
- Validate the selected OpenSpec change and keep its tasks synchronized.
- Check repository paths against `docs/rules/filetree.md`.
- Search the diff for generated artifacts, credentials, private keys, absolute
  machine-specific paths, and unrelated changes.

## Area gates

| Impact | Required gate |
| --- | --- |
| Python model/export/runtime | focused tests, then the relevant repository test suite |
| RTL/testbench | `make -C src/test lint` and `make -C src/test sim` unless a more focused command is required and the full commands follow before handoff |
| Numeric contract | golden-model edge cases including signed endpoints, rounding, saturation, and overflow affected by the change |
| Vivado Tcl/constraints | reproducible batch build or the narrowest available Tcl validation, plus utilization/timing/address inspection as applicable |
| Overlay/MMIO/physical I/O | board smoke test with matching `.bit`/`.hwh` and an observable PASS covering changed behavior |
| Documentation/examples | link/path check and executable notebook/example smoke test when practical |

## Gate outcomes

- `passed`: command or observable board evidence succeeded.
- `not_applicable`: impact analysis proves the gate cannot be affected; record
  the reasoning.
- `blocked`: required tool, license, board, network, or dependency is unavailable;
  record the exact condition and safe next command.
- `failed`: evidence contradicts the acceptance criterion; keep the OpenSpec
  task incomplete and diagnose or revise the design.

Simulation, lint, synthesis, and board validation are different claims. Never
substitute one for another.
