# Simulation Rules

`src/test/Makefile` is the stable interface between contributors, CI, and the
available simulator. It must expose:

- `make lint`: static RTL validation.
- `make sim`: all discovered self-checking testbenches.
- `make clean`: generated test output removal.

Rules:

- Synthesizable RTL lives under `src/hw/rtl/<design>/`; testbench RTL lives
  under `src/hw/tb/<design>/` and is never synthesized.
- Testbenches are named `tb_<module>.sv` so the Makefile and CI can discover
  them.
- Every RTL behavior change requires corresponding test coverage or an explicit
  explanation of why existing coverage is unchanged.
- Use Python/cocotb when a numeric accelerator needs comparison with the golden
  model in `src/test/model/`. Use self-checking SystemVerilog for protocols,
  control logic, and state machines when appropriate.
- A simulator process exiting zero is insufficient if the log contains an
  error, failure, or mismatch. The Makefile must turn those conditions into a
  non-zero target result.
- Generated simulation output belongs under `src/test/build/` or
  `src/test/waves/` and must remain ignored.
- Run `make -C src/test lint sim` before requesting merge.

## Interactive iteration

`make -C src/test lint sim` remains the gate for CI and for merge. It is not
the fastest way to work while a change is still being shaped.

`tools/ic/` exists for that inner loop: `ic lint` after each RTL edit, `ic sim`
for a verdict plus a waveform handle, and `ic first_mismatch` to locate the
cycle where a signal diverges from its reference without opening the waveform.
`ic synth --mode estimate` gives rough area in seconds; only
`ic synth --mode full` produces timing that may be reported as timing.

Rules:

- The `ic` tools supplement the Makefile; they never replace it. A change is
  not ready to merge on `ic sim` alone.
- `ic` writes to `.ic/`, which is machine-local and never committed. Simulation
  output produced by the Makefile still belongs under `src/test/build/` or
  `src/test/waves/`.
- `ic sim` traces through a generated probe module. Do not add
  `$dumpfile`/`$dumpvars` to a checked-in testbench to make it work.
- Verilator, the default backend, is two-state. Use `--backend icarus` for
  reset, initialisation, and X-propagation problems.

See [../ic-design-tools/README.md](../ic-design-tools/README.md).
