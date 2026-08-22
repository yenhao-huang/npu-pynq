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
