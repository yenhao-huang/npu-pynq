# Simulation Layer

`sim/` turns `src/rtl` and `src/tb` into a pass/fail exit code. It is the only
thing CI can meaningfully run without an EDA licence.

```text
sim/
|-- Makefile
|-- cocotb/          Python tests, one per DUT
|-- verilator/       C++ wrappers, if used
`-- waves/           .vcd output, ignored
```

## Choosing the harness

Use **cocotb** when the design has a software golden model to compare against —
accelerators, DSP, anything numeric. The reference implementation lives in
`sw/` and the test calls it directly, so the oracle is never reimplemented in
HDL.

Use a plain **SystemVerilog testbench with Icarus** for control logic, protocol
handshakes, and state machines, where the check is a waveform assertion rather
than a numeric comparison.

Both can coexist. Icarus supports enough SystemVerilog for testbenches with
`-g2012`; Verilator is faster but requires synthesizable-subset testbenches or a
C++ wrapper.

## Makefile contract

Expose exactly three targets so CI never needs to know the tool:

- `make lint` — static check, no elaboration of testbenches
- `make sim` — run every test, non-zero exit on failure
- `make clean`

A minimal Icarus implementation:

```makefile
RTL  := $(wildcard ../src/rtl/**/*.sv)
TB   := $(wildcard ../src/tb/**/*.sv)

lint:
	verilator --lint-only -Wall --top-module $(TOP) $(RTL)

sim: $(patsubst %.sv,%.run,$(TB))

%.run: %.sv
	iverilog -g2012 -o build/$(@F) $(RTL) $<
	vvp build/$(@F) | tee build/$(@F).log
	! grep -q "ERROR\|FAIL" build/$(@F).log
```

The final `grep` matters: `vvp` exits 0 even when `$display` reported a
mismatch, so without it CI is green on every failure.
