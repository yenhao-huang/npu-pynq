# Simulation entry points. CI runs `lint` and `sim`; both must work with only
# apt-installable tools (verilator, iverilog).

RTL    := $(wildcard ../src/rtl/*/*.sv)
TB_DIR := ../src/tb
BUILD  := build
TESTS  := mac_unit mac_axi_lite

.PHONY: all lint sim clean
all: lint sim

lint:
	verilator --lint-only -Wall -Wno-DECLFILENAME \
	  --top-module mac_axi_lite $(RTL)

sim: $(addprefix $(BUILD)/,$(addsuffix .ok,$(TESTS)))

$(BUILD)/%.ok: $(RTL)
	@mkdir -p $(BUILD)
	iverilog -g2012 -s tb_$* -o $(BUILD)/$*.vvp $(RTL) $(TB_DIR)/*/tb_$*.sv
	vvp $(BUILD)/$*.vvp | tee $(BUILD)/$*.log
	@# vvp exits 0 even when the testbench reports a mismatch, so check the log.
	@! grep -qiE 'error|fail|mismatch' $(BUILD)/$*.log
	@touch $@

clean:
	rm -rf $(BUILD) waves
