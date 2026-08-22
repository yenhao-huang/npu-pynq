# Verification entry points. CI runs `lint` and `sim`; both must work with only
# apt-installable tools (verilator, iverilog).
#
# TOP   overrides the module verilator lints as the design root.
# TESTS lists testbench basenames without the tb_ prefix; each must exist as
#       ../hw/tb/<design>/tb_<name>.sv.

RTL    := $(wildcard ../hw/rtl/*/*.sv)
TB_DIR := ../hw/tb
BUILD  := build
TOP    ?=
TESTS  ?= $(patsubst tb_%.sv,%,$(notdir $(wildcard $(TB_DIR)/*/tb_*.sv)))
PYTHON ?= python

.PHONY: all model lint sim clean
all: model lint sim

model:
	cd ../.. && $(PYTHON) -m unittest discover -s src/test/tests -v
	cd ../.. && $(PYTHON) -m unittest discover -s examples/matrix-multiplication/tests -v

lint:
ifeq ($(strip $(RTL)),)
	@echo "no RTL under src/hw/rtl yet, nothing to lint"
else
	verilator --lint-only -Wall -Wno-DECLFILENAME \
	  $(if $(TOP),--top-module $(TOP),) $(RTL)
endif

sim: $(addprefix $(BUILD)/,$(addsuffix .ok,$(TESTS)))
ifeq ($(strip $(TESTS)),)
	@echo "no testbenches under src/hw/tb yet, nothing to simulate"
endif

$(BUILD)/%.ok: $(RTL)
	@mkdir -p $(BUILD)
	iverilog -g2012 -s tb_$* -o $(BUILD)/$*.vvp $(RTL) $(TB_DIR)/*/tb_$*.sv
	vvp $(BUILD)/$*.vvp | tee $(BUILD)/$*.log
	@# vvp exits 0 even when the testbench reports a mismatch, so check the log.
	@! grep -qiE 'error|fail|mismatch' $(BUILD)/$*.log
	@touch $@

clean:
	rm -rf $(BUILD) waves/*.vcd waves/*.fst
