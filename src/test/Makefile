# Verification entry points. CI runs `lint` and `sim`; both must work with only
# apt-installable tools (verilator, iverilog).
#
# TOP   overrides the module verilator lints as the design root.
# TESTS lists testbench basenames without the tb_ prefix; each must exist as
#       ../hw/tb/<design>/tb_<name>.sv.

# RTL is every .sv under ../hw/rtl except the CVA6 scalar core sources that
# need the CVA6 packages: the upstream submodule (hundreds of files that must be
# read in package order, and that iverilog cannot compile) and our files that
# import its packages. Those are read only through scalar_core/cva6/cva6.f,
# which also defines NPU_SCALAR_CVA6. Rule: a file under scalar_core/cva6/ that
# depends on a CVA6 or pulp package goes in CVA6_ONLY; a self-contained one
# (cva6_npu_mmio.sv, cva6_dma.sv) stays in RTL so lint and sim cover it.
CVA6_DIR  := ../hw/rtl/npu_accelerator/scalar_core/cva6
CVA6_ONLY := $(CVA6_DIR)/cva6_config.sv $(CVA6_DIR)/scalar_core_cva6.sv
RTL    := $(filter-out $(CVA6_ONLY),$(shell find ../hw/rtl -type f -name '*.sv' -not -path '*/scalar_core/cva6/upstream/*' -print))
TB_DIR := ../hw/tb
BUILD  := build
TOP    ?=
# tb_scalar_core_cva6 needs CVA6 and Verilator; it runs only under sim-cva6.
TESTS  ?= $(filter-out scalar_core_cva6,$(patsubst tb_%.sv,%,$(notdir $(wildcard $(TB_DIR)/*/tb_*.sv))))
PYTHON ?= python

.PHONY: all model lint lint-cva6 sim sim-cva6 sim-cva6-custom clean
all: model lint sim

model:
	cd ../.. && $(PYTHON) -m unittest discover -s src/test/tests -v

lint:
ifeq ($(strip $(RTL)),)
	@echo "no RTL under src/hw/rtl yet, nothing to lint"
else
	verilator --lint-only -Wall -Wno-DECLFILENAME \
	  $(if $(TOP),--top-module $(TOP),) $(RTL)
endif

# Optional, not run by CI: lints the scalar core with SCALAR_IMPL="CVA6" (CVA6,
# the pulp AXI blocks and our cva6_* files). Needs the submodule:
#   git submodule update --init src/hw/rtl/npu_accelerator/scalar_core/cva6/upstream
#   (then, inside it: git submodule update --init core/cvfpu
#    core/cache_subsystem/hpdcache, and core/cvfpu/src/fpu_div_sqrt_mvp)
# --timescale gives the upstream files, which have none, the one ours use.
lint-cva6:
	verilator --lint-only -Wall -Wno-DECLFILENAME --timescale 1ns/1ps \
	  --top-module scalar_core -GSCALAR_IMPL='"CVA6"' \
	  $(CVA6_DIR)/waiver.vlt -F $(CVA6_DIR)/cva6.f $(RTL)

# Optional, not run by CI: system simulation of npu_accelerator with
# SCALAR_IMPL="CVA6" (tb_scalar_core_cva6) under Verilator. The ISA test
# programs of vectors/isa_frontend_16x16.txt are translated to RISC-V by
# vectors/risc_v_npu/isa_to_rv.py into $(CVA6_VECTORS); every final memory
# image must equal isa_sim.py's. Needs the CVA6 submodule (see lint-cva6), a
# Verilator with --timing support and a C++20 compiler.
# sim-cva6-custom builds the same bench with SCALAR_IMPL="CUSTOM" on the
# original ISA programs (the A side of exp/scalar_core_compare).
#
# The build is two steps so it also works on a Windows host whose Verilator
# has no C++ compiler next to it: Verilator writes C++ into $(BUILD)/obj_<impl>,
# then make + g++ build and run it on $(CVA6_HOST). Defaults:
#   Linux:   CVA6_VERILATE = verilator, CVA6_HOST empty.
#   Windows (MSYS2 make, PATH=/c/msys64/usr/bin:/c/msys64/ucrt64/bin:$$PATH):
#            CVA6_VERILATE = verilator_bin.exe of MSYS2 ucrt64 (the perl
#            wrapper is bypassed); CVA6_HOST = WSL (Ubuntu, g++ >= 10, make,
#            python3), which compiles against the same Verilator include tree
#            through /mnt/c (CVA6_VROOT). The generated __ver.d holds Windows
#            paths that GNU make in WSL cannot parse, so it is emptied (it only
#            serves Verilator's own rebuild check).
CVA6_TB      := $(TB_DIR)/npu_accelerator/tb_scalar_core_cva6.sv
CVA6_VECTORS := $(BUILD)/cva6_frontend_16x16.txt
CVA6_VFLAGS  := --cc --exe --main --timing -Wno-fatal -Wno-lint -Wno-style \
                --timescale 1ns/1ps --top-module tb_scalar_core_cva6
CVA6_JOBS    ?= 4
ifneq ($(findstring MSYS,$(shell uname -s))$(findstring MINGW,$(shell uname -s)),)
MSYS_VERILATOR_ROOT ?= C:/msys64/ucrt64/share/verilator
CVA6_VERILATE ?= VERILATOR_ROOT=$(MSYS_VERILATOR_ROOT) /c/msys64/ucrt64/bin/verilator_bin.exe
WSL_DISTRO    ?= Ubuntu
WSL_TEST_DIR  := $(shell echo '$(CURDIR)' | sed -E 's|^/([a-z])/|/mnt/\1/|')
CVA6_HOST     ?= MSYS_NO_PATHCONV=1 wsl -d $(WSL_DISTRO) --cd $(WSL_TEST_DIR) --
CVA6_VROOT    ?= /mnt/c/msys64/ucrt64/share/verilator
else
CVA6_VERILATE ?= verilator
CVA6_HOST     ?=
CVA6_VROOT    ?= $(shell verilator --getenv VERILATOR_ROOT)
endif

# $(1) = impl (cva6 | custom), $(2) = extra Verilator arguments
define cva6_sim
	@mkdir -p $(BUILD)
	$(CVA6_VERILATE) $(CVA6_VFLAGS) -Mdir $(BUILD)/obj_$(1) $(2) \
	  $(CVA6_DIR)/waiver.vlt -F $(CVA6_DIR)/cva6.f $(RTL) $(CVA6_TB)
	: > $(BUILD)/obj_$(1)/Vtb_scalar_core_cva6__ver.d
	$(CVA6_HOST) make -s -j$(CVA6_JOBS) -C $(BUILD)/obj_$(1) \
	  -f Vtb_scalar_core_cva6.mk VERILATOR_ROOT=$(CVA6_VROOT)
	$(CVA6_HOST) $(BUILD)/obj_$(1)/Vtb_scalar_core_cva6 | tee $(BUILD)/scalar_core_cva6_$(1).log
	@# CVA6's Verilator commit tracer (cva6.sv) writes PCs to the working directory.
	-@mv -f trace_hart_0.dasm $(BUILD)/trace_hart_0_$(1).dasm 2>/dev/null
	@grep -q '^PASS tb_scalar_core_cva6' $(BUILD)/scalar_core_cva6_$(1).log
endef

$(CVA6_VECTORS): vectors/isa_frontend_16x16.txt vectors/risc_v_npu/isa_to_rv.py vectors/risc_v_npu/rv32asm.py
	@mkdir -p $(BUILD)
	cd ../.. && $(PYTHON) -m src.test.vectors.risc_v_npu.isa_to_rv -o src/test/$@

sim-cva6: $(CVA6_VECTORS)
	$(call cva6_sim,cva6,)

sim-cva6-custom:
	$(call cva6_sim,custom,-DTB_IMPL_CUSTOM)

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
