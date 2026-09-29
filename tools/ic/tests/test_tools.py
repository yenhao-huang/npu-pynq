"""End-to-end tests against the real backends.

Each is skipped when its tool is absent, so the suite is useful on a machine
with only some of the toolchain installed -- which is the normal case, since
Vivado is not on most of them.
"""

from __future__ import annotations

import pytest

from ic_core import dispatch
from ic_core.errors import BackendUnavailable, InvalidInput

from conftest import FIXTURES, needs, needs_env


@needs("verilator")
def test_lint_reports_a_real_diagnostic(store, tmp_path):
    bad = tmp_path / "bad.sv"
    bad.write_text(
        "module bad(input logic clk, output logic [3:0] q);\n"
        "  logic [7:0] wide;\n"
        "  always_ff @(posedge clk) q <= wide;\n"
        "endmodule\n"
    )
    out = dispatch("lint", {"files": [str(bad)], "top": "bad"}, cwd=tmp_path)
    assert out["warning_count"] >= 1
    widths = [i for i in out["issues"] if (i["code"] or "").startswith("WIDTH")]
    assert widths and widths[0]["line"] == 3
    # Warnings alone do not fail a lint; only errors do.
    assert out["ok"] and out["error_count"] == 0


@needs("verilator")
def test_lint_fails_on_a_syntax_error(store, tmp_path):
    bad = tmp_path / "broken.sv"
    bad.write_text("module broken;\n  this is not verilog\nendmodule\n")
    out = dispatch("lint", {"files": [str(bad)], "top": "broken"}, cwd=tmp_path)
    assert not out["ok"] and out["error_count"] >= 1


@needs("verilator")
def test_sim_returns_handles_not_contents(store, counter_wave):
    out = counter_wave
    assert out["wave"].endswith("wave.fst")
    assert out["log"].endswith("sim.log")
    # The verdict, not the log: this is the context budget the design exists for.
    assert len(out["summary"]) <= 40
    assert out["fail_count"] >= 1, "the fixture DUT is deliberately broken"


@needs("verilator")
@needs("fst2vcd")
def test_first_mismatch_finds_the_injected_bug(store, counter_wave):
    """The fixture's DUT stalls once at 7, so it must diverge at cycle 8."""
    out = dispatch(
        "first_mismatch",
        {"wave": counter_wave["wave"], "ref": "ref_count", "dut": "dut_count", "context": 3},
    )
    assert out["found"]
    assert out["ref_value"]["dec"] - out["dut_value"]["dec"] == 1
    assert out["ref_value"]["dec"] == 8
    # Context must show the last agreeing cycle, or it is not context.
    assert out["context"][0]["ref"]["dec"] == out["context"][0]["dut"]["dec"]


@needs("verilator")
@needs("fst2vcd")
def test_first_mismatch_reports_agreement_honestly(store, counter_wave):
    out = dispatch(
        "first_mismatch",
        {"wave": counter_wave["wave"], "ref": "ref_count", "dut": "ref_count"},
    )
    assert out["found"] is False and out["compared_cycles"] > 0


@needs("verilator")
@needs("fst2vcd")
def test_signals_and_value_queries(store, counter_wave):
    listed = dispatch("signals", {"wave": counter_wave["wave"], "pattern": "*count*"})
    assert {"TOP.tb_counter.ref_count", "TOP.tb_counter.dut_count"} <= {
        s["path"] for s in listed["signals"]
    }
    assert listed["clock"] == "TOP.tb_counter.clk"

    at = dispatch("value_at", {"wave": counter_wave["wave"],
                               "signals": ["ref_count", "nonexistent"], "cycle": 5})
    assert at["values"]["ref_count"]["dec"] == 5
    assert at["missing"] == ["nonexistent"]

    over = dispatch("value_range", {"wave": counter_wave["wave"], "signal": "ref_count",
                                    "from_cycle": 0, "to_cycle": 9, "max_points": 4})
    assert len(over["points"]) == 4 and over["truncated"] and over["total"] > 4


@needs("verilator")
@needs("fst2vcd")
def test_signal_names_accept_a_unique_suffix(store, counter_wave):
    """An agent should not have to reconstruct a full hierarchical path."""
    out = dispatch("value_at", {"wave": counter_wave["wave"], "signals": ["ref_count"], "cycle": 3})
    assert out["values"]["ref_count"]["dec"] == 3


@needs("verilator")
@needs("fst2vcd")
def test_debug_queries_do_not_litter_the_run_store(store, counter_wave):
    before = len(store.list_runs(limit=100))
    for cycle in range(5):
        dispatch("value_at", {"wave": counter_wave["wave"], "signals": ["ref_count"], "cycle": cycle})
    assert len(store.list_runs(limit=100)) == before


@needs("gtkwave")
@needs("verilator")
@needs("fst2vcd")
def test_show_wave_writes_a_savefile_without_a_display(store, counter_wave):
    out = dispatch("show_wave", {"wave": counter_wave["wave"],
                                 "signals": ["ref_count", "dut_count"],
                                 "center_cycle": 8, "launch": False})
    from pathlib import Path

    savefile = Path(out["savefile"])
    assert savefile.exists(), "the save file must survive the run being committed"
    body = savefile.read_text()
    assert "TOP.tb_counter.ref_count[7:0]" in body
    assert "[dumpfile]" in body
    assert out["launched"] is False


@needs("yosys")
def test_synth_estimate_reports_area_and_admits_it_is_an_estimate(store, tmp_path):
    out = dispatch(
        "synth",
        {"files": [str(FIXTURES / "counter.sv")], "top": "counter_ref", "mode": "estimate"},
        cwd=tmp_path,
    )
    assert out["ok"] and out["backend"] == "yosys"
    assert out["utilization"]["ffs"] == 8, "an 8-bit counter has 8 flip-flops"
    assert out["utilization"]["luts"], "INV cells occupy LUTs on 7-series"
    assert out["utilization"]["brams"] == 0, "zero, not null, when the stat block parsed"
    assert out["timing"]["met"] is None, "Yosys does no timing analysis"
    assert "estimate" in out["note"].lower()


def test_synth_full_selects_vivado_from_mode_alone():
    """`mode` is the agent-facing knob; the backend follows from it."""
    from ic_core.tools.synth import SynthIn

    assert SynthIn(files=["x.sv"], top="x", mode="full").backend == "vivado"
    assert SynthIn(files=["x.sv"], top="x").backend == "yosys"


def test_missing_backend_is_a_clear_error_not_a_crash(monkeypatch):
    import shutil as shutil_module

    monkeypatch.setattr(shutil_module, "which", lambda name: None)
    with pytest.raises(BackendUnavailable) as raised:
        dispatch("lint", {"files": ["x.sv"], "top": "x"})
    assert "verilator" in str(raised.value)


def test_invalid_input_names_the_offending_field():
    with pytest.raises(InvalidInput) as raised:
        dispatch("lint", {"top": "x"})
    assert any(e["field"] == "files" for e in raised.value.details["errors"])


def test_unknown_signal_suggests_how_to_find_the_right_one(store, counter_wave):
    with pytest.raises(InvalidInput) as raised:
        dispatch("value_range", {"wave": counter_wave["wave"], "signal": "no_such_signal"})
    assert "signals" in str(raised.value.details)


@needs("gtkwave")
@needs("verilator")
@needs("fst2vcd")
def test_savefile_carries_trace_flags_and_bit_ranges(store, counter_wave):
    """GTKWave ignores a bare signal name: without a `@22`/`@28` flag line, and
    without a vector's bit range, it opens an empty window that looks like the
    tool worked."""
    from pathlib import Path

    out = dispatch("show_wave", {"wave": counter_wave["wave"],
                                 "signals": ["ref_count", "enable"],
                                 "center_cycle": 8, "launch": False})
    body = Path(out["savefile"]).read_text().splitlines()
    assert "@22" in body and "TOP.tb_counter.ref_count[7:0]" in body
    assert "@28" in body and "TOP.tb_counter.enable" in body
    marker = [line for line in body if line.startswith("*")]
    assert marker and marker[0].split()[1] != "-1", "the marker must land on the cycle"


@needs("verilator")
def test_sim_runs_from_the_callers_directory(store, tmp_path):
    """Testbenches open fixtures by relative path, as they do under `make sim`."""
    (tmp_path / "data.txt").write_text("42\n")
    tb = tmp_path / "tb_reads_file.sv"
    tb.write_text(
        "module tb_reads_file;\n"
        "  integer f, v;\n"
        "  initial begin\n"
        '    f = $fopen("data.txt", "r");\n'
        '    if (f == 0) $display("FAIL cannot open data.txt");\n'
        '    else begin void\'($fscanf(f, "%d", v)); $display("PASS read %0d", v); end\n'
        "    $finish;\n"
        "  end\n"
        "endmodule\n"
    )
    out = dispatch("sim", {"files": [str(tb)], "tb": "tb_reads_file"}, cwd=tmp_path)
    assert out["ok"], out["summary"]
    assert out["wave"] and store.resolve(out["wave"]).exists()


@needs("iverilog")
def test_icarus_sim_traces_without_bind(store, tmp_path):
    """Icarus 11 has no `bind`; the probe must still produce a waveform."""
    out = dispatch(
        "sim",
        {"files": [str(FIXTURES / "counter.sv"), str(FIXTURES / "tb_counter.sv")],
         "tb": "tb_counter", "backend": "icarus"},
        cwd=tmp_path,
    )
    assert out["built"], out["summary"]
    assert out["wave"] and out["wave"].endswith("wave.vcd")
    found = dispatch("first_mismatch", {"wave": out["wave"], "ref": "ref_count",
                                        "dut": "dut_count", "backend": "vcd"})
    assert found["found"] and found["ref_value"]["dec"] == 8


# -- ppa -----------------------------------------------------------------


def test_ppa_maps_registers_before_technology_mapping():
    """`dfflibmap` must precede `abc`, or the flops reach OpenROAD as generic
    `$_DFF_` cells it cannot link."""
    from ic_core.tools.ppa import PpaIn
    from ic_core.tools.ppa.openroad import yosys_script

    script = yosys_script(
        PpaIn(files=["a.sv"], top="a", liberty=["lib.lib"]), ["lib.lib"], __import__("pathlib").Path("n.v")
    )
    assert script.index("dfflibmap") < script.index("abc ")
    assert "-liberty lib.lib" in script
    assert "stat -liberty lib.lib" in script


def test_ppa_estimate_does_not_floorplan_and_placed_does():
    from ic_core.tools.ppa import PpaIn
    from ic_core.tools.ppa.openroad import openroad_script

    common = {"files": ["a.sv"], "top": "a", "liberty": ["lib.lib"], "clock_port": "clk"}
    estimate = openroad_script(PpaIn(**common), ["lib.lib"], "n.v", None, [])
    assert "initialize_floorplan" not in estimate
    assert "create_clock -name clk -period 10.0 [get_ports clk]" in estimate
    assert "report_power" in estimate and "report_worst_slack" in estimate

    placed = openroad_script(
        PpaIn(**common, mode="placed", site="FreePDK45", hor_layer="metal3", ver_layer="metal2"),
        ["lib.lib"], "n.v", "tech.lef", ["cells.lef"],
    )
    assert "initialize_floorplan -site FreePDK45" in placed
    assert placed.index("read_lef tech.lef") < placed.index("read_liberty lib.lib")
    assert placed.index("global_placement") < placed.index("detailed_placement")
    # Parasitics are re-estimated after legalisation, not only before it.
    assert placed.count("estimate_parasitics -placement") == 2


def test_ppa_without_a_clock_port_says_the_slack_is_unconstrained():
    from ic_core.tools.ppa import PpaIn
    from ic_core.tools.ppa.openroad import _note, openroad_script

    params = PpaIn(files=["a.sv"], top="a", liberty=["lib.lib"])
    script = openroad_script(params, ["lib.lib"], "n.v", None, [])
    assert "create_clock -name clk -period 10.0\n" in script, "a virtual clock"
    assert "unconstrained" in _note("estimate", params, virtual_clock=True)
    assert "unconstrained" not in _note("estimate", params, virtual_clock=False)


def test_ppa_reads_the_pdk_from_the_environment(monkeypatch, tmp_path):
    """An agent calls `ppa(files, top)`; the machine supplies the technology."""
    from ic_core.tools.ppa import PpaIn
    from ic_core.tools.ppa.openroad import resolve_pdk

    import os

    lib = tmp_path / "cells.lib"
    lib.write_text("library(test) {}\n")
    monkeypatch.setenv("IC_PDK_LIBERTY", str(lib))
    assert resolve_pdk(PpaIn(files=["a.sv"], top="a"))["liberty"] == [str(lib)]

    second = tmp_path / "more.lib"
    second.write_text("library(more) {}\n")
    monkeypatch.setenv("IC_PDK_LIBERTY", os.pathsep.join([str(lib), str(second)]))
    assert len(resolve_pdk(PpaIn(files=["a.sv"], top="a"))["liberty"]) == 2


def test_ppa_names_every_missing_piece_of_the_technology(monkeypatch):
    from ic_core.tools.ppa import PpaIn
    from ic_core.tools.ppa.openroad import resolve_pdk

    for name in ("IC_PDK_LIBERTY", "IC_PDK_TECH_LEF", "IC_PDK_LEF", "IC_PDK_SITE",
                 "IC_PDK_HOR_LAYER", "IC_PDK_VER_LAYER"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(InvalidInput) as raised:
        resolve_pdk(PpaIn(files=["a.sv"], top="a"))
    assert any("liberty" in m for m in raised.value.details["missing"])

    # One error listing all five, rather than five round trips.
    with pytest.raises(InvalidInput) as raised:
        resolve_pdk(PpaIn(files=["a.sv"], top="a", mode="placed"))
    assert len(raised.value.details["missing"]) == 6


def test_ppa_rejects_a_technology_file_that_does_not_exist(monkeypatch):
    from ic_core.tools.ppa import PpaIn
    from ic_core.tools.ppa.openroad import resolve_pdk

    monkeypatch.delenv("IC_PDK_LIBERTY", raising=False)
    with pytest.raises(InvalidInput) as raised:
        resolve_pdk(PpaIn(files=["a.sv"], top="a", liberty=["/nonexistent/cells.lib"]))
    assert "/nonexistent/cells.lib" in raised.value.details["missing"]


def test_ppa_activity_outside_zero_to_one_is_rejected():
    """Power scales with activity, so a nonsense value would produce a
    plausible-looking number rather than an error."""
    with pytest.raises(InvalidInput) as raised:
        dispatch("ppa", {"files": ["a.sv"], "top": "a", "input_activity": 4.0})
    assert raised.value.details["errors"]


def test_ppa_needs_yosys_as_well_as_openroad(monkeypatch, store, tmp_path):
    """`requires` covers one executable; the mapping step needs the other, and
    a missing one must be the same answerable error rather than a crash."""
    import shutil as shutil_module

    real_which = shutil_module.which
    monkeypatch.setattr(
        shutil_module, "which", lambda name: None if name == "yosys" else real_which(name) or "/usr/bin/openroad"
    )
    with pytest.raises(BackendUnavailable) as raised:
        dispatch("ppa", {"files": [str(FIXTURES / "counter.sv")], "top": "counter_ref"},
                 cwd=tmp_path)
    assert "yosys" in str(raised.value)


def test_ppa_docker_provider_mounts_project_run_and_pdk_directories(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from ic_core.tools.ppa import PpaIn
    from ic_core.tools.ppa.openroad import _openroad_environment

    project = tmp_path / "project"
    work = tmp_path / "store" / "work"
    artifacts = tmp_path / "store" / "artifacts"
    pdk = tmp_path / "pdk"
    for directory in (project, work, artifacts, pdk):
        directory.mkdir(parents=True)
    liberty = pdk / "cells.lib"
    liberty.write_text("library(test) {}\n")
    script = work / "ppa.tcl"
    netlist = artifacts / "netlist.v"
    monkeypatch.setenv("IC_OPENROAD_DOCKER_IMAGE", "openroad/orfs:pinned")

    env = _openroad_environment(
        SimpleNamespace(cwd=project),
        PpaIn(files=["design.sv"], top="top", liberty=[str(liberty)]),
        {"liberty": [str(liberty)], "lef": [], "tech_lef": None},
        script,
        netlist,
    )
    mounts = set(json.loads(env["IC_OPENROAD_MOUNTS"]))
    assert {str(project.resolve()), str(work.resolve()), str(artifacts.resolve()),
            str(pdk.resolve())} <= mounts


@needs("openroad")
@needs("yosys")
@needs_env("IC_PDK_LIBERTY")
def test_ppa_measures_power_area_and_fmax_on_a_real_library(store, tmp_path):
    """The whole flow, on a machine that has a PDK. Everything above this line
    tests the translation; this tests that the two tools actually run."""
    design = tmp_path / "acc.sv"
    design.write_text(
        "module acc(input logic clk, input logic [3:0] a, input logic [3:0] b,\n"
        "           output logic [4:0] q);\n"
        "  always_ff @(posedge clk) q <= a + b;\n"
        "endmodule\n"
    )
    out = dispatch(
        "ppa",
        {"files": [str(design)], "top": "acc", "clock_port": "clk", "clock_period_ns": 2.0},
        cwd=tmp_path,
    )
    assert out["ok"], out["summary"]
    assert out["area"]["cell_area_um2"] > 0
    assert out["area"]["sequential_cells"] == 5, "a 5-bit register"
    assert out["power"]["total_w"] > 0
    assert out["power"]["total_w"] >= out["power"]["leakage_w"]
    assert out["timing"]["fmax_mhz"] and out["timing"]["wns_ns"] is not None
    # Handles, not contents: the log stays on disk like every other tool's.
    assert out["netlist"].endswith("netlist.v") and out["report"].endswith("ppa.log")
    assert "not placed or routed" in out["note"]
