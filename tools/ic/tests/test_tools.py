"""End-to-end tests against the real backends.

Each is skipped when its tool is absent, so the suite is useful on a machine
with only some of the toolchain installed -- which is the normal case, since
Vivado is not on most of them.
"""

from __future__ import annotations

import pytest

from ic_core import dispatch
from ic_core.errors import BackendUnavailable, InvalidInput

from conftest import FIXTURES, needs


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
