"""Backend output translation. These run without any EDA tool installed,
which is the point: a parser regression should not need Verilator to catch."""

from __future__ import annotations

from ic_core.tools.lint.iverilog import parse as parse_icarus
from ic_core.tools.lint.verilator import parse as parse_verilator
from ic_core.tools.sim.summary import summarize
from ic_core.tools.synth.yosys import parse_stat

VERILATOR_LOG = """\
%Warning-WIDTHEXPAND: src/hw/rtl/x.sv:41:23: Operator ASSIGN expects 32 bits.
                    : ... note: In instance x
%Error: src/hw/rtl/y.sv:12:1: syntax error, unexpected ';'
%Error: Exiting due to 1 error(s)
"""


def test_verilator_diagnostics():
    issues = parse_verilator(VERILATOR_LOG)
    assert [i.severity for i in issues] == ["warning", "error"]
    assert issues[0].code == "WIDTHEXPAND"
    assert (issues[0].file, issues[0].line, issues[0].column) == ("src/hw/rtl/x.sv", 41, 23)
    assert issues[1].line == 12


def test_verilator_summary_lines_are_not_issues():
    """"Exiting due to N error(s)" has no file and would be a phantom issue."""
    assert all(i.file != "<unknown>" for i in parse_verilator(VERILATOR_LOG))


def test_icarus_diagnostics():
    issues = parse_icarus(
        "tb.sv:9: warning: implicit definition of wire 'x'.\n"
        "tb.sv:14: syntax error\n"
    )
    assert [i.severity for i in issues] == ["warning", "error"]
    assert issues[1].line == 14


def test_summary_puts_failures_first_and_never_drops_them():
    text = "\n".join(["noise"] * 50 + ["FAIL case 3: expected 7 got 6", "PASS case 4"])
    lines, truncated, passes, fails = summarize(text, limit=2)
    assert fails == 1 and passes == 1
    assert lines[0].startswith("FAIL")
    assert truncated


def test_summary_falls_back_to_the_tail():
    lines, _, passes, fails = summarize("a\nb\nc\nd\n", limit=2)
    assert lines == ["c", "d"] and passes == 0 and fails == 0


YOSYS_STAT = """\
=== counter_ref ===

   Number of cells:                 10
     LUT2                            2

=== design hierarchy ===

   Number of wires:                184
   Number of memory bits:            0
   Number of cells:                316
     DSP48E1                         1
     FDRE                           67
     LUT2                           40
     LUT6                           60
"""


def test_yosys_stat_uses_the_whole_design_not_a_submodule():
    utilization, summary = parse_stat(YOSYS_STAT)
    assert utilization.cells == 316
    assert utilization.luts == 100  # LUT2 + LUT6
    assert utilization.ffs == 67
    assert utilization.dsps == 1
    assert any("316" in line for line in summary)


def test_value_formatting_is_width_aware():
    from ic_core.tools.debug.vcd_reader import format_value

    # Sign-extension against the declared width, not the transmitted bits:
    # VCD drops leading zeros, so "b1000" on an 8-bit signal is 8, not -8.
    assert format_value("b1000", 8)["signed"] == 8
    assert format_value("b10000000", 8)["signed"] == -128
    assert format_value("b10000000", 8)["hex"] == "0x80"
    # A one-bit signal is never signed.
    assert format_value("1", 1)["signed"] == 1
    # X and Z never become numbers.
    unknown = format_value("b1x01", 4)
    assert unknown["unknown"] and unknown["dec"] is None and unknown["hex"] is None
