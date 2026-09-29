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


# -- ppa / OpenROAD ------------------------------------------------------

YOSYS_LIBERTY_STAT = """\
=== counter_ref ===

   Number of wires:                 42
   Number of cells:                 31
     AND2_X1                          3
     DFF_X1                           8
     INV_X1                           5
     NAND2_X1                        15

   Chip area for module '\\counter_ref': 60.325000
"""

OPENROAD_LOG = """\
[INFO ODB-0227] LEF file: Nangate45.lef
@ic section area
Design area 60 u^2 3% utilization.
@ic section timing
worst slack -0.42
tns -1.68
@ic section power
Group                  Internal  Switching    Leakage      Total
                          Power      Power      Power      Power (Watts)
----------------------------------------------------------------
           Sequential   1.05e-04   1.79e-05   1.13e-08   1.23e-04  47.8%
        Combinational   2.98e-05   4.39e-05   9.96e-09   7.38e-05  28.6%
                Clock   3.28e-05   2.79e-05   1.72e-09   6.07e-05  23.6%
                Macro   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.0%
                  Pad   0.00e+00   0.00e+00   0.00e+00   0.00e+00   0.0%
----------------------------------------------------------------
                Total   1.68e-04   8.98e-05   2.30e-08   2.58e-04 100.0%
"""


def test_yosys_liberty_stat_gives_cell_area_and_register_count():
    from ic_core.tools.ppa.openroad import parse_yosys_area

    area = parse_yosys_area(YOSYS_LIBERTY_STAT)
    assert area.cell_area_um2 == 60.325
    assert area.cells == 31
    assert area.sequential_cells == 8, "DFF_X1 is the only sequential cell"


def test_power_table_is_read_by_group_not_just_total():
    from ic_core.tools.ppa.openroad import parse_power

    power = parse_power(OPENROAD_LOG)
    assert power.total_w == 2.58e-04
    assert (power.internal_w, power.switching_w, power.leakage_w) == (1.68e-04, 8.98e-05, 2.30e-08)
    assert power.sequential_w == 1.23e-04
    assert power.clock_w == 6.07e-05
    assert power.combinational_w == 7.38e-05


def test_power_is_absent_rather_than_zero_when_the_table_is_missing():
    """A design that failed to link must not report 0 W as if it were measured."""
    from ic_core.tools.ppa.openroad import parse_power

    assert parse_power("[ERROR STA-0173] no liberty library").total_w is None


def test_fmax_is_derived_from_slack_against_the_constraint():
    from ic_core.tools.ppa.openroad import parse_timing

    timing = parse_timing(OPENROAD_LOG, period_ns=10.0)
    assert timing.wns_ns == -0.42 and timing.tns_ns == -1.68
    assert timing.met is False
    # 10 ns requested, 0.42 ns short: the design runs at 10.42 ns.
    assert timing.fmax_mhz == round(1000.0 / 10.42, 2)


def test_fmax_follows_slack_past_the_requested_period():
    """Negative slack lengthens the achievable period rather than voiding it:
    10 ns requested and 12 ns short is a 22 ns design, not a failure to report."""
    from ic_core.tools.ppa.openroad import parse_timing

    timing = parse_timing("worst slack -12.0\ntns -40.0\n", period_ns=10.0)
    assert timing.wns_ns == -12.0 and timing.fmax_mhz == round(1000.0 / 22.0, 2)


def test_fmax_is_null_rather_than_negative_on_a_degenerate_slack():
    """Slack larger than the period itself has no physical reading -- it comes
    from an unconstrained path -- so fmax is withheld instead of inverted."""
    from ic_core.tools.ppa.openroad import parse_timing

    assert parse_timing("worst slack 12.0", period_ns=10.0).fmax_mhz is None


def test_timing_without_a_slack_report_is_null_not_met():
    from ic_core.tools.ppa.openroad import parse_timing

    timing = parse_timing("no paths found", period_ns=5.0)
    assert timing.met is None and timing.clock_period_ns == 5.0


def test_area_merges_utilization_without_losing_the_yosys_number():
    """Yosys owns cell area because it needs no floorplan; utilization and die
    area only exist once one has run."""
    from ic_core.tools.ppa.openroad import merge_area, parse_yosys_area

    merged = merge_area(parse_yosys_area(YOSYS_LIBERTY_STAT), OPENROAD_LOG)
    assert merged.cell_area_um2 == 60.325, "not overwritten by the rounded 60 u^2"
    assert merged.utilization_pct == 3.0
    assert merged.die_area_um2 is None, "no floorplan ran in this log"
    assert merge_area(parse_yosys_area(""), "@ic die_area_um2 1234.5\n").die_area_um2 == 1234.5
