"""OpenROAD PPA backend.

Two external programs, one run directory. Yosys maps the RTL onto the cell
library and reports the area; OpenROAD links the resulting netlist and reports
power and slack. They are one run because they answer one question, and
splitting them would leave an agent holding a netlist it has to name.

    RTL --yosys--> netlist.v --openroad--> power, slack  (+ floorplan, placement)
             |                                   |
          area, cell counts                   die area, parasitics

Area comes from Yosys rather than from OpenROAD's `report_design_area` even
when both are available. `stat -liberty` sums the same cell areas, it works
without a floorplan, and it means the area number never depends on whether
this machine's OpenROAD needed LEF to read the netlist.

The PDK is named by the environment (`IC_PDK_LIBERTY` and friends) so an agent
can call `ppa(files, top)` and get an answer. Nothing here targets the
Zynq-7020: standard-cell PPA is a yardstick for ranking two implementations,
and `synth --mode full` remains the only statement about the board.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from ...errors import BackendUnavailable, InvalidInput
from ...process import run as run_process
from ...registry import backend
from . import Area, Power, PpaIn, PpaOut, PpaTiming

# -- report parsing ------------------------------------------------------
#
# Every regex below is exercised by tests/test_parsers.py against captured
# report text, so a format change is caught without a PDK installed.

#: Yosys `stat -liberty` closes with the summed cell area of the top module.
CHIP_AREA = re.compile(r"Chip area for (?:top )?module '\\?([^']+)':\s*([\d.]+)")
STAT_CELLS = re.compile(r"Number of cells:\s*(\d+)")
#: A mapped sequential cell is whatever the library calls a flop; every
#: standard-cell library spells them DFF/DLL/SDFF/LATCH somewhere in the name.
SEQ_CELL = re.compile(r"^\s+(\S*(?:DFF|DLL|SDFF|LATCH|dff|latch)\S*)\s+(\d+)\s*$", re.M)

#: OpenSTA: "Design area 1532 u^2 12% utilization."
DESIGN_AREA = re.compile(r"Design area\s+([\d.]+)\s*u\^2\s+([\d.]+)\s*%\s*utilization")
#: OpenSTA prints "worst slack -0.42", with an optional MAX/MIN qualifier.
WORST_SLACK = re.compile(r"worst slack\s+(?:MAX\s+|MIN\s+)?(-?[\d.]+)")
TNS = re.compile(r"^\s*tns\s+(-?[\d.]+)", re.M)
#: The `report_power` group table. Four columns of watts, then a percentage.
POWER_ROW = re.compile(
    r"^\s*(Sequential|Combinational|Clock|Macro|Pad|Total)\s+"
    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
    re.M,
)
#: Values this backend's Tcl prints itself, for what no report command gives.
MARKER = re.compile(r"^@ic\s+(\S+)\s+(.*)$", re.M)


def parse_yosys_area(text: str) -> Area:
    """Read `stat -liberty`: cell area, instance count, how many are registers."""
    area_match = None
    for area_match in CHIP_AREA.finditer(text):
        pass  # the last one is the top module, earlier ones are submodules
    cells_match = None
    for cells_match in STAT_CELLS.finditer(text):
        pass
    sequential = None
    seq_rows = SEQ_CELL.findall(text)
    if seq_rows:
        sequential = sum(int(count) for _name, count in seq_rows)
    return Area(
        cell_area_um2=float(area_match.group(2)) if area_match else None,
        cells=int(cells_match.group(1)) if cells_match else None,
        sequential_cells=sequential,
    )


def parse_power(text: str) -> Power:
    """Read the `report_power` table. Watts, as OpenSTA prints them."""
    rows = {m.group(1): m for m in POWER_ROW.finditer(text)}
    total = rows.get("Total")
    if total is None:
        return Power()

    def group_total(name: str) -> float | None:
        row = rows.get(name)
        return float(row.group(5)) if row else None

    return Power(
        internal_w=float(total.group(2)),
        switching_w=float(total.group(3)),
        leakage_w=float(total.group(4)),
        total_w=float(total.group(5)),
        sequential_w=group_total("Sequential"),
        combinational_w=group_total("Combinational"),
        clock_w=group_total("Clock"),
    )


def parse_timing(text: str, period_ns: float) -> PpaTiming:
    """Slack against the constraint, and the fmax it implies.

    `fmax` is derived rather than measured: the achievable period is the
    requested one minus the slack. It is pre-route and therefore optimistic,
    which `note` on the result says, but it ranks two designs correctly.
    """
    slack_match = WORST_SLACK.search(text)
    if not slack_match:
        return PpaTiming(clock_period_ns=period_ns, met=None)
    wns = float(slack_match.group(1))
    tns_match = TNS.search(text)
    achievable = period_ns - wns
    return PpaTiming(
        clock_period_ns=period_ns,
        wns_ns=wns,
        tns_ns=float(tns_match.group(1)) if tns_match else None,
        met=wns >= 0,
        fmax_mhz=round(1000.0 / achievable, 2) if achievable > 0 else None,
    )


def parse_markers(text: str) -> dict[str, str]:
    return {name: value.strip() for name, value in MARKER.findall(text)}


def merge_area(base: Area, text: str) -> Area:
    """Fold OpenROAD's view of the area into Yosys's.

    Yosys owns `cell_area_um2` because it needs no floorplan. Utilization and
    die area only exist once one has run, so they come from here or not at all.
    """
    merged = base.model_copy()
    design = DESIGN_AREA.search(text)
    if design:
        merged.utilization_pct = float(design.group(2))
        if merged.cell_area_um2 is None:
            merged.cell_area_um2 = float(design.group(1))
    die = parse_markers(text).get("die_area_um2")
    if die:
        try:
            merged.die_area_um2 = round(float(die), 3)
        except ValueError:
            pass
    return merged


# -- script generation ---------------------------------------------------


def yosys_script(params: PpaIn, liberty: list[str], netlist: Path) -> str:
    """Map to cells the way a standard-cell flow does, then report the area.

    `dfflibmap` before `abc` is not interchangeable: registers must be bound to
    library flops first, or `abc` maps their logic and leaves generic `$_DFF_`
    cells that OpenROAD cannot link.
    """
    lines = []
    for define in params.defines:
        name, _, value = define.partition("=")
        lines.append(f"verilog_defines -D{name}={value or 1}")
    includes = "".join(f" -I{d}" for d in params.include_dirs)
    for source in params.files:
        lines.append(f"read_verilog -sv{includes} {source}")
    liberty_flags = " ".join(f"-liberty {lib}" for lib in liberty)
    lines += [
        f"hierarchy -check -top {params.top}",
        f"synth -top {params.top} -flatten",
        f"dfflibmap {liberty_flags}",
        f"abc {liberty_flags}",
        "setundef -zero",
        "splitnets",
        "opt_clean",
        f"write_verilog -noattr {netlist}",
        f"stat {liberty_flags}",
    ]
    return "\n".join(lines) + "\n"


def openroad_script(params: PpaIn, liberty: list[str], netlist: Path,
                    tech_lef: str | None, lef: list[str]) -> str:
    """Link the netlist, constrain it, and report power and slack.

    LEF is read when it is available even in `estimate` mode: some OpenROAD
    builds need cell masters to read a netlist at all. When it is absent and
    the link fails, the `@ic link_error` marker turns that into a message
    naming `tech_lef` rather than an opaque non-zero exit.
    """
    lines = ["# Generated by ic ppa. Regenerate rather than edit."]
    if tech_lef:
        lines.append(f"read_lef {tech_lef}")
    lines += [f"read_lef {path}" for path in lef]
    lines += [f"read_liberty {lib}" for lib in liberty]
    lines += [
        f"if {{[catch {{read_verilog {netlist}}} err]}} {{",
        '    puts "@ic link_error $err"',
        "    exit 1",
        "}",
        f"if {{[catch {{link_design {params.top}}} err]}} {{",
        '    puts "@ic link_error $err"',
        "    exit 1",
        "}",
    ]

    if params.sdc:
        lines.append(f"read_sdc {params.sdc}")
    elif params.clock_port:
        lines.append(
            f"create_clock -name clk -period {params.clock_period_ns} "
            f"[get_ports {params.clock_port}]"
        )
    else:
        # A virtual clock still constrains input-to-output paths, which is
        # better than no constraint; `note` on the result says register paths
        # were left unconstrained so the slack is not read as a full answer.
        lines.append(f"create_clock -name clk -period {params.clock_period_ns}")

    if params.mode == "placed":
        lines += [
            f"initialize_floorplan -site {params.site} "
            f"-utilization {params.utilization} -aspect_ratio 1.0 -core_space 2.0",
            f"place_pins -hor_layers {params.hor_layer} -ver_layers {params.ver_layer}",
            "global_placement -density 0.6",
            "estimate_parasitics -placement",
            "detailed_placement",
            "catch {check_placement}",
            # Again after legalisation: detailed placement moves cells, and the
            # parasitics that matter are the ones for where they ended up.
            "estimate_parasitics -placement",
            # Die area has no report command; it is read off the block, in
            # database units, and converted. Wrapped in `catch` because this is
            # the one place that reaches into odb's API rather than a command.
            "catch {",
            "    set block [ord::get_db_block]",
            "    set die [$block getDieArea]",
            "    set dbu [expr {1.0 * [$block getDefUnits]}]",
            "    set width [expr {[$die dx] / $dbu}]",
            "    set height [expr {[$die dy] / $dbu}]",
            '    puts "@ic die_area_um2 [expr {$width * $height}]"',
            "}",
        ]

    lines += [
        "set_power_activity -input -activity %s" % params.input_activity,
        "catch {set_propagated_clock [all_clocks]}",
        'puts "@ic section area"',
        "catch {report_design_area}",
        'puts "@ic section timing"',
        "report_worst_slack",
        "report_tns",
        'puts "@ic section power"',
        "report_power",
        "exit 0",
    ]
    return "\n".join(lines) + "\n"


# -- the backend ---------------------------------------------------------


def _from_env(name: str) -> list[str]:
    value = os.environ.get(name, "").strip()
    return [part for part in value.split(os.pathsep) if part] if value else []


def resolve_pdk(params: PpaIn) -> dict:
    """Fill unset PDK inputs from the environment, then check what is required.

    Done here rather than in the model because the environment is only
    meaningful at run time; the error is still `invalid_input`, and still names
    every missing piece at once.
    """
    liberty = params.liberty or _from_env("IC_PDK_LIBERTY")
    tech_lef = params.tech_lef or (_from_env("IC_PDK_TECH_LEF") or [None])[0]
    lef = params.lef or _from_env("IC_PDK_LEF")
    site = params.site or os.environ.get("IC_PDK_SITE") or None
    hor_layer = params.hor_layer or os.environ.get("IC_PDK_HOR_LAYER") or None
    ver_layer = params.ver_layer or os.environ.get("IC_PDK_VER_LAYER") or None

    missing = []
    if not liberty:
        missing.append("liberty (or IC_PDK_LIBERTY)")
    if params.mode == "placed":
        for label, value in (
            ("tech_lef (or IC_PDK_TECH_LEF)", tech_lef),
            ("lef (or IC_PDK_LEF)", lef),
            ("site (or IC_PDK_SITE)", site),
            ("hor_layer (or IC_PDK_HOR_LAYER)", hor_layer),
            ("ver_layer (or IC_PDK_VER_LAYER)", ver_layer),
        ):
            if not value:
                missing.append(label)
    if missing:
        raise InvalidInput(
            f"ppa mode={params.mode!r} needs a standard-cell technology",
            missing=missing,
        )
    absent = [path for path in [*liberty, *lef, tech_lef] if path and not Path(path).exists()]
    if absent:
        raise InvalidInput("ppa technology files do not exist", missing=absent)
    return {
        "liberty": liberty,
        "tech_lef": tech_lef,
        "lef": lef,
        "site": site,
        "hor_layer": hor_layer,
        "ver_layer": ver_layer,
    }


def _note(mode: str, params: PpaIn, virtual_clock: bool) -> str:
    parts = [
        "Standard-cell numbers for the named library, not the Zynq-7020; use "
        "`synth --mode full` for the board."
    ]
    if mode == "placed":
        parts.append(
            "Placed but not routed: parasitics are estimated from placement and "
            "no clock tree was built, so slack is optimistic."
        )
    else:
        parts.append(
            "Not placed or routed: no wire load at all, so slack is optimistic "
            "and switching power excludes interconnect."
        )
    parts.append(
        f"Power is estimated at {params.input_activity} input activity, not from a "
        "simulation trace; compare designs at the same activity."
    )
    if virtual_clock:
        parts.append(
            "No clock_port was given, so the clock is virtual and "
            "register-to-register paths are unconstrained."
        )
    return " ".join(parts)


def _openroad_environment(
    ctx, params: PpaIn, pdk: dict, script: Path, netlist: Path
):
    """Tell the managed Docker wrapper which host directories to bind.

    Paths retain their host absolute names inside the container, so the Tcl is
    identical for native and Docker OpenROAD. The wrapper also binds cwd and
    applies the host UID/GID before starting the pinned image.
    """
    if not os.environ.get("IC_OPENROAD_DOCKER_IMAGE"):
        return None

    directories = {str(Path(ctx.cwd).resolve()), str(script.parent.resolve()),
                   str(netlist.parent.resolve())}
    inputs = [*pdk["liberty"], *pdk["lef"], pdk["tech_lef"], params.sdc]
    for value in inputs:
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = Path(ctx.cwd) / candidate
            directories.add(str(candidate.resolve().parent))
    env = os.environ.copy()
    env["IC_OPENROAD_MOUNTS"] = json.dumps(sorted(directories))
    return env


@backend("ppa", "openroad", requires="openroad", version_cmd=["openroad", "-version"])
class OpenRoadPpa:
    def ppa(self, params: PpaIn, ctx) -> PpaOut:
        # `requires` covers OpenROAD; the mapping step needs Yosys as well, and
        # a missing one is the same kind of answerable failure.
        if shutil.which("yosys") is None:
            raise BackendUnavailable(
                "ppa backend 'openroad' also needs 'yosys' on PATH",
                backend="openroad",
                requires="yosys",
            )
        pdk = resolve_pdk(params)
        virtual_clock = not (params.sdc or params.clock_port)
        note = _note(params.mode, params, virtual_clock)
        log = ctx.run.artifacts / "ppa.log"
        netlist = ctx.run.artifacts / "netlist.v"

        # 1. Map to cells. No netlist means nothing downstream can run, so this
        #    failure returns rather than letting OpenROAD fail more obscurely.
        map_script = ctx.run.work / "map.ys"
        map_script.write_text(yosys_script(params, pdk["liberty"], netlist))
        mapped = run_process(
            ["yosys", "-s", str(map_script)],
            log_path=log,
            cwd=ctx.cwd,
            timeout_s=params.timeout_s,
        )
        map_text = mapped.text()
        area = parse_yosys_area(map_text)
        if mapped.exit_code != 0 or not netlist.exists():
            return PpaOut(
                ok=False,
                mode=params.mode,
                top=params.top,
                area=area,
                power=Power(),
                timing=PpaTiming(clock_period_ns=params.clock_period_ns),
                liberty=pdk["liberty"],
                summary=_errors(map_text) or ["mapping produced no netlist"],
                report=ctx.run.handle("ppa.log"),
                netlist=None,
                exit_code=mapped.exit_code or 1,
                duration_s=round(mapped.duration_s, 3),
                backend="openroad",
                backend_version=ctx.backend_version,
                note="Mapping failed; no power, area or timing was measured. " + note,
            )

        # 2. Analyse it. Appended to the same log: one run, one transcript.
        analysis_script = ctx.run.work / "ppa.tcl"
        analysis_script.write_text(
            openroad_script(params, pdk["liberty"], netlist, pdk["tech_lef"], pdk["lef"])
        )
        analysed = run_process(
            ["openroad", "-no_init", "-exit", str(analysis_script)],
            log_path=log,
            cwd=ctx.cwd,
            timeout_s=max(params.timeout_s - mapped.duration_s, 60.0),
            env=_openroad_environment(ctx, params, pdk, analysis_script, netlist),
            append=True,
        )
        text = analysed.text()

        area = merge_area(area, text)
        power = parse_power(text)
        timing = parse_timing(text, params.clock_period_ns)

        link_error = parse_markers(text).get("link_error")
        summary = _summary(text, area, power, timing)
        if link_error:
            summary = [
                f"OpenROAD could not link the netlist: {link_error}",
                "Supply tech_lef and lef (or IC_PDK_TECH_LEF / IC_PDK_LEF); some "
                "builds need cell masters to read a netlist.",
            ]
        elif analysed.exit_code != 0 or analysed.timed_out:
            summary = _errors(text) or summary

        return PpaOut(
            ok=(
                analysed.exit_code == 0
                and not analysed.timed_out
                and power.total_w is not None
            ),
            mode=params.mode,
            top=params.top,
            area=area,
            power=power,
            timing=timing,
            liberty=pdk["liberty"],
            summary=summary,
            report=ctx.run.handle("ppa.log"),
            netlist=ctx.run.handle("netlist.v"),
            exit_code=analysed.exit_code,
            duration_s=round(mapped.duration_s + analysed.duration_s, 3),
            backend="openroad",
            backend_version=ctx.backend_version,
            note=note,
        )


def _errors(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.startswith(("ERROR", "[ERROR", "Error:")) or "ERROR:" in line
    ][:8]


def _summary(text: str, area: Area, power: Power, timing: PpaTiming) -> list[str]:
    """The three numbers in one line each, plus whatever warned."""
    lines = []
    if area.cell_area_um2 is not None:
        cells = f", {area.cells} cells" if area.cells is not None else ""
        lines.append(f"area {area.cell_area_um2} um^2{cells}")
    if power.total_w is not None:
        lines.append(f"power {power.total_w} W total")
    if timing.wns_ns is not None:
        fmax = f", fmax {timing.fmax_mhz} MHz" if timing.fmax_mhz else ""
        lines.append(f"wns {timing.wns_ns} ns{fmax}")
    lines += [
        line.strip()
        for line in text.splitlines()
        if line.startswith(("[WARNING", "Warning:"))
    ][:4]
    return lines[:8]
