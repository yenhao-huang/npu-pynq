from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tpugen_types import PPA, FlowError

# ORFS metric keys and script layout can move between releases. Keep the
# validated image as the default, while allowing an existing local image.
PINNED_IMAGE = "openroad/orfs:26Q3-605-g2d29bdaf8"
DEFAULT_IMAGE = os.environ.get("TPUGEN_ORFS_IMAGE") or PINNED_IMAGE

# Formal equivalence checking is off: the image's kepler-formal binary uses
# instructions this host's CPU does not implement and dies with SIGILL. It
# checks netlist equivalence and contributes nothing to GDSII or PPA.
_DISABLED = {"LEC_CHECK": "0", "SEC_CHECK": "0"}

_SDC = """\
current_design {design}

set clk_name  core_clock
set clk_port_name {clk_port}
set clk_period {period}
set clk_io_pct 0.2

set clk_port [get_ports $clk_port_name]

create_clock -name $clk_name -period $clk_period $clk_port
set clk_io_name vclk_$clk_name
create_clock -name $clk_io_name -period $clk_period
set_clock_latency 0.070 [get_clocks $clk_name]
set_clock_latency 0.070 [get_clocks $clk_io_name]

set non_clock_inputs [all_inputs -no_clocks]

set_input_delay [expr $clk_period * $clk_io_pct] -clock $clk_io_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_io_name [all_outputs]
"""

_CONFIG = """\
export DESIGN_NAME     = {design}
export DESIGN_NICKNAME = {nickname}
export PLATFORM        = {platform}

export VERILOG_FILES        = $(sort $(wildcard /work/src/*.v))
export VERILOG_INCLUDE_DIRS = /work/src
export SDC_FILE             = /work/constraint.sdc

export CORE_UTILIZATION       = {core_utilization}
export PLACE_DENSITY_LB_ADDON = {place_density_lb_addon}
export TNS_END_PERCENT        = 100
"""


@dataclass
class ORFSConfig:
    image: str = DEFAULT_IMAGE
    platform: str = "nangate45"
    design_name: str = "systolic_array_top"
    nickname: str = "tpu_gen"
    clock_port: str = "clk"
    clock_period: float = 5.0          # ns
    core_utilization: int = 30
    place_density_lb_addon: float = 0.20
    threads: int = field(default_factory=lambda: os.cpu_count() or 2)
    timeout: int = 6 * 3600            # seconds


@dataclass
class ORFSResult:
    ppa: PPA
    gds: Path
    metrics_file: Path
    log_file: Path
    workdir: Path


def check_image(image: str = DEFAULT_IMAGE) -> None:
    """Require the selected ORFS image in the current Docker daemon."""
    if shutil.which("docker") is None:
        raise FlowError(
            "docker not found. OpenROAD is the only supported backend; "
            f"install docker and run: docker pull {image}"
        )
    probe = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        raise FlowError(
            f"Cannot use OpenROAD image {image}: {detail or 'docker image inspect failed'}"
        )


def _write_inputs(workdir: Path, cfg: ORFSConfig) -> None:
    (workdir / "constraint.sdc").write_text(
        _SDC.format(
            design=cfg.design_name,
            clk_port=cfg.clock_port,
            period=cfg.clock_period,
        ),
        encoding="utf-8",
    )
    (workdir / "config.mk").write_text(
        _CONFIG.format(
            design=cfg.design_name,
            nickname=cfg.nickname,
            platform=cfg.platform,
            core_utilization=cfg.core_utilization,
            place_density_lb_addon=cfg.place_density_lb_addon,
        ),
        encoding="utf-8",
    )


def _parse_metrics(path: Path) -> PPA:
    """Map the ORFS finish-stage metrics onto the paper's Area / WNS / Power."""
    if not path.exists():
        raise FlowError(f"OpenROAD wrote no final metrics at {path}")
    data = json.loads(path.read_text())

    required = {
        "area": "finish__design__instance__area",
        "wns": "finish__timing__setup__ws",
        "total_power": "finish__power__total",
    }
    missing = [k for k in required.values() if k not in data]
    if missing:
        raise FlowError(
            f"OpenROAD metrics {path} is missing {missing}; "
            "the pinned ORFS version may have changed its key names"
        )
    if data.get("finish__flow__errors__count", 0):
        raise FlowError(
            f"OpenROAD reported {data['finish__flow__errors__count']} errors; "
            f"see {path.parent}"
        )

    return PPA(
        area=float(data[required["area"]]),
        wns=float(data[required["wns"]]),
        total_power=float(data[required["total_power"]]),
        source="openroad",
        raw={
            "die_area": data.get("finish__design__die__area"),
            "core_area": data.get("finish__design__core__area"),
            "instance_count": data.get("finish__design__instance__count"),
            "utilization": data.get("finish__design__instance__utilization"),
            "tns": data.get("finish__timing__setup__tns"),
            "fmax": data.get("finish__timing__fmax"),
            "power_internal": data.get("finish__power__internal__total"),
            "power_switching": data.get("finish__power__switching__total"),
            "power_leakage": data.get("finish__power__leakage__total"),
            "setup_violations": data.get("finish__timing__drv__setup_violation_count"),
        },
    )


def run_orfs(workdir: Path, cfg: ORFSConfig | None = None) -> ORFSResult:
    """Run the full ORFS flow on workdir/src and return GDSII plus PPA.

    workdir/src must already hold the .v sources and options_definitions.vh.
    """
    cfg = cfg or ORFSConfig()
    check_image(cfg.image)

    workdir = workdir.resolve()
    src = workdir / "src"
    if not any(src.glob("*.v")):
        raise FlowError(f"no Verilog sources staged in {src}")
    _write_inputs(workdir, cfg)

    make_vars = " ".join(
        [
            "DESIGN_CONFIG=/work/config.mk",
            "WORK_HOME=/work",
            f"NPROC={cfg.threads}",
            *[f"{k}={v}" for k, v in _DISABLED.items()],
        ]
    )
    inner = (
        "source /OpenROAD-flow-scripts/env.sh >/dev/null && "
        f"cd /OpenROAD-flow-scripts/flow && make {make_vars}"
    )
    cmd = ["docker", "run", "--rm"]
    if os.name != "nt":
        cmd.extend(["-u", f"{os.getuid()}:{os.getgid()}"])
    cmd.extend(["-v", f"{workdir}:/work", cfg.image, "bash", "-lc", inner])

    log_file = workdir / "orfs.log"
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=cfg.timeout
        )
    except subprocess.TimeoutExpired as exc:
        log_file.write_text((exc.stdout or "") + (exc.stderr or ""), encoding="utf-8")
        raise FlowError(
            f"OpenROAD timed out after {cfg.timeout}s; see {log_file}"
        ) from exc

    log_file.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        raise FlowError(
            f"OpenROAD flow failed (exit {proc.returncode}); full log at "
            f"{log_file}\n--- tail ---\n{tail}"
        )

    base = workdir / "results" / cfg.platform / cfg.nickname / "base"
    gds = base / "6_final.gds"
    if not gds.exists():
        raise FlowError(f"OpenROAD produced no GDSII at {gds}; see {log_file}")

    metrics_file = workdir / "logs" / cfg.platform / cfg.nickname / "base" / "6_report.json"
    return ORFSResult(
        ppa=_parse_metrics(metrics_file),
        gds=gds,
        metrics_file=metrics_file,
        log_file=log_file,
        workdir=workdir,
    )
