"""Explicit runtime configuration; importing this module performs no I/O."""
from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Config:
    problems: Path = ROOT / "problems"
    repo: Path = Path(os.environ.get("AB_TOOL_REPO", ROOT.parents[2] / "worktrees" / "npu-dev-ab"))
    output: Path = ROOT / "results_modular"
    model: str = "sonnet"
    backend: str = "claude"
    shell: bool = False
    # PATH seen by shell commands when shell=True: system EDA tools, never the ic venv.
    shell_path: str = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    effort: str = "medium"
    repetitions: int = 3
    budget: str = "3.00"
    rounds: int = 1
    timeout: int = 1800
    period_ns: float = 10.0
    activity: float = 0.1

    @property
    def venv(self):
        return self.repo / ".venv" / "bin"

    @property
    def tool_env(self):
        home = str(Path.home())
        return {
            "PATH": ":".join([f"{home}/.local/bin", str(self.venv),
                "/Applications/Docker.app/Contents/Resources/bin", "/opt/homebrew/bin",
                "/usr/bin", "/bin", "/usr/sbin", "/sbin"]),
            "IC_PDK_LIBERTY": f"{home}/pdk/NangateOpenCellLibrary_typical.lib",
            "IC_PDK_TECH_LEF": f"{home}/pdk/lef/NangateOpenCellLibrary.tech.lef",
            "IC_PDK_LEF": f"{home}/pdk/lef/NangateOpenCellLibrary.macro.mod.lef",
            "CPATH": "/opt/homebrew/include", "LIBRARY_PATH": "/opt/homebrew/lib", "HOME": home,
        }
