"""Compile the pinned checkpoint ahead of time for host and PYNQ."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from src.export.cli import main
    raise SystemExit(main(["export", "resnet18", "--source",
                          str(ROOT / "examples/resnet18/model/resnet18-f37072fd.pth"),
                          "--out", str(ROOT / "examples/resnet18/model/aot"), *sys.argv[1:]]))
