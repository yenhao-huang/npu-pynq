"""Where the upstream TPU-Gen assets live. Nothing here is copied."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_CODE = ROOT / "src_code"

RTL_DIR = SRC_CODE / "rtl-20250603T193300Z-1-001" / "rtl"
OPTIONS_VH = RTL_DIR / "options_definitions.vh"
TOP_MODULE = "systolic_array_top"

TRAIN_DATASET = SRC_CODE / "beta_train_all.json"
TEST_DATASET = SRC_CODE / "beta_test_all.json"

RUNS = ROOT / "runs"


def check() -> None:
    from tpugen_types import FlowError

    for p in (RTL_DIR, OPTIONS_VH, TRAIN_DATASET):
        if not p.exists():
            raise FlowError(f"missing upstream asset: {p}")
