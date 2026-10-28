"""Independently audit the saved physical evidence before claiming acceptance."""
import argparse
import base64
import json
from pathlib import Path
import sys

import nbformat
import numpy as np

from acceptance import compare, package_info, save, sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))
p = argparse.ArgumentParser()
p.add_argument("--run-id", default="board-20261009-04")
a = p.parse_args()
run = ROOT / "runs" / a.run_id
load = lambda path: json.loads(Path(path).read_text())
reference = ROOT / "data/reference"
manifest = load(ROOT / "packages/smollm/manifest.json")
generation, repeat = load(run / "generation.json"), load(run / "repeat.json")
info = package_info(ROOT / "packages/smollm")
assert info["hashes"] == load(reference / "package.json")["hashes"] == load(run / "package.json")["hashes"]
assert manifest["meta"]["config"]["layers"] == 30 and manifest["meta"]["options"]["layers"] is None
assert manifest["meta"]["options"]["awq"] and manifest["meta"]["options"]["bits"] == 8
overlay = load(ROOT / "hardware/overlay/npu_matrix.manifest.json")
assert all(sha256(ROOT / "hardware/overlay" / overlay[k]["name"]) == overlay[k]["sha256"] for k in ("bit", "hwh"))
prov = load(reference / "provenance.json")
assert prov["backend"] == 0 and prov["mmio"] is None and prov["uid"] != 0
assert prov["library_sha256"] == sha256(ROOT / "packages/smollm/pynq/model.so")
assert prov["source_manifest_sha256"] == sha256(ROOT / "packages/smollm/manifest.json")
for result in (generation, repeat):
    assert len(result["ids"]) == 24 and len(result["prompt_ids"]) == 19 and result["decode_steps"] == 23
    assert result["npu"]["jobs"] > 0 and result["npu"]["cycles"] > 0 and result["runtime"]["error"] == 0
    assert not result["teacher_forced"] and not result["eos_positions"]
    assert result["prompt_ids"] == load(reference / "generation.json")["prompt_ids"]
comparison = compare(run / "generation.npz", reference / "generation.npz")
repeat_check = compare(run / "repeat.npz", run / "generation.npz")
assert comparison["token_ids_equal"] and comparison["logits_allclose"]
assert repeat_check["token_ids_equal"] and repeat_check["logits_allclose"]
for name, result in (("generation", generation), ("repeat", repeat)):
    arrays = np.load(run / (name + ".npz"))
    assert arrays["logits"].shape == (24, 49152) and np.isfinite(arrays["logits"]).all()
    assert arrays["ids"].tolist() == result["ids"]
assert np.allclose(np.load(run / "prefill.npz")["logits"], np.load(reference / "prefill.npz")["logits"], rtol=1e-3, atol=1e-3)
gemm = load(run / "gemm.json")
assert len(gemm["cases"]) == 5 and all(c["exact"] and c["jobs"] > 0 and c["cycles"] > 0 for c in gemm["cases"])
from src.vendors.npu_pynq.isa.layout import plan_gemm
assert all(c["jobs"] == plan_gemm(*c["shape"]).jobs for c in gemm["cases"])
smoke = load(run / "smoke/decode.json")
assert smoke["finite"] and smoke["runtime"]["error"] == 0
nb = nbformat.read(run / "smollm_board.executed.ipynb", as_version=4)
nbformat.validate(nb)
codes = [c for c in nb.cells if c.cell_type == "code"]
assert len(codes) == 7 and [c.execution_count for c in codes] == list(range(1, 8))
assert not any(o.output_type == "error" for c in codes for o in c.outputs)
source = nbformat.read(ROOT / "smollm_board.ipynb", as_version=4)
assert [c.source for c in source.cells] == [c.source for c in nb.cells]
assert all(not c.outputs and c.execution_count is None for c in source.cells if c.cell_type == "code")
figures = run / "figures"
figures.mkdir(exist_ok=True)
count = 0
for c in codes:
    for o in c.outputs:
        if "image/png" in o.get("data", {}):
            count += 1
            (figures / f"figure-{count}.png").write_bytes(base64.b64decode(o.data["image/png"]))
assert count >= 4
tables = sum("<table>" in str(o.get("data", {}).get("text/html", "")) for c in codes for o in c.outputs)
assert tables >= 3
for name, tests in (("baseline-compiler", 4), ("baseline-operators", 6)):
    log = (ROOT / "runs/mac-tests" / (name + ".log")).read_text(encoding="utf-8-sig")
    assert f"Ran {tests} tests" in log and "\nOK" in log and "skipped" not in log and "FAILED" not in log
verdict = load(run / "verdict.json")
assert all(verdict.values())
audit = {"pass": True, "run_id": a.run_id, "reference_comparison": comparison,
         "repeat_comparison": repeat_check, "executed_code_cells": len(codes),
         "figures": count, "tables": tables, "full_model_layers": 30,
         "notebook_sha256": sha256(run / "smollm_board.executed.ipynb")}
if a.run_id == "board-20261009-04":
    assert "NPU_DEPLOY/NPU_REPO unset; notebook kernel cwd=" in (run / "runner.log").read_text()
    audit["nested_jupyter_cwd_without_path_environment"] = True
save(run / "audit.json", audit)
print(json.dumps(audit, indent=2))
