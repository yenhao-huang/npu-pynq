"""Execute with bounded cells and save the notebook even on failure."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

p = argparse.ArgumentParser()
p.add_argument("--deploy", required=True)
p.add_argument("--repo", required=True)
p.add_argument("--run-id", required=True)
p.add_argument("--cell-timeout", type=int, default=900)
a = p.parse_args()
deploy = Path(a.deploy).resolve()
run = deploy / "runs" / a.run_id
run.mkdir(parents=True, exist_ok=True)
os.environ.update(NPU_DEPLOY=str(deploy), NPU_REPO=str(Path(a.repo).resolve()), NPU_RUN_ID=a.run_id)
nb = nbformat.read(deploy / "smollm_board.ipynb", as_version=4)
try:
    NotebookClient(nb, timeout=a.cell_timeout, kernel_name="python3", resources={"metadata": {"path": str(deploy)}}).execute()
finally:
    nbformat.write(nb, run / "smollm_board.executed.ipynb")
