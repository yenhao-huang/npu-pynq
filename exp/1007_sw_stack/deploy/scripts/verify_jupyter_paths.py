"""Reproduce Jupyter's nested notebook cwd without injected deployment variables."""
import argparse
import os
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

p = argparse.ArgumentParser()
p.add_argument("--run-id", default="board-20261009-04")
a = p.parse_args()
deploy = Path(__file__).resolve().parents[1]
run = deploy / "runs" / a.run_id
run.mkdir(parents=True, exist_ok=True)
for name in ("NPU_DEPLOY", "NPU_REPO", "NPU_RUN_ID"):
    os.environ.pop(name, None)
# Only label the saved evidence; directory resolution gets no assistance.
os.environ["NPU_RUN_ID"] = run.name
nb = nbformat.read(deploy / "smollm_board.ipynb", as_version=4)
print(f"NPU_DEPLOY/NPU_REPO unset; notebook kernel cwd={run}", flush=True)
try:
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(run)}}).execute()
finally:
    nbformat.write(nb, run / "smollm_board.executed.ipynb")
