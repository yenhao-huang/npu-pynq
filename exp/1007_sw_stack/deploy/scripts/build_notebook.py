"""Create the output-free sequential physical-board acceptance notebook."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""# SmolLM2 on PYNQ-Z1 — physical acceptance
Run the cells in order in the existing root PYNQ Python kernel. This loads the FPGA.
All evidence is saved under `runs/<run-id>/`. The source notebook has no outputs;
the executed copy is the physical evidence. Run through `execute_notebook.py` for
a per-cell deadline and guaranteed partial-notebook saving. After a timeout, reload
the overlay before retrying; do not continue with potentially active hardware.
""")
code('''
import html, json, os, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown
if os.environ.get("NPU_DEPLOY"):
    DEPLOY = Path(os.environ["NPU_DEPLOY"]).expanduser().resolve()
else:
    # Jupyter starts a kernel in the opened notebook's directory, including runs/.
    candidates = []
    for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        candidates.extend((parent, parent / "exp/1007_sw_stack/deploy"))
    DEPLOY = next((p for p in candidates if (p / "scripts/acceptance.py").is_file()
                   and (p / "packages/smollm/manifest.json").is_file()), None)
    if DEPLOY is None:
        raise FileNotFoundError("Cannot locate deploy/scripts and model package from this notebook directory.")
assert (DEPLOY / "scripts/acceptance.py").is_file(), f"Invalid deployment directory: {DEPLOY}"
REPO = Path(os.environ.get("NPU_REPO", DEPLOY.parents[2])).resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DEPLOY / "scripts"))
from acceptance import package_info, memory, save, prefill, generate, compare, hardware_gemm, MAX_NEW
def metrics_table(rows):
    body = "".join("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in row) + "</tr>" for row in rows)
    display(HTML("<table><thead><tr><th>Metric</th><th>Value</th><th>Unit</th></tr></thead><tbody>" + body + "</tbody></table>"))
RUN = DEPLOY / "runs" / os.environ.get("NPU_RUN_ID", time.strftime("manual-%Y%m%d-%H%M%S"))
RUN.mkdir(parents=True, exist_ok=True)
PACKAGE = DEPLOY / "packages/smollm"
REFERENCE = DEPLOY / "data/reference"
assert os.geteuid() == 0, "Use the existing authorized root PYNQ kernel."
info = package_info(PACKAGE)
save(RUN / "package.json", info)
display(info)
''')
md("## 1. Load the verified overlay and prove integer GEMM correctness")
code('''
from pynq import Overlay
overlay = Overlay(str(DEPLOY / "hardware/overlay/npu_matrix.bit"))
gemm_result = hardware_gemm(overlay, RUN / "gemm.json")
display(gemm_result)
metrics_table([(str(c["shape"]), "PASS" if c["exact"] else "FAIL", f'{c["jobs"]} jobs / {c["cycles"]} cycles') for c in gemm_result["cases"]])
''')
md("## 2. Load the complete model using streaming weights")
code('''
from src.inference.engine import LLMEngine
start = time.monotonic()
engine = LLMEngine(PACKAGE, backend="pynq", overlay=overlay, stream_weights=True)
load_result = {"seconds": time.monotonic() - start, "memory": memory(), "stream_weights": engine.model.stream_weights}
save(RUN / "load.json", load_result)
display(load_result)
metrics_table([("Full model load", round(load_result["seconds"], 3), "seconds"),
               ("Mapped weights", round(info["weights_bytes"] / 2**20, 2), "MiB"),
               ("KV state", round(info["state_bytes"] / 2**20, 2), "MiB"),
               ("CMA I/O + stage + programs", round((info["io_bytes"] + info["stage_bytes"] + info["program_bytes"]) / 2**20, 2), "MiB")])
''')
md("## 3. Prefill, inspect logits, and execute one decode token")
code('''
prefill_result = prefill(engine, RUN / "smoke")
token = int(np.argmax(engine.logits[0]))
start = time.monotonic()
smoke_logits = engine.decode(token).copy()
smoke = {"first_token": token, "text": engine.tokenizer.decode([token]),
         "seconds": time.monotonic() - start, "finite": bool(np.isfinite(smoke_logits).all()), "runtime": engine.model.stats()}
save(RUN / "smoke/decode.json", smoke)
assert smoke["finite"] and smoke["runtime"]["error"] == 0
display(prefill_result, smoke)
top = np.argsort(smoke_logits)[-10:][::-1]
plt.figure(figsize=(10, 3))
plt.bar(range(10), smoke_logits[top])
plt.xticks(range(10), [engine.tokenizer.decode([int(t)]) for t in top], rotation=30)
plt.title("Top logits after the first decode token")
plt.show()
''')
md("## 4. Generate 24 greedy tokens and show measured performance\nThe trace always records 24 samples, including any EOS positions. It uses 23 decode calls after prefill.")
code('''
view = display(Markdown("Generating…"), display_id=True)
result = generate(engine, RUN, display_token=lambda text: view.update(Markdown(text)))
display(result)
metrics_table([("Prompt", len(result["prompt_ids"]), "tokens"), ("Generated", len(result["ids"]), "tokens"),
               ("Prefill", round(result["prefill_seconds"], 3), "seconds"),
               ("Decode throughput", round(result["decode_tokens_per_second"], 3), "tokens/s"),
               ("Physical NPU jobs", result["npu"]["jobs"], "jobs"),
               ("Physical NPU cycles", result["npu"]["cycles"], "cycles")])
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
axes[0].plot(range(1, len(result["per_token_seconds"]) + 1), result["per_token_seconds"], marker="o")
axes[0].set(xlabel="Decode step", ylabel="Seconds", title="Physical decode latency")
axes[1].bar(["Prefill", "Decode"], [result["prefill_seconds"], result["decode_seconds"]])
axes[1].set(ylabel="Seconds", title="Generation timing")
plt.tight_layout(); plt.show()
snapshots = [info["memory"], load_result["memory"], result["memory"]]
labels = ["Before overlay", "Model loaded", "After generation"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
axes[0].bar(labels, [s.get("VmRSS_kib", 0) / 1024 for s in snapshots])
axes[0].set(ylabel="MiB", title="Process resident memory")
axes[1].bar(labels, [s.get("CmaFree_kib", 0) / 1024 for s in snapshots])
axes[1].set(ylabel="MiB", title="Free CMA pages (snapshot)")
plt.tight_layout(); plt.show()
''')
md("## 5. Compare the identical quantized host package\nIf greedy token paths diverge, teacher forcing isolates arithmetic differences using the same input history.")
code('''
reference_info = json.loads((REFERENCE / "package.json").read_text())
assert info["hashes"] == reference_info["hashes"], "Reference package differs."
comparison = compare(RUN / "generation.npz", REFERENCE / "generation.npz")
save(RUN / "comparison.json", comparison)
display(comparison)
if not comparison["token_ids_equal"]:
    teacher = np.load(REFERENCE / "generation.npz")["ids"].tolist()
    generate(engine, RUN, "teacher", teacher_ids=teacher)
    teacher_comparison = compare(RUN / "teacher.npz", REFERENCE / "generation.npz")
    save(RUN / "teacher-comparison.json", teacher_comparison)
    display(teacher_comparison)
plt.figure(figsize=(9, 3))
plt.plot(comparison["per_step_max_abs_error"], marker="o")
plt.ylabel("Maximum absolute logit error"); plt.xlabel("Token step")
plt.title("Board versus host simulator"); plt.show()
''')
md("## 6. Repeat from reset and report the acceptance gates")
code('''
repeat = generate(engine, RUN, "repeat")
repeat_comparison = compare(RUN / "repeat.npz", RUN / "generation.npz")
save(RUN / "repeat-comparison.json", repeat_comparison)
verdict = {"gemm": gemm_result["pass"], "tokens": len(result["ids"]) == MAX_NEW,
           "hardware_active": result["npu"]["jobs"] > 0 and result["npu"]["cycles"] > 0,
           "reference_tokens": comparison["token_ids_equal"], "reference_logits": comparison["logits_allclose"],
           "repeat_tokens": repeat_comparison["token_ids_equal"], "repeat_logits": repeat_comparison["logits_allclose"],
           "runtime_error_free": result["runtime"]["error"] == repeat["runtime"]["error"] == 0}
verdict["pass"] = all(verdict.values())
save(RUN / "verdict.json", verdict)
display(verdict, repeat_comparison)
assert verdict["pass"], "Acceptance is incomplete; inspect the saved evidence."
''')
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3 (PYNQ)", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.validate(nb)
nbf.write(nb, root / "smollm_board.ipynb")
