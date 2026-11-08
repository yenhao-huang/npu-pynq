"""Run the same 24-token quantized trace on the compiled host ISA simulator."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from acceptance import generate, package_info, save
from src.inference.engine import LLMEngine

p = argparse.ArgumentParser()
p.add_argument("--package", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out)
save(out / "package.json", package_info(a.package))
engine = LLMEngine(a.package, backend="sim", stream_weights=True)
result = generate(engine, out, display_token=lambda text: print(text, flush=True))
print(result, flush=True)
