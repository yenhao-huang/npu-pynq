"""Experiment entry point using the shared source implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import json
from src.compiler.export import verify_export
from acceptance import save
root = Path(__file__).resolve().parents[1]
result = verify_export(root / "packages")
save(root / "runs/official-inspect/export-verification.json", result)
print(json.dumps(result, indent=2))
