"""Experiment entry point using the shared source implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.compiler.export import fetch
fetch(Path(__file__).resolve().parents[1] / "packages")
