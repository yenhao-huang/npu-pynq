"""Compatibility import for historical experiment notebooks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.inference.graph import *
