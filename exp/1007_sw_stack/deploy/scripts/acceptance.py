"""Shared notebook/host acceptance routines; preserve evidence after every stage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

PROMPT = "Explain why the sky is blue in simple words."
MAX_NEW = 24


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def memory():
    result = {}
    for name in ("meminfo", "self/status"):
        p = Path("/proc") / name
        if p.exists():
            for line in p.read_text().splitlines():
                key, _, value = line.partition(":")
                if key in ("MemAvailable", "CmaTotal", "CmaFree", "VmRSS", "VmHWM"):
                    result[key + "_kib"] = int(value.strip().split()[0])
    return result
