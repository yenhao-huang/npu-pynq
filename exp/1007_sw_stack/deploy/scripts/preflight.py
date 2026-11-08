"""Read-only board readiness and ARM-library probe; no FPGA programming."""
import ctypes
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY.parents[2]))
from acceptance import memory, package_info, save

package = DEPLOY / "packages/smollm"
library = package / "pynq/model.so"
header = library.read_bytes()[:52]
assert header[:4] == b"\x7fELF" and header[4:6] == b"\x01\x01"
machine = struct.unpack_from("<H", header, 18)[0]
flags = struct.unpack_from("<I", header, 36)[0]
assert machine == 40 and flags & 0x400, (machine, flags)
for dep in ("libm.so.6", "libgcc_s.so.1"):
    ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
loaded = ctypes.CDLL(str(library))
assert getattr(loaded, "_mlir_ciface_prefill") and getattr(loaded, "_mlir_ciface_decode")
report = {"platform": platform.platform(), "python": sys.version,
          "executable": sys.executable, "uid": os.geteuid(), "memory": memory(),
          "disk_free_bytes": shutil.disk_usage(DEPLOY).free,
          "versions": {p: importlib.metadata.version(p) for p in ("pynq", "numpy", "nbclient", "nbformat", "matplotlib")},
          "elf": {"machine": machine, "flags": hex(flags), "loaded": True},
          "sudo_noninteractive": subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0,
          "package": package_info(package)}
save(DEPLOY / "runs/preflight/board.json", report)
print(report)
