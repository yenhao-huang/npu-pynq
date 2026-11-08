#!/bin/bash
# Record what the board offers the software stack: memory, CMA, CPU features,
# toolchain, Python packages, and whether pynq can allocate uncached buffers.
set -u
cd "$1"
out=results/probe.txt
{
  echo "== uname"; uname -a
  echo "== os"; cat /etc/os-release | head -3
  echo "== cpu"; grep -m1 -E 'Features' /proc/cpuinfo; grep -c processor /proc/cpuinfo; grep -m1 -i 'bogomips' /proc/cpuinfo
  echo "== meminfo"; grep -E 'MemTotal|MemFree|MemAvailable|Cma|SwapTotal' /proc/meminfo
  echo "== disk"; df -h /home/xilinx /tmp | cat
  echo "== tools"; for t in gcc cc ld as clang python3 git; do printf '%s: ' $t; command -v $t || echo missing; done
  gcc --version 2>&1 | head -1
  echo "== libs"; ls /lib/arm-linux-gnueabihf/libgcc_s.so.1 /lib/arm-linux-gnueabihf/libm.so.6 2>&1
  echo "== cmdline"; cat /proc/cmdline
  echo "== dma heaps"; ls /dev/dma_heap 2>&1; ls /dev | grep -E 'xlnk|udmabuf|uio|zocl' | head
} > "$out" 2>&1
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 - >> "$out" 2>&1 <<'PY'
import sys, inspect, time
print("== python", sys.version)
import numpy as np
print("numpy", np.__version__)
import pynq
print("pynq", pynq.__version__)
from pynq import allocate
print("allocate signature", inspect.signature(allocate))
for kw in ({}, {"cacheable": False}):
    try:
        t = time.time()
        b = allocate(shape=(16 << 20,), dtype=np.uint8, **kw)
        b[:] = 1
        dt = time.time() - t
        t = time.time(); s = int(b[::4096].sum()); r = time.time() - t
        t = time.time(); c = np.array(b); rd = time.time() - t
        print("allocate", kw, "ok phys=0x%x" % b.physical_address, "fill+alloc %.3fs" % dt, "copy 16MiB %.3fs" % rd)
        b.freebuffer()
    except Exception as e:
        print("allocate", kw, "failed:", repr(e))
big = []
try:
    for i in range(64):
        big.append(allocate(shape=(16 << 20,), dtype=np.uint8))
except Exception as e:
    print("max CMA allocation: %d x 16 MiB (%r)" % (len(big), e))
else:
    print("allocated", len(big), "x 16 MiB")
for b in big:
    b.freebuffer()
for mod in ("scipy", "torch", "safetensors", "tokenizers", "PIL"):
    try:
        m = __import__(mod); print(mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print(mod, "missing")
PY
cat "$out"
