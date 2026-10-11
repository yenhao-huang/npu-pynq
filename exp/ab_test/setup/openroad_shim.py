#!/usr/bin/env python3
"""Host shim: run OpenROAD in the long-lived `ic-openroad` container.

The pinned ORFS image does not put openroad on PATH, and Docker Desktop's
bind mounts (virtiofs) leave OpenROAD stuck in uninterruptible I/O under
concurrent use. So the container has no host mounts: the script and every
file it names are streamed in through `docker exec -i tar`, paths are
rewritten, and the report comes back on stdout. The PDK is copied into the
container once, at /pdk. Calls are serialized and bounded.
"""
import fcntl, io, os, re, subprocess, sys, tarfile, uuid
from pathlib import Path

DOCKER = "/Applications/Docker.app/Contents/Resources/bin/docker"
BIN = "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
PDK_HOST = str(Path.home() / "pdk")
TIMEOUT = 180
env = dict(os.environ, PATH="/Applications/Docker.app/Contents/Resources/bin:" + os.environ.get("PATH", ""))

args = sys.argv[1:]
script = next((a for a in args if a.endswith(".tcl") and os.path.isfile(a)), None)
if script is None:  # -version and the like
    sys.exit(subprocess.run([DOCKER, "exec", "ic-openroad", "timeout", "-s", "KILL", "60", BIN, *args],
                            stdin=subprocess.DEVNULL, env=env).returncode)

job = f"/job/{uuid.uuid4().hex}"
text = Path(script).read_text()
files = {}
def remap(m):
    path = m.group(0)
    if path.startswith(PDK_HOST + "/"):
        return "/pdk/" + path[len(PDK_HOST) + 1:]
    real = os.path.realpath(path)
    if os.path.isfile(real):
        name = f"f{len(files)}_{os.path.basename(real)}"
        files.setdefault(real, name)
        return f"{job}/{files[real]}"
    return path
text = re.sub(r"(?<![\w.])/[^\s\]\[{}\"';]+", remap, text)

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tar:
    for real, name in files.items():
        tar.add(real, arcname=name)
    data = text.encode()
    info = tarfile.TarInfo("run.tcl"); info.size = len(data)
    tar.addfile(info, io.BytesIO(data))

with open("/tmp/ic-openroad.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    subprocess.run([DOCKER, "exec", "-i", "ic-openroad", "sh", "-c", f"mkdir -p {job} && tar -x -C {job}"],
                   input=buf.getvalue(), check=True, env=env, timeout=60)
    rest = [a for a in args if a != script]
    try:
        rc = subprocess.run([DOCKER, "exec", "-w", job, "ic-openroad", "timeout", "-s", "KILL", str(TIMEOUT),
                             BIN, *rest, f"{job}/run.tcl"], stdin=subprocess.DEVNULL, env=env,
                            timeout=TIMEOUT + 30).returncode
    except subprocess.TimeoutExpired:
        print("@ic link_error openroad timed out", flush=True)
        rc = 124
    subprocess.run([DOCKER, "exec", "ic-openroad", "rm", "-rf", job], stdin=subprocess.DEVNULL, env=env, timeout=60)
sys.exit(rc)
