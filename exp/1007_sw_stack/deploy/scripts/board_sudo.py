"""Use the user's PYNQ_PASSWORD from .env through sudo stdin, never argv/logs."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--env-file", required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
password = None
for line in Path(a.env_file).read_text(encoding="utf-8-sig").splitlines():
    key, sep, value = line.strip().removeprefix("export ").partition("=")
    if sep and key.strip() == "PYNQ_PASSWORD":
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        password = value
        break
if not password or "\n" in password or "\r" in password:
    sys.exit("PYNQ_PASSWORD is missing or invalid; value was not printed.")
if not a.command:
    sys.exit("A sudo command is required.")
command = "sudo -S -p '' -- " + shlex.join(a.command)
process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                         "-o", "StrictHostKeyChecking=yes", "xilinx@192.168.2.99", command],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
process.stdin.write((password + "\n").encode())
process.stdin.close()
# Stream complete lines so long notebook runs expose progress, while redacting
# defensively if a remote tool unexpectedly reflects stdin.
secret = password.encode()
for line in process.stdout:
    sys.stdout.buffer.write(line.replace(secret, b"[REDACTED]"))
    sys.stdout.buffer.flush()
sys.exit(process.wait())
