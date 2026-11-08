"""Temporarily hide optional JIT; run active suites; restore it even on failure."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]


def main():
    optional = (ROOT/'src/jit').resolve()
    hidden = (ROOT/'build/jit-isolation').resolve()
    if not optional.is_relative_to(ROOT) or not hidden.is_relative_to(ROOT):
        raise RuntimeError('Isolation paths escape the worktree')
    if not optional.is_dir() or hidden.exists():
        raise RuntimeError('Expected a JIT source and an unused isolation path')
    hidden.parent.mkdir(exist_ok=True)
    env = dict(os.environ)
    tmp = ROOT/'build/tmp'
    tmp.mkdir(exist_ok=True)
    env.update(TMP=str(tmp),TEMP=str(tmp))
    suites = ['src/test/tests','examples/resnet18/tests','examples/matrix-multiplication/tests']
    optional.rename(hidden)
    try:
        for suite in suites:
            result = subprocess.run([sys.executable,'-m','unittest','discover','-s',suite,'-q'],
                                    cwd=ROOT,env=env)
            if result.returncode:
                return result.returncode
    finally:
        hidden.rename(optional)
    print('PASS: active suites execute with src/jit physically absent')
    return 0


if __name__ == '__main__': raise SystemExit(main())
