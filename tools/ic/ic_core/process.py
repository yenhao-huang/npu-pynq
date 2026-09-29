"""Subprocess helper.

stdout and stderr go straight to files under the run directory. A Verilator
build log or a Vivado transcript is megabytes; none of it belongs in memory,
and none of it belongs in an agent's context.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    argv: list[str]
    exit_code: int
    duration_s: float
    log_path: Path
    timed_out: bool = False

    def text(self, limit_bytes: int = 4_000_000) -> str:
        """Read the captured log back, capped so a runaway log cannot blow up."""
        if not self.log_path.exists():
            return ""
        data = self.log_path.read_bytes()[:limit_bytes]
        return data.decode("utf-8", errors="replace")


def run(
    argv: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    timeout_s: float | None = None,
    env: dict | None = None,
    append: bool = False,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    mode = "ab" if append else "wb"
    with open(log_path, mode) as sink:
        sink.write(f"$ {' '.join(argv)}\n".encode())
        sink.flush()
        job = _WindowsJob() if os.name == 'nt' else None
        try:
            proc = subprocess.Popen(
                argv, stdout=sink, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                start_new_session=(os.name != 'nt'),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
            )
        except OSError as exc:
            if job is not None: job.close()
            if not isinstance(exc,FileNotFoundError): raise
            sink.write(f"{exc}\n".encode())
            return CommandResult(argv, 127, time.monotonic() - started, log_path)
        if job is not None:
            try:
                job.assign(proc)
            except OSError:
                proc.terminate(); proc.wait(); job.close()
                raise
        try:
            exit_code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill only the process tree created by this run. Parent-only
            # termination leaves compiler/solver children consuming resources.
            if job is not None:
                job.terminate()
            else:
                try: os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError: pass
            try:
                exit_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    proc.kill()
                else:
                    try: os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                exit_code = proc.wait()
            if os.name != 'nt':
                # A child may ignore TERM after its parent has already exited.
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            sink.write(f"\n[ic] timed out after {timeout_s}s\n".encode())
        finally:
            if job is not None: job.close()
    return CommandResult(
        argv, exit_code, time.monotonic() - started, log_path, timed_out
    )


class _WindowsJob:
    """Owned Windows job; closing its handle terminates attached descendants."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        class Limits(ctypes.Structure):
            _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
                      ('flags',wintypes.DWORD),('min_working',ctypes.c_size_t),('max_working',ctypes.c_size_t),
                      ('active',wintypes.DWORD),('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
        class Extended(ctypes.Structure):
            _fields_=[('limits',Limits),('io',ctypes.c_ulonglong*6),('process_memory',ctypes.c_size_t),
                      ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        self.ctypes=ctypes
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        for name,args,result in [
            ('CreateJobObjectW',[ctypes.c_void_p,wintypes.LPCWSTR],wintypes.HANDLE),
            ('SetInformationJobObject',[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD],wintypes.BOOL),
            ('AssignProcessToJobObject',[wintypes.HANDLE,wintypes.HANDLE],wintypes.BOOL),
            ('TerminateJobObject',[wintypes.HANDLE,wintypes.UINT],wintypes.BOOL),
            ('CloseHandle',[wintypes.HANDLE],wintypes.BOOL)]:
            function=getattr(self.api,name);function.argtypes=args;function.restype=result
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        limits=Extended(); limits.limits.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error=ctypes.WinError(ctypes.get_last_error());self.close();raise error

    def assign(self,proc):
        if not self.api.AssignProcessToJobObject(self.handle,int(proc._handle)):
            if proc.poll() is not None: return
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle,1):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle);self.handle=None
