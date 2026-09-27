"""Turning a simulation log into the few lines that explain the verdict.

Shared by every sim backend: what counts as a failure is a property of how
testbenches in this repository report, not of which simulator ran them.
"""

from __future__ import annotations

import re

FAIL = re.compile(
    r"\b(FAIL(?:ED|URE)?|ERROR|MISMATCH|Assertion failed|\$error|\$fatal|UVM_ERROR|UVM_FATAL)\b",
    re.IGNORECASE,
)
PASS = re.compile(r"\bPASS(?:ED)?\b", re.IGNORECASE)
# Verilator and Icarus compile diagnostics, kept even when nothing else matched.
BUILD_DIAG = re.compile(r"^%(Error|Warning)|^[^:\s][^:]*:\d+:.*error", re.IGNORECASE)


def summarize(text: str, limit: int) -> tuple[list[str], bool, int, int]:
    """Return (lines, truncated, pass_count, fail_count).

    Failure lines come first and are never dropped in favour of a pass line;
    an agent reading a truncated summary must still see what broke.
    """
    fails: list[str] = []
    passes = 0
    diagnostics: list[str] = []
    tail: list[str] = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if FAIL.search(line):
            fails.append(line)
        elif PASS.search(line):
            passes += 1
            if len(diagnostics) < 4:
                diagnostics.append(line)
        elif BUILD_DIAG.search(line):
            diagnostics.append(line)
        tail.append(line)

    selected = fails[:limit]
    room = limit - len(selected)
    if room > 0:
        selected += [d for d in diagnostics if d not in selected][:room]
    room = limit - len(selected)
    if room > 0 and not selected:
        selected = tail[-room:]
    truncated = len(fails) > limit or len(tail) > len(selected)
    return selected, truncated, passes, len(fails)
