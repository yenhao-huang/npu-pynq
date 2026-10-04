"""A small Verilog preprocessor: enough to resolve `ifdef-gated dependencies.

The TPU-Gen RTL selects its multiplier, adder and pre-approximation units
entirely through macros, so which modules a design instantiates can only be
known after the conditionals are resolved against the generated .vh.
"""

from __future__ import annotations

import re

from tpugen_types import FlowError

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_DIRECTIVE = re.compile(
    r"^\s*`(define|undef|ifdef|ifndef|elsif|else|endif)\b[ \t]*(\S+)?[ \t]*(.*)$"
)


def strip_comments(text: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(" ", text))


def collect_defines(vh_text: str, seed: dict[str, str] | None = None) -> dict[str, str]:
    """Evaluate a header to its final macro set.

    The upstream header defines macros inside `ifdef blocks that depend on
    macros defined earlier in the same file (MITCHELL implies
    SHARED_PRE_APPROX, ALM_LOA implies ALM), so this is a single ordered pass,
    repeated until it reaches a fixed point.
    """
    defines = dict(seed or {})
    for _ in range(8):
        before = dict(defines)
        _scan(vh_text, defines)
        if defines == before:
            return defines
    raise FlowError("macro definitions did not converge; check for circular `ifdef")


def _scan(text: str, defines: dict[str, str]) -> None:
    # stack of (this_branch_active, any_branch_taken)
    stack: list[tuple[bool, bool]] = []
    for line in strip_comments(text).splitlines():
        m = _DIRECTIVE.match(line)
        if not m:
            continue
        kind, arg, rest = m.group(1), m.group(2), (m.group(3) or "").strip()
        active = all(a for a, _ in stack)

        if kind == "ifdef":
            taken = active and arg in defines
            stack.append((taken, taken))
        elif kind == "ifndef":
            taken = active and arg not in defines
            stack.append((taken, taken))
        elif kind == "elsif":
            if not stack:
                raise FlowError("`elsif without `ifdef")
            _, any_taken = stack[-1]
            outer = all(a for a, _ in stack[:-1])
            taken = outer and not any_taken and arg in defines
            stack[-1] = (taken, any_taken or taken)
        elif kind == "else":
            if not stack:
                raise FlowError("`else without `ifdef")
            _, any_taken = stack[-1]
            outer = all(a for a, _ in stack[:-1])
            taken = outer and not any_taken
            stack[-1] = (taken, any_taken or taken)
        elif kind == "endif":
            if not stack:
                raise FlowError("`endif without `ifdef")
            stack.pop()
        elif kind == "define" and active and arg:
            defines[arg] = rest
        elif kind == "undef" and active and arg:
            defines.pop(arg, None)
    if stack:
        raise FlowError("unterminated `ifdef block")


def resolve(text: str, defines: dict[str, str]) -> str:
    """Return only the source lines that survive the conditionals."""
    out: list[str] = []
    stack: list[tuple[bool, bool]] = []
    for line in strip_comments(text).splitlines():
        m = _DIRECTIVE.match(line)
        if not m:
            if all(a for a, _ in stack):
                out.append(line)
            continue
        kind, arg = m.group(1), m.group(2)
        if kind in ("ifdef", "ifndef"):
            hit = (arg in defines) if kind == "ifdef" else (arg not in defines)
            taken = all(a for a, _ in stack) and hit
            stack.append((taken, taken))
        elif kind == "elsif":
            _, any_taken = stack[-1]
            outer = all(a for a, _ in stack[:-1])
            taken = outer and not any_taken and arg in defines
            stack[-1] = (taken, any_taken or taken)
        elif kind == "else":
            _, any_taken = stack[-1]
            outer = all(a for a, _ in stack[:-1])
            taken = outer and not any_taken
            stack[-1] = (taken, any_taken or taken)
        elif kind == "endif":
            stack.pop()
    return "\n".join(out)
