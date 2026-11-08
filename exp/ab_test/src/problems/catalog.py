"""Discover problems without import-time dataset access."""
from dataclasses import dataclass
from pathlib import Path
import re
from .patches import REF_TOP_RENAME

@dataclass(frozen=True)
class Problem:
    id: str
    directory: Path
    top: str
    reference: Path
    category: str

    def description(self):
        return (self.directory / "design_description.txt").read_text()

def discover(root):
    probs = {}
    for d in sorted(root.rglob("design_description.txt")):
        pdir = d.parent
        refs = list(pdir.glob("verified_*.v"))
        if not refs:
            continue
        # The top is whichever reference module the shipped testbench instantiates.
        names = [re.sub(r"^verified_", "", n) for n in re.findall(r"^\s*module\s+(\w+)", refs[0].read_text(), re.M)]
        tb = (pdir / "testbench.v").read_text()
        used = [n for n in names if re.search(rf"^\s*{n}\s*(#|\w)", tb, re.M)]
        top = used[0] if used else (names[0] if names else pdir.name)
        if pdir.name in probs:
            raise ValueError(f"Duplicate problem ID: {pdir.name}")
        top = REF_TOP_RENAME.get(pdir.name, (top, top))[1]
        probs[pdir.name] = Problem(pdir.name, pdir, top, refs[0], pdir.relative_to(root).parts[0])
    return probs

def reference_source(problem):
    """The reference with its `verified_` prefix removed, as the testbench expects."""
    src = re.sub(r"\bverified_", "", problem.reference.read_text())
    if problem.id in REF_TOP_RENAME:
        old, new = REF_TOP_RENAME[problem.id]
        src = re.sub(rf"\b{old}\b", new, src)
    return src

def prefixed_reference(problem):
    src = reference_source(problem)
    names = re.findall(r"^\s*module\s+(\w+)", src, re.M)
    for n in sorted(set(names), key=len, reverse=True):
        src = re.sub(rf"\b{n}\b", f"ref_{n}", src)
    return src
