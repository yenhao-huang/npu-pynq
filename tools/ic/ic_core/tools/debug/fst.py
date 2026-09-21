"""FST debug backend -- the default.

Open question Q2 in the architecture plan was whether Python can read FST at
all. It can, but not natively: there is no maintained pure-Python FST reader.
What there is, shipped with GTKWave and therefore already required by the
`view` category, is `fst2vcd`.

So this backend converts once and caches the result next to the trace, keyed
by the source file's size and mtime. The conversion is the expensive step and
a debugging session issues many queries against one waveform, so paying it
once turns every subsequent query into a plain VCD scan.

Simulations still write FST, which is what the plan wanted: the format on disk
stays small, and the expansion is a derived, disposable artifact under
`.ic/cache/`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ...errors import BackendUnavailable, ToolFailed
from ...paths import store_root
from ...process import run as run_process
from ...registry import backend
from .vcd import _VcdBase


def cache_path(source: Path) -> Path:
    stat = source.stat()
    key = hashlib.sha256(
        f"{source.resolve()}:{stat.st_size}:{int(stat.st_mtime)}".encode()
    ).hexdigest()[:16]
    return store_root() / "cache" / f"{source.stem}-{key}.vcd"


def to_vcd(source: Path) -> Path:
    """Expand an FST to VCD once, then reuse it."""
    if source.suffix.lower() == ".vcd":
        return source
    target = cache_path(source)
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".vcd.partial")
    result = run_process(
        ["fst2vcd", str(source), "-o", str(partial)],
        log_path=target.with_suffix(".convert.log"),
        timeout_s=1800,
    )
    if result.exit_code == 127:
        raise BackendUnavailable(
            "fst2vcd is not on PATH; install gtkwave, or use backend='vcd' "
            "with a VCD trace",
            requires="fst2vcd",
        )
    if result.exit_code != 0 or not partial.exists():
        raise ToolFailed(
            f"fst2vcd failed on {source.name}",
            exit_code=result.exit_code,
            log=str(result.log_path),
        )
    partial.replace(target)
    return target


@backend("debug", "fst", requires="fst2vcd")
class FstDebug(_VcdBase):
    def _prepare(self, path: Path) -> Path:
        return to_vcd(path)
