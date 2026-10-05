"""equiv -- the category contract.

`sim` answers "does this design pass the testbench I wrote". `equiv` answers
"does this rewrite still do what the previous version did". It is the tool
that makes optimisation safe: an agent keeps a design it has already verified
as the reference, rewrites it for a better `ppa`, and asks `equiv` whether the
rewrite changed any observable behaviour -- without writing a new testbench.

| Design          | Method                                                 | Verdict   |
| --------------- | ------------------------------------------------------ | --------- |
| combinational   | Yosys miter + SAT, over every input value              | a proof   |
| sequential      | random differential simulation, several seeds, resets  | evidence  |

A sequential verdict is evidence, not a proof: it drives the same random
stimulus into both designs cycle by cycle and compares every output after each
clock edge. Outputs the reference leaves unknown (X) are not compared.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..common import Backendable


class EquivIn(Backendable):
    ref_files: list[str] = Field(
        description="Sources of the reference design, e.g. the version already verified"
    )
    dut_files: list[str] = Field(description="Sources of the rewritten design to check")
    top: str = Field(description="Top module name; both designs must use it and have the same ports")
    clock_ports: list[str] = Field(
        default_factory=list,
        description="Clock inputs. Default: 1-bit inputs whose name contains clk or clock.",
    )
    reset_ports: list[str] = Field(
        default_factory=list,
        description="Reset inputs. Default: 1-bit inputs whose name contains rst or reset; active-low if the name ends in n.",
    )
    cycles: int = Field(default=3000, description="Random cycles per seed (sequential designs)")
    seeds: int = Field(default=3, description="Independent random seeds (sequential designs)")
    timeout_s: float = Field(default=300.0, description="Kill the check after this many seconds")


class Mismatch(BaseModel):
    seed: int | None = None
    cycle: int | None = None
    port: str
    ref: str
    dut: str


class EquivOut(BaseModel):
    ok: bool = Field(description="The check ran to completion (independent of the verdict)")
    equivalent: bool | None = Field(
        default=None, description="True if no difference was found; null if the check could not run"
    )
    method: str = Field(description="'sat-proof' (combinational) or 'random-sim' (sequential)")
    sequential: bool | None = None
    compared_cycles: int = 0
    mismatches: list[Mismatch] = Field(
        default_factory=list, description="First differences found, at most five"
    )
    counterexample: dict[str, str] = Field(
        default_factory=dict, description="Input values that distinguish the designs (SAT only)"
    )
    summary: list[str] = Field(default_factory=list)
    report: str | None = Field(default=None, description="Full log handle 'run_id/equiv.log'")
    duration_s: float = 0.0
    backend: str
    backend_version: str
    note: str | None = None
    run_id: str | None = None


from ...registry import Category, Op, register_category  # noqa: E402

CATEGORY = register_category(
    Category(
        name="equiv",
        summary="Check that a rewritten design behaves like a reference version.",
        default_backend="yosys",
        ops=[
            Op(
                name="equiv",
                In=EquivIn,
                Out=EquivOut,
                summary=(
                    "Check whether dut_files behave exactly like ref_files (same top, same "
                    "ports). Combinational designs get a SAT proof with a counterexample on "
                    "failure; sequential designs get random differential simulation with "
                    "resets, reporting the first mismatching cycle. Use it after every "
                    "optimisation rewrite, with the last verified version as ref_files."
                ),
                long_running=True,
            )
        ],
    )
)

from . import yosys  # noqa: E402,F401
