"""Prompts for exploring correct designs with different PPA tradeoffs."""
PROMPT = """Implement the Verilog module specified below and write it to ./solution.v in the current directory.

Priorities, in order:
1. Functional correctness: exactly the module name, ports and behaviour the specification describes. A hidden testbench and hidden randomized checks against a reference implementation decide correctness.
2. Explore tradeoffs between standard-cell area (area_um2), power (power_w), and delay (delay_ns = 1000 / fmax_mhz), using Nangate45. All three objectives are minimized. A design dominates another only if it is no worse in every objective and strictly better in at least one. Seek nondominated candidates, without combining the objectives into a scalar score. Never sacrifice required behaviour.

Use whatever tools you have. If you have IC design tools, the standard-cell library for `ppa` is already configured, so it needs only files, top and (for sequential designs) clock_port. Record area_um2, power_w, fmax_mhz, delay_ns and wns_ns when available. WNS is diagnostic, not a Pareto objective. Do not invent unavailable measurements. Put only synthesizable RTL in solution.v; keep any testbench in a separate file. The harness saves the design left in solution.v at the end of each round.

Specification:
"""

DIRECTED = """Required workflow (you have IC design tools):
a. Write a self-checking testbench covering reset and boundary cases. Run `sim` until it passes, and run `lint` on solution.v.
b. Run `ppa` and record area_um2, power_w, fmax_mhz, delay_ns = 1000 / fmax_mhz, and wns_ns.
c. Try an alternative implementation exploring a different area, power, or delay tradeoff. Re-run your testbench and `ppa`.
d. Leave a correct nondominated candidate in solution.v. Among incomparable candidates, prefer a new tradeoff over repeating an earlier submission. Keep alternative implementations in separate files.

"""

CONTINUE = (
    "Round {k} of {n}: explore a structurally different correct candidate targeting {focus}. "
    "Compare area_um2, power_w, and delay_ns = 1000 / fmax_mhz against earlier candidates, "
    "using available measurements. All three are minimized; improvements in one may trade off "
    "against another. Keep correct candidates that are not dominated by another measured candidate. "
    "Record wns_ns for diagnostics only. Leave a new correct tradeoff in ./solution.v for this "
    "round's snapshot; keep alternatives in separate files. If correctness is uncertain, retain "
    "the previous correct solution. Do not invent unavailable measurements."
)

def continuation_prompt(number, total):
    focus = ("lower power", "lower delay", "smaller area")[(number - 2) % 3]
    return CONTINUE.format(k=number, n=total, focus=focus)

def initial_prompt(description, arm):
    return PROMPT.replace("Specification:\n", (DIRECTED if arm == "tools_directed" else "") + "Specification:\n") + description
