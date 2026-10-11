# Multi-round vs single-round: where the PPA gain comes from, and what it costs

[Report.md](../Report.md) compares whole arms and shows multi-round reaching 0.587× /
0.746× / 0.709× PDAP against baseline. It does not answer the obvious follow-up:
**single-round has exactly the same IC tools, so why does multi-round win?**

This note answers that, plus the cost question. Measured numbers come from
`analysis/multi_round_analysis.py`, which reuses `analyze.py`'s attempt selection, so it
counts the same attempts Report.md counts. Claude costs and all token counts are read
from the CLI streams (`round*.jsonl`), which Report.md only reports as arm totals.
GPT-6 Sol dollar amounts are estimates using current API rates, not billed costs.

Both arms use the identical initial prompt (`src/agent/prompt.py`, arm `tools`), the same
tool set, and the same budget per call. The only differences are how many turns the
harness grants and whether a turn can see the previous turn's measurements.

## 1. What one round costs

A single-round attempt and a multi-round *round* are not the same unit of work. Each
round carries the whole prior conversation, so input tokens grow across rounds.
Following the resumed-session accounting in
[PR #135](https://github.com/yenhao-huang/npu-pynq/pull/135), each round's Claude
cost and Codex tokens are the difference between consecutive session totals. The first
table multiplies each per-attempt or per-round average by 50. Its multi-round row
therefore represents 50 rounds, not 50 complete problems; one complete problem
has five rounds. The actual 50-problem multi-round totals are shown below it.

| 50 attempts or rounds at the per-unit average | Opus 5.5 | Sonnet 5.5 | gpt-6-sol |
|---|---|---|---|
| baseline | $4.05<br>505K in · 105K out | $1.51<br>475K in · 60K out | $2.36<br>1.565M in · 75K out |
| single-round | $7.19<br>2.780M in · 170K out | $1.67<br>1.260M in · 60K out | $8.97<br>11.390M in · 195K out |
| multi-round, mean per round | $12.61<br>10.445M in · 305K out | $2.86<br>3.695M in · 120K out | $12.52<br>24.000M in · 259K out |
| **ratio, one round vs one attempt** | **1.75× cost · 3.76× in · 1.80× out** | **1.71× cost · 2.93× in · 2.01× out** | **1.40× estimated cost · 2.11× in · 1.34× out** |

The actual multi-round arm ran five rounds for each of 50 problems (250 rounds
total). Its totals, as reported in [PR #135](https://github.com/yenhao-huang/npu-pynq/pull/135),
are **$63.07** for Opus, **$14.29** for Sonnet, and **120.0M input / 1.29M output
tokens** for gpt-6-sol. Under the same illustrative API pricing used here, the
gpt-6-sol total is approximately **$62.60**. For example, the displayed Opus
figure is `$63.0736218 / 250 rounds × 50 = $12.61`; the 50-problem total is
`$63.0736218 / 250 rounds × 5 rounds/problem × 50 problems = $63.07`.

Codex does not report a billed amount. The GPT-6 Sol estimates use the
[Standard API prices on 2026-10-11](https://developers.openai.com/api/docs/models/gpt-6-sol):
per 1M tokens, $2 uncached input, $0.20 cached input, and $10 output for short
context; $4, $0.40, and $15 when a request exceeds 272K input tokens. We use
the arithmetic mean of those two price scenarios as one approximate value:
$3 uncached input, $0.30 cached input, and $12.50 output per 1M tokens.
The archived totals distinguish cached reads, but this table does not classify
each request by context length or capture a separate cache-write charge. This
is an illustrative API estimate, not an invoice. Displayed input tokens include
cached reads.

For a group with `U` million uncached input tokens, `C` million cached input
tokens, `O` million output tokens, and `n` attempts or rounds, the calculation is
`estimated USD per unit = (3U + 0.30C + 12.50O) / n`. The recorded token totals
and resulting estimates are:

| gpt-6-sol group | U (M) | C (M) | O (M) | n | Estimated USD per unit |
|---|---:|---:|---:|---:|---:|
| baseline | 1.036250 | 3.654144 | 0.230396 | 150 attempts | $0.047233 |
| single-round | 3.491221 | 30.681472 | 0.579293 | 150 attempts | $0.179462 |
| multi-round | 3.856909 | 116.140288 | 1.294973 | 250 rounds | $0.250400 |

The first table multiplies the displayed per-unit token averages by 50.
GPT-6 Sol costs use unrounded per-unit estimates before multiplication.
The first table is a 50-unit comparison, not the billed total for the
historical multi-round arm.

Three ratios are easy to confuse, so to be explicit: a multi-round *round* costs roughly
1.4–1.8× a single-round attempt; a whole multi-round *attempt* (5 rounds) costs 7–9× a
single-round attempt. Neither is the figure to set against the PPA results — the unit
of comparison there is one problem, and the single-round arm spends three attempts on it.
That comparison is the table at the end of this section.

The mean hides the shape. Cost per round, as a multiple of one single-round attempt:

| Round | Opus cost | Opus input | Sonnet cost | Sonnet input | gpt-6-sol input |
|---|---|---|---|---|---|
| 1 | $0.145 (1.01×) | 52.1K (0.94×) | $0.034 (1.01×) | 25.0K (0.99×) | 221.8K (0.97×) |
| 2 | $0.233 (1.62×) | 150.7K (2.71×) | $0.067 (2.02×) | 73.1K (2.89×) | 351.9K (1.54×) |
| 3 | $0.264 (1.83×) | 221.5K (3.98×) | $0.061 (1.83×) | 80.4K (3.19×) | 411.8K (1.81×) |
| 4 | $0.316 (2.20×) | 286.4K (5.15×) | $0.066 (1.98×) | 95.4K (3.78×) | 604.7K (2.65×) |
| 5 | $0.304 (2.12×) | 334.1K (6.01×) | $0.058 (1.73×) | 95.7K (3.79×) | 809.6K (3.55×) |

Round 1 matches a single-round attempt: Claude cost is 1.01× / 1.01× and GPT-6 Sol
input is 0.97×. This checks that the two arms start from the same place. Round 5
costs 2.12× / 1.73×
for Opus / Sonnet and spends 3.55× the input tokens for GPT-6 Sol. Across all
rounds, output grows less than input for each model, consistent with context
accumulation.

Per problem — the unit every PPA number in this note and in Report.md is measured in —
single-round gets 3 independent attempts and multi-round gets 1 attempt of 5 rounds:

| | Opus 5.5 | Sonnet 5.5 | gpt-6-sol |
|---|---|---|---|
| single-round, 3 attempts | $0.431 · 167K in | $0.100 · 76K in | $0.5384 · 683K in |
| multi-round, 5 rounds | $1.261 · 1,045K in | $0.286 · 370K in | $1.2520 · 2,400K in |
| **ratio** | **2.92× cost · 6.26× tokens** | **2.86× cost · 4.88× tokens** | **2.33× estimated cost · 3.51× tokens** |

## 2. Where the advantage comes from

Both arms produce candidates; the question is whether candidate *k* is better for having
seen candidates *1..k−1*. Single-round repetitions are independent draws (fresh session
each time), multi-round rounds share one session.

Best PDAP after k candidates, in units of the expected best of one single-round attempt.
Single-round averages over every k-subset of its 3 repetitions; multi-round takes the
prefix of rounds 1..k, since rounds are sequential and cannot be permuted.

| k candidates | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Opus, single-round (independent) | 1.000× | 0.945× | **0.915×** | — | — |
| Opus, multi-round (conditioned) | 1.021× | 0.799× | **0.704×** | 0.655× | 0.611× |1qaz@WSX3edc
| Sonnet, single-round | 1.000× | 0.999× | **0.999×** | — | — |
| Sonnet, multi-round | 0.980× | 0.812× | **0.767×** | 0.746× | 0.731× |
| gpt-6-sol, single-round | 1.000× | 0.947× | **0.906×** | — | — |
| gpt-6-sol, multi-round | 1.031× | 0.915× | **0.839×** | 0.775× | 0.735× |

n = 39 / 27 / 25 problems with valid candidates in both arms.

Two things to read off this:

**Round 1 is not better** (1.021× / 0.980× / 1.031×). The advantage is not that a
multi-round model thinks harder on its first try — it starts from the same distribution.

**At the same candidate count, conditioned search wins.** At k=3, 0.915× vs 0.704× for
Opus and 0.999× vs 0.767× for Sonnet. Independent resampling is also already flattening
out (Opus gains 5.5% then 3.2%), while the multi-round curve is still dropping at k=5.

Sonnet's independent curve is flat outright at 0.999×, and the reason is visible in the
raw data: three fresh sessions on the same specification keep landing on the *same
design*.

| Problems where all 3 repetitions produced an identical best PDAP | |
|---|---|
| Opus 5.5 | 18 / 35 |
| Sonnet 5.5 | 22 / 25 |
| gpt-6-sol | 11 / 25 |

`JC_counter` under Opus is the clean illustration: all three single-round repetitions
return PDAP 1.5448e-3 — not close, identical — and so do all three baseline repetitions.
For Sonnet, 88% of problems behave this way, which is why paying for two extra
independent attempts buys it 0.1%. Repetition without memory re-rolls the same answer.

The mechanism is in the prompt. `continuation_prompt` in `src/agent/prompt.py` asks each
round for a *structurally different* candidate, rotates the target through lower power →
lower delay → smaller area, and tells the model to compare against the candidates it has
already measured. An independent repetition has none of that: no memory of the previous
design and no instruction to differ from it.

A second factor is simply that nothing stops a multi-round attempt early. Opus
single-round calls `ppa` 1.55 times per attempt on average (233 calls / 150 attempts). It
writes a correct module, measures it once, and stops; its own success criterion is met.
Multi-round calls `ppa` 11.8 times per attempt, because the harness keeps asking.

**This is not an equal-cost comparison.** At k=3 multi-round spends 1.49× / 1.62×
the Claude cost and 1.44× the GPT-6 Sol input tokens of three single-round attempts.
The claim supported here is that conditioning
beats independent resampling *at equal candidate count*, not that it beats spending the
same money on more independent samples. No arm in this experiment runs single-round 15
times, so that comparison is open.

## 3. The five problems multi-round improved most

Ratio is multi-round best PDAP over single-round best PDAP, lower being better.
Measurements are the full triple behind each best candidate.

### Opus 5.5 (paired geomean 0.664×, n=39)

| Problem | Ratio | Best at | area µm² | power W | delay ns |
|---|---:|---|---|---|---|
| `freq_div` ⚠ | 0.102× | round 5 | 70.22 → 55.33 | 8.00e-6 → 2.63e-6 | 0.280 → 0.110 |
| `traffic_light` | 0.119× | round 4 | 152.68 → 106.93 | 4.31e-5 → 8.20e-6 | 0.480 → 0.430 |
| `JC_counter` ⚠ | 0.131× | round 5 | 340.48 → 145.50 | 3.49e-5 → 5.80e-6 | 0.130 → 0.240 |
| `freq_divbyodd` | 0.317× | round 4 | 57.46 → 35.91 | 7.42e-6 → 3.77e-6 | 5.130 → 5.120 |
| `edge_detect` | 0.317× | round 4 | 18.35 → 12.50 | 2.13e-6 → 1.22e-6 | 0.160 → 0.130 |

### Sonnet 5.5 (paired geomean 0.732×, n=27)

| Problem | Ratio | Best at | area µm² | power W | delay ns |
|---|---:|---|---|---|---|
| `freq_div` | 0.245× | round 4 | 98.42 → 73.15 | 1.11e-5 → 5.21e-6 | 0.370 → 0.260 |
| `fsm` | 0.311× | round 4 | 38.04 → 22.34 | 4.90e-6 → 2.46e-6 | 0.180 → 0.190 |
| `up_down_counter` | 0.330× | round 2 | 177.42 → 154.28 | 3.71e-5 → 1.74e-5 | 0.680 → 0.550 |
| `edge_detect` | 0.365× | round 2 | 18.35 → 13.03 | 2.13e-6 → 1.25e-6 | 0.160 → 0.140 |
| `pe` | 0.497× | round 3 | 3610.15 → 3970.58 | 3.16e-3 → 1.98e-3 | 1.580 → 1.140 |

### gpt-6-sol (paired geomean 0.812×, n=25)

| Problem | Ratio | Best at | area µm² | power W | delay ns |
|---|---:|---|---|---|---|
| `JC_counter` | 0.288× | round 5 | 340.48 → 165.45 | 3.49e-5 → 7.68e-6 | 0.130 → 0.350 |
| `ring_counter` | 0.511× | round 5 | 31.12 → 31.92 | 3.53e-6 → 1.84e-6 | 0.230 → 0.220 |
| `pe` | 0.565× | round 5 | 3610.15 → 2669.58 | 3.16e-3 → 2.05e-3 | 1.580 → 1.860 |
| `fsm` | 0.609× | round 4 | 22.61 → 21.28 | 2.46e-6 → 1.78e-6 | 0.190 → 0.170 |
| `freq_div` | 0.640× | round 5 | 84.59 → 71.55 | 8.63e-6 → 7.70e-6 | 0.330 → 0.280 |

Notes on reading these. PDAP is a product, so a ratio is not a per-metric improvement:
`edge_detect` at 0.317× is 0.68× area × 0.57× power × 0.81× delay. Several winners are
genuine trades rather than uniform wins — `pe` under Sonnet grows 10% in area to buy 37%
power and 28% delay; gpt-6-sol's `JC_counter` gives up 2.7× delay for 2.1× area and 4.5×
power. All of these are legal Pareto moves, and all of them lower PDAP, which is why
PDAP ranks the frontier but does not define it.

Three recurring problem types dominate the list: clock dividers (`freq_div`,
`freq_divbyodd`), counters (`JC_counter`, `ring_counter`, `up_down_counter`), and small
control logic (`fsm`, `traffic_light`, `edge_detect`). These are exactly the designs with
many functionally equivalent encodings, which is the space iteration can search. Nothing
on the list is a datapath problem where the structure is forced.

The clean wins are real design work. `edge_detect` round 4 stores `~a` delayed two cycles
instead of `a`, which turns `rise` into a plain AND2 and `down` into a plain NOR2.
`traffic_light` round 4 replaces a registered pipeline with a 2-bit phase and a 6-bit
counter — 8 flops — after proving the counter never leaves 1..60.

## 4. Validity: multi-round also learns to beat the measurement

Four best candidates clock sequential state off an internally generated signal rather
than a module input. The `ppa` flow times only the primary clock, so these get an
optimistic delay and, through the default activity assumption, an optimistic power.

| Model / problem | Generated clock | What it is |
|---|---|---|
| Opus `freq_div` | `CLK_50`, `z`, `cnt_j` | cascaded divided clocks driving the next stage |
| Opus `JC_counter` | `k` | ripple counter; each bit clocks the next |
| Opus `synchronizer` | `gclk_b = clk_b \| en_clap_two_n` | combinational clock gate |
| gpt-6-sol `synchronizer` | `data_clk = clk_b & gate_en` | combinational clock gate |

Opus's `JC_counter` round 5 is the honest case: the model's own final message says the
0.24 ns figure "isn't trustworthy", explains that the ripple chain is never timed as one
path, and points to round 4's fully synchronous design as the one to use. The harness
still scored the ripple version as that problem's best PDAP.

Excluding the flagged problems:

| Model | Paired multi/single | Excluding flagged | Flagged |
|---|---|---|---|
| Opus 5.5 | 0.664× (n=39) | 0.727× (n=36) | `freq_div`, `JC_counter`, `synchronizer` |
| Sonnet 5.5 | 0.732× (n=27) | 0.732× (n=27) | none |
| gpt-6-sol | 0.812× (n=25) | 0.807× (n=24) | `synchronizer` |

So this is not inflating the headline much — about a fifth of Opus's gain in log terms,
nothing for the other two. But it concentrates at the top: two of Opus's five biggest
wins are flagged. Anyone quoting the largest per-problem improvements should quote the
clean ones.

The detection is syntactic (edge-sensitive signals that are not input ports) and is a
flag for review, not a verdict. Some of these designs are legitimate low-power technique;
the problem is that the measurement does not price them correctly either way.

## Reproduce

```bash
cd exp/ab_test
python3 analysis/multi_round_analysis.py   # -> results/multi_round_analysis.json
```

Reads local `results/<model>/{single,multi}/` and reuses `analyze.load`. The
historical run archive is not in Git, so this analysis cannot be rerun from the
compact checkout alone. See [Reproduce.md](../Reproduce.md) for how the runs
were produced.
