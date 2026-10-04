# TPU-Gen configuration space observed in the upstream dataset

This document describes the configurations in the pinned TPU-Gen training and
test data used by exp/tpu_gen. It is not a survey of the PYNQ FPGA overlay,
which has different parameters and a different numeric contract. An observed
header is a data record, not proof that elaboration, synthesis, timing, or
functional verification succeeded.

The companion observed-header-combinations.csv has 5,000 rows: 4,000 from
beta_train_all.json and 1,000 from beta_test_all.json. Each row records the
source and one-based record index, numeric macros, selected multiplier and
adder, and two static checks. Those checks are necessary conditions only.

## How a configuration becomes hardware

DesignSpec or an LLM request produces options_definitions.vh. Its numeric
macros set top parameters and its selector macros activate conditional RTL
branches. Retrieval follows module instantiations from the top, then Icarus
Verilog elaborates the selected files. ORFS performs synthesis and physical
implementation. For example, M=4 and N=4 instantiate 16 processing elements;
increasing them also changes FIFO, wiring, and selection logic.

## Numeric parameters

| Macro | Meaning | Observed values | Current check |
| --- | --- | --- | --- |
| M, N | Array rows and columns | 4, 6, 8, 16, 32, 64 | Positive; the dataset only has square arrays. |
| DW | Input data width | 8, 16, 32 | Positive. |
| WW | Weight width | 3, 4, 5, 6, 7, 8, 16, 32 | Positive. |
| MULT_DW | Accurate multiplier portion | 2, 3, 4, 5, 6, 8, 10, 12 | Positive and at most min(DW, WW). |
| ADDER_PARAM | Adder setting | 16 | Required; no full range check. |
| ROUN_WIDTH | Rounding setting | 1 | Required; no full range check. |
| NIBBLE_WIDTH | ASM nibble width | 4 | Required; no full range check. |
| VBL | Multiplier setting | 16 | Required; no full range check. |
| DEPTH | FIFO depth | Not specified in dataset headers | Top default is 64. |

Other top parameters are derived from these values and should not be
interpreted as independent LLM choices.

| Array | Processing elements | Dataset records |
| --- | ---: | ---: |
| 4x4 | 16 | 1,897 |
| 6x6 | 36 | 1,250 |
| 8x8 | 64 | 943 |
| 16x16 | 256 | 510 |
| 32x32 | 1,024 | 396 |
| 64x64 | 4,096 | 4 |

## Arithmetic selectors

The implementation lists DRUM_APTPU, BAM, UDM, EIM, ALM variants, MITCHELL,
ROBA, DRALM, ASM, and other multipliers. HIGH_REG is a register modifier, not
an independent multiplier. Several branches need SHARED_PRE_APPROX, and ALM
variants imply ALM. The adder list includes HERLOA, OLOCA4, SETA, MHERLOA,
MHEAA, LZTA, LOAWA, LOA, HOERAA, HOAANED, HEAA, APPROX5, SA_ADDER, and
LDCA. ACCURATE_ACCUMULATE is the exact accumulation branch.

The dataset contains only three primary multiplier families and 31 observed
multiplier/adder pairs:

| Multiplier | Adder counts in the 5,000 headers |
| --- | --- |
| DRUM_APTPU | APPROX5 175; HEAA 192; HERLOA 312; HOAANED 196; HOERAA 196; LDCA 1; LOA 238; LOAWA 252; LZTA 319; MHEAA 313; MHERLOA 301; OLOCA4 331; SETA 324 |
| BAM | APPROX5 54; HEAA 69; HERLOA 185; HOAANED 67; HOERAA 107; LOA 127; LOAWA 130; LZTA 129; MHEAA 128; MHERLOA 132; OLOCA4 174; SETA 187 |
| UDM | HERLOA 65; LZTA 51; MHEAA 73; MHERLOA 58; OLOCA4 49; SETA 65 |

These counts cannot be multiplied by the numeric value counts to infer the
number of valid designs. In 1,942 of the 5,000 headers, MULT_DW exceeds
min(DW, WW), so the current DesignSpec.validate or check_macros rejects them.
The dataset calls the adder OLOCA4, but the RTL branch uses OLOCA; do not treat
that combination as validated. The macro checker requires at least one listed
multiplier and one adder (or ACCURATE_ACCUMULATE); it does not enforce exactly
one. If multiple selectors are defined, RTL conditional order decides the
active branch.

## Reference header

This complete 4x4 INT8 example passes the required static macro checks.
It still needs retrieval, elaboration, and synthesis before hardware can be
claimed usable.

~~~verilog
`define DRUM_APTPU
`define APPROX5
`define ROUN_WIDTH 1
`define NIBBLE_WIDTH 4
`define DW 8
`define WW 8
`define M 4
`define N 4
`define MULT_DW 4
`define ADDER_PARAM 8
`define VBL 16
`define SHARED_PRE_APPROX
~~~

Prefer lib/vh_template.py to construct headers so implied macros stay
consistent. The validation path is check_macros -> retrieve ->
stage_sources -> elaborate, followed by measured OpenROAD output.

## Provenance

The lists and defaults come from lib/prompt_formatter/formatter.py and
lib/vh_template.py. Header validation and retrieval are in lib/validate/parse.py
and lib/retrieval/. The upstream RTL archive is restored into
src_code/rtl-20250603T193300Z-1-001/rtl/ at the commit pinned in the parent
README. The CSV was extracted from the code field of each upstream JSON row;
the dataset's historical metrics were not rerun in this PR and are not
substituted for local ORFS measurements.
