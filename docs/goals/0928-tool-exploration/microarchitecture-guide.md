# Microarchitecture guide

The project counts each callable generator as one operation. The alternatives
below are microarchitectures of the same declared behavior; they are not extra
tools. A shorter expected logic path or smaller storage structure is a design
hypothesis until correctness and routed PPA evidence pass.

## Arithmetic and reduction

| Operation | Microarchitecture | What it builds | Expected tradeoff |
| --- | --- | --- | --- |
| Adder reduction | `serial` | Adds lanes through one left-associated combinational chain. | Simple reference with depth proportional to lane count. |
|  | `balanced` | Adds pairs at each level in a balanced binary tree. | Logarithmic add depth; similar arithmetic count. |
|  | `compressor` | Uses 3:2 carry-save stages, then one final carry-propagate add. | Shorter carry propagation with more intermediate logic. |
| Prefix adder | `native` | Uses the RTL `+` operator as the synthesis baseline. | Lets the vendor choose its adder mapping. |
|  | `kogge_stone` | Computes carries with a dense parallel-prefix graph. | Low prefix depth and high wiring/fanout cost. |
|  | `sklansky` | Uses a divide-and-conquer prefix graph. | Fewer prefix nodes, with higher fanout near the root. |
| Constant multiplier | `native` | Uses a compile-time constant `*`. | Vendor-controlled baseline; may map to logic or DSPs. |
|  | `binary` | Adds shifted copies for every set coefficient bit. | No multiplier primitive; cost follows coefficient popcount. |
|  | `csd` | Adds or subtracts shifted copies from a signed-digit coefficient. | Fewer nonzero terms for coefficients with runs of ones. |
| Multiple-constant multiplier | `independent` | Builds each constant product as its own binary shift/add graph. | No cross-output sharing. |
|  | `shared` | Memoizes powers and factors of `2^k +/- 1` across outputs. | Reuses intermediate sums; heuristic rather than globally minimal. |
| Constant modulo | `native` | Uses RTL remainder by the fixed Mersenne modulus. | Vendor-controlled reference. |
|  | `folded` | Adds fixed-width chunks and repeatedly folds carry bits end-around. | Replaces a wide remainder network with bounded additions. |
| Saturating ALU | `dual` | Computes add and subtract on separate extended paths, then clamps. | Direct control and duplicated arithmetic. |
|  | `shared` | Selects conditional inversion/carry-in around one extended adder. | Shares arithmetic at the cost of input/control muxing. |

## Multipliers and datapaths

| Operation | Microarchitecture | What it builds | Expected tradeoff |
| --- | --- | --- | --- |
| FIR window | `direct` | Multiplies every supplied sample by its coefficient and reduces the products with a balanced tree. | Full parallel product count and short reduction depth. |
|  | `symmetric` | Pre-adds symmetric sample pairs, then multiplies once per coefficient pair. | Roughly halves products for palindromic coefficients; preadd width grows by one bit. |
|  | `direct_serial` | Keeps all products but reduces them in a serial chain. | Ablation baseline that isolates reduction topology. |
| Dot product | `serial` | Reduces signed products through a carry-propagating chain. | Small structural reference with long depth. |
|  | `balanced` | Reduces signed products in a balanced tree. | Logarithmic reduction depth. |
|  | `compressor` | Compresses sign-extended products in 3:2 stages before a final add. | Higher logic use for less carry propagation. |
| Booth multiplier | `native` | Uses exact signed RTL multiplication. | Vendor-controlled baseline. |
|  | `radix2` | Generates one Booth row per multiplier bit from adjacent-bit recoding. | Replaces dense partial products with add/subtract rows. |
|  | `radix4` | Recodes two multiplier bits per row, including signed `+/-2A`. | About half as many rows, with more complex row selection. |
| Iterative multiplier | `parallel` | Registers one combinational product. | II 1 and large combinational multiplier. |
|  | `serial` | Reuses one add/shift step for one multiplier bit per cycle. | Much smaller datapath, about one bit per work cycle. |
|  | `radix4` | Consumes two multiplier bits per cycle. | About half the serial latency with a larger digit step. |
| Divider | `parallel` | Registers combinational quotient and remainder operators. | High parallel cost and II 1. |
|  | `serial` | Performs one restoring division bit per cycle. | Reuses compare/subtract/shift logic over many cycles. |
|  | `radix4` | Performs two restoring bit steps per cycle. | Lower latency than serial with more work in each cycle. |
| Matrix tile | `parallel` | Computes every signed product and matrix output together. | Maximum product concurrency and II 1. |
|  | `systolic` | Moves operands through an output-stationary nearest-neighbor PE wavefront. | Regular local communication and reused time, with batch latency/II cost. |

FIR and dot-product experiments also select `logic` or `dsp` multiplier
mapping. That setting controls resource inference; it does not change the
mathematical architecture.

## Count, selection and shift networks

| Operation | Microarchitecture | What it builds | Expected tradeoff |
| --- | --- | --- | --- |
| Population count | `linear` | Adds input bits one after another. | Linear dependency depth. |
|  | `tree` | Adds width-growing partial counts pairwise. | Logarithmic depth with explicit intermediate widths. |
| Priority encoder | `linear` | Scans every bit and repeatedly overwrites the current valid/index result. | Long priority mux chain. |
|  | `tree` | Combines valid/index pairs hierarchically, favoring the higher half. | Logarithmic selection depth. |
| Leading-zero count | `linear` | Scans bits through a priority chain. | Simple, linear-depth baseline. |
|  | `tree` | Recursively counts each half and selects by high-half emptiness. | Parallel recursive count structure. |
|  | `binary_search` | Tests the current high half, records one decision bit, then narrows to one half. | One reduction and mux per logarithmic level. |
| Barrel shifter | `linear` | Decodes every shift amount into a full case mux. | Wide decoded mux structure. |
|  | `tree` | Cascades conditional shifts of 1, 2, 4, and so on. | Logarithmic staged mux network. |
| Masked one-hot mux | `linear` | Masks every lane and ORs them through a chain. | Linear OR depth. |
|  | `tree` | ORs masked lanes pairwise. | Logarithmic OR depth; multi-hot remains defined as OR. |
| Stable argmax | `linear` | Compares the running winner against each next lane. | Linear comparison/mux depth. |
|  | `tree` | Runs pairwise comparison tournaments. | Logarithmic depth while preserving the lowest-index tie rule. |

## GF(2) transforms

| Operation | Microarchitecture | What it builds | Expected tradeoff |
| --- | --- | --- | --- |
| CRC update | `unrolled` | Expands every MSB-first recurrence step in sequence. | Direct recurrence with depth proportional to data width. |
|  | `matrix` | Derives every next-state bit as an independent balanced XOR row. | Parallel rows with duplicated subexpressions. |
|  | `shared` | Greedily reuses frequent XOR pairs across matrix rows. | Fewer XOR nodes when useful common terms exist. |
| LFSR jump | `unrolled` | Expands the autonomous recurrence for every requested step. | Direct but potentially deep. |
|  | `matrix` | Exponentiates the GF(2) transition matrix and emits balanced rows. | Step count is compiled into a parallel transform. |
|  | `shared` | Reuses frequent XOR pairs across jump-matrix rows. | Trades a heuristic search for less duplicated XOR logic. |

## Storage, state and flow control

| Operation | Microarchitecture | What it builds | Expected tradeoff |
| --- | --- | --- | --- |
| FIFO | `shift` | Stores words in an array and shifts entries on pop. | Many data moves and straightforward indexing. |
|  | `circular` | Keeps read/write pointers and leaves stored words in place. | Scales better and can infer distributed or block RAM. |
| Register file | `registers` | Replicates resettable flip-flop words for two asynchronous reads. | Fast arbitrary access with high FF/mux cost. |
|  | `banked` | Splits words across LUTRAM-like banks with validity bits and an explicit conflict rule. | Lower storage cost with bank-selection/conflict logic. |
| Phase controller | `binary` | Stores the phase as a binary counter-like state code. | Few state bits and decode logic on outputs. |
|  | `onehot` | Stores one bit per phase and rotates the active bit. | More FFs, simpler state decode. |
| Backpressure pipeline | `elastic` | Uses one occupied slot per stage and lets readiness propagate combinationally backward. | Capacity equals stage count; long ready path. |
|  | `skid` | Gives each stage head/spill slots and derives readiness from local registered occupancy. | Cuts the ready path, doubles capacity, and adds recovery behavior. |

All registered combinational candidates use the same input/output observation
wrapper. Stateful candidates instead use protocol-specific scoreboards so
capacity, backpressure, reset, latency and II differences remain visible.
