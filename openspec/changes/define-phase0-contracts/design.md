## Context

The repository currently has validation entry points but no numeric model,
software ABI, performance estimator, RTL, runtime, or exporter. Phase 1A must
implement arithmetic against executable behavior, while Phase 1B and 1C need a
stable control/data contract. See `proposal.md` for motivation and the three
delta specs for observable requirements.

The target is PYNQ-Z1 / xc7z020clg400-1. Generated Vivado projects and
bitstreams are excluded from source control, and board or synthesis results
cannot be claimed from host-only validation.

## Goals / Non-Goals

**Goals:**

- Make every Phase 0 rule executable and directly importable by later tests.
- Keep arithmetic functions scalar-or-array deterministic across Python and
  SystemVerilog endpoint cases.
- Express the ABI from one constant table and validate jobs before hardware
  access.
- Make performance assumptions explicit inputs rather than hidden constants.
- Provide unit tests that prove the specification examples and boundary cases.

**Non-Goals:**

- Implement PE or array RTL, AXI protocol logic, DMA, Vivado Tcl, or a board
  overlay; those belong to Phase 1A and 1B changes.
- Define ResNet operator scheduling or exporter format beyond the tensor and
  requantization contract required by later phases.
- Claim measured bandwidth, timing closure, utilization, or board performance.

## Decisions

### Keep contract code under `src/test/model`

Phase 0 adds `numeric.py`, `abi.py`, and `performance.py` plus package exports.
Focused standard-library `unittest` suites live under `src/test/tests/`. This
matches the repository filetree and lets RTL verification import the golden
model without creating a second contract location.

Alternative considered: place the ABI directly under `src/runtime/`. Rejected
for Phase 0 because the runtime does not yet exist and the first consumer is
differential verification. When production runtime begins, shared contract
code can be promoted to `src/model/` as required by the filetree rule.

### Saturate every accumulator update

The golden model performs product formation with unlimited Python precision,
adds in reduction-index order, and clamps after every update to the signed
INT32 interval. This produces deterministic overflow behavior and maps to a
PE-local comparison/multiplexer implementation.

Alternative considered: two's-complement wrap. Rejected because silent wrap
creates large sign discontinuities and makes an accidentally oversized
reduction indistinguishable from valid output.

### Use integer-only requantization

Requantization uses a signed Q1.31 multiplier, a non-negative right shift, an
integer divide with ties away from zero, output zero point addition, and INT8
saturation. No floating-point operation participates in bit-accurate expected
results. A helper may derive parameters from floating-point scale later, but
that helper is outside the hardware contract.

Alternative considered: round-to-even using host language defaults. Rejected
because Python, NumPy, synthesis code, and other exporters can otherwise differ
at half-way values.

### Represent ABI fields as typed constants and validated values

`abi.py` defines version/capability/error enums, the exact register offsets,
immutable matrix job parameters, and 64-byte-aligned 32-bit physical buffer
ranges. Validation returns a structured exception before any MMIO or DMA call.
Register encoding remains little-endian 32-bit and the first 256 bytes are
reserved for ABI version 1.

Alternative considered: duplicate numeric offsets in runtime and RTL tests.
Rejected because duplicated literals allow software and hardware maps to drift.

### Separate ideal kernel cycles from end-to-end roofline time

`performance.py` returns a report containing operation count, payload bytes,
tile geometry, ideal compute cycles, compute time, transport time, launch
overhead, limiting factor, throughput, and resource headroom. Inputs are
immutable target/configuration records. No estimate is presented as a measured
result.

For tiled multiplication, each tile contributes `tile_m + tile_n + tile_k - 2`
fill/compute/drain cycles and tiles are conservatively serialized. This is a
reviewable lower-complexity Phase 0 model; Phase 1A can refine overlap only by
updating the contract and tests.

Alternative considered: report only peak `2*rows*cols*fclk`. Rejected because
it hides fill/drain, edge utilization, memory traffic, and launch cost.

### Add a Python model gate to the existing Makefile

The Makefile gains a `model` target that runs unittest discovery. The stable
`all` entry point runs `model`, `lint`, and `sim`; existing RTL-only commands
remain unchanged. This gives Phase 0 a CI-compatible gate without requiring
pytest or altering global Python environments.

## Risks / Trade-offs

- [Per-MAC saturation costs timing/resources] -> Phase 1A measures the cost;
  changing to wrap or end-only saturation requires an explicit spec change.
- [Conservative serialized tiling underestimates a pipelined design] -> Keep
  overlap as an explicit future model parameter backed by measured evidence.
- [600 MB/s is a planning assumption, not a board measurement] -> Reports
  label it as assumed and Phase 1B records measured sustained bandwidth.
- [Stacked branch depends on draft PR #11] -> Keep Phase 0 commits isolated and
  sync `dev` after #11 merges before opening the final Phase 0 PR.
- [Python integers hide machine overflow] -> Every public arithmetic boundary
  performs explicit range validation or saturation, with endpoint tests.

## Migration Plan

1. Land the repository/OpenSpec foundation from PR #11 into `dev`.
2. Sync `npu/issue2-a` with `dev` without rewriting published history.
3. Land the Phase 0 OpenSpec artifacts, executable contracts, and tests.
4. Require Phase 1A to import the numeric model for differential vectors and
   Phase 1B/1C to consume ABI constants rather than duplicate values.
5. If Phase 0 is rejected, revert its isolated commit; no hardware or runtime
   artifact exists yet, so there is no on-board migration.
