## Purpose

Defines reproducible resource, compute, transport, and acceptance calculations
for choosing and validating a PYNQ-Z1 NPU configuration.

## ADDED Requirements

### Requirement: Explicit target assumptions
Every performance report SHALL identify array rows and columns, clock rate,
operand and accumulator widths, sustained memory bandwidth, fixed launch
overhead, bytes transferred, and target device resource limits. Default
planning assumptions SHALL be 100 MHz, 600 MB/s sustained PS-PL bandwidth,
and xc7z020 limits of 53,200 LUTs, 106,400 flip-flops, 140 BRAM36 blocks, and
220 DSP48E1 slices.

#### Scenario: Reproducible report
- **WHEN** two reports use identical matrix dimensions and target assumptions
- **THEN** they produce identical modeled cycles, time, throughput, bandwidth demand, and resource headroom

### Requirement: Matrix operation and traffic accounting
An MxK by KxN matrix multiplication SHALL count 2*M*N*K operations. Minimum
payload traffic SHALL count M*K INT8 input bytes, K*N INT8 weight bytes, and
4*M*N INT32 output bytes. Reports SHALL state separately any padding,
retransmission, descriptor, or cache-maintenance traffic.

#### Scenario: Payload accounting
- **WHEN** M=2, N=3, and K=4
- **THEN** the model reports 48 operations and 44 minimum payload bytes

### Requirement: Compute and roofline timing
Ideal array cycles SHALL be M+N+K-2 for a fully utilized single tile and SHALL
include explicit tile count and per-tile fill/drain for larger matrices.
Modeled job time SHALL be the maximum of compute time and transport time plus
fixed launch overhead. Throughput SHALL use modeled operations divided by
modeled job time and SHALL identify whether compute or transport is limiting.

#### Scenario: Bandwidth-bound job
- **WHEN** payload bytes divided by sustained bandwidth exceeds compute cycles divided by clock rate
- **THEN** modeled time uses transport time as the limiting component

#### Scenario: Compute-bound job
- **WHEN** compute cycles divided by clock rate is at least transport time
- **THEN** modeled time uses compute time as the limiting component

### Requirement: Resource budget gate
A candidate configuration SHALL report absolute and percentage use for LUT,
flip-flop, BRAM36, and DSP resources. The default production planning gate
SHALL reserve 25 percent of each device resource for integration and routing;
any estimate above 75 percent SHALL fail the planning gate unless an explicit
reviewed override records the reason.

#### Scenario: Resource over budget
- **WHEN** a candidate estimates 106 DSP48E1 slices out of a 140-slice planning budget
- **THEN** the DSP planning gate fails and identifies the excess

### Requirement: Model-to-measurement acceptance
Board reports SHALL record measured cycles, wall-clock time, achieved
operations per second, payload bandwidth, array utilization, synthesis
utilization, and timing slack. Before production acceptance, measured kernel
cycles for supported matrix shapes SHALL be within 10 percent of the cycle
model and reported end-to-end time SHALL identify all unmodeled overhead.

#### Scenario: Cycle model acceptance
- **WHEN** a supported job models 1000 kernel cycles and measures from 900 through 1100 inclusive
- **THEN** the kernel-cycle agreement gate passes

#### Scenario: Cycle model rejection
- **WHEN** a supported job models 1000 kernel cycles and measures outside 900 through 1100
- **THEN** the gate fails and the report requires an explained model or implementation correction
