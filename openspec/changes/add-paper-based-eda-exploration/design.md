# Design

Operations reside in existing lint/debug/synth categories; the pipeline category
contains an RTLRewriter-inspired composition. Per-operation input defaults select
the new backend without changing defaults for existing tools. Transport clients
remain unchanged. Packaging explicitly includes the new pipeline category.

Measurements use identical part, constraint, tool version, metric units and flow
before comparison. A source fingerprint binds each result to its RTL. LUT counts
are FPGA resource counts; routed combinational delay is not clock frequency.
Power remains unknown until an activity-qualified power flow is implemented.

The checker enumerates all binary inputs for explicitly combinational, unsigned
interfaces (at most 16 input bits), fails on X/Z outputs, and records source hashes.
It does not establish sequential equivalence, reset safety, or four-state behavior.
The pipeline stops on failed checks and records child run IDs. It never rewrites
the user's RTL. Rule lookup and width analysis produce advice with preconditions.

Experiment loop: hypothesis -> exact inputs/command -> recorded result -> next
decision. Each tool has its own experiment folder; the shared runner supports
single-tool execution, resuming results, and a fresh output directory.
