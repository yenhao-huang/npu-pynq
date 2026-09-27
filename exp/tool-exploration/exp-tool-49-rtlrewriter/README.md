# Checkpointed architecture sweep

Operation: architecture_sweep. Compose generation, vector checks, SAT proofs
and repeated clocked implementation. Checkpoints include all core Python source,
validated inputs and actual tool versions; an OS advisory lock prevents concurrent
writers. Generated core and wrapper fingerprints bind correctness to measurement.
A failed case remains in the final result. No pipeline outcome alone accepts PPA.

Run network_study.py for tools 23 through 28. The verify-only mode exercises the
same correctness gates without launching Vivado. Full acceptance needs physical
runs and the separate gain/diversity audit.
