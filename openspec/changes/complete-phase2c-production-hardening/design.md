## Context

Phase 2B adds content-addressed model acceptance, exact host execution,
physical cycle capture, and transactional board evidence. Its trusted PYNQ run
and approved trained bundle are not yet available, so Issue #8 remains blocked
for implementation acceptance. Phase 2C must consume an immutable Phase 2B
release identity rather than silently accepting the current branch, synthetic
fixture, or locally generated bitstream as its production baseline.

The existing ABI has stable identity, capability, status, error, timeout, and
cycle registers. Existing delivery uses immutable deployment directories and an
active symlink, but rollback verification, soak scheduling, diagnostics, and a
complete evidence graph are not yet production contracts.

## Goals / Non-Goals

**Goals:**

- Make compatibility and recovery state explicit and machine-verifiable.
- Detect stale data, nondeterminism, resource drift, and partial soak runs.
- Bind every production claim to the correct trusted producer and exact inputs.
- Preserve known-good deployment and evidence through promotion or rollback failures.
- Give operators useful bounded diagnostics without exposing tensors or secrets.

**Non-Goals:**

- Change Phase 0 quantized arithmetic or retrain/select a model.
- Treat host, simulation, or synthetic evidence as board or production evidence.
- Add silent compatibility fallbacks, unbounded automatic retries, telemetry
  services, or credential storage.
- Merge Phase 2C ahead of Phase 2B or close Issue #8 before its dependency.

## Decisions

### 1. Phase 2B acceptance identity is a hard input

Phase 2C begins implementation only after Issue #7 merges and publishes a
trusted evidence root with an approved model/corpus and protected-board result.
All Phase 2C manifests reference that root digest. An explicit dependency is
preferred over duplicating or weakening Phase 2B gates.

### 2. ABI lifecycle uses compatible minor extensions

Keep the existing ABI major and numeric contract. Add new observability or
reset semantics through declared capability bits and compatible minor fields;
reject incompatible majors and unknown required bits before side effects.
Changing the major merely to add diagnostics was rejected because it would
invalidate already proven arithmetic and package behavior.

### 3. One state machine owns reset and recovery

Represent idle, running, failed, recovering, and unavailable states explicitly.
Each transition has one bounded deadline and ledger entry. Automatic recovery
is limited to the failed invocation and never replays model work. This avoids
ambiguous nested retries and makes failure campaigns reproducible.

### 4. Soak schedules are content-addressed inputs

A canonical schedule binds corpus IDs, changed-input order, warmup, invocation
and duration minima, deterministic failure points, sampling cadence, and drift
thresholds. Both count and duration must pass. Metrics use bounded summaries;
raw samples and tensors stay in external approved assets.

### 5. Release evidence is a typed directed acyclic graph

Each evidence node declares its type, producer trust class, direct input
digests, schema, and result. Graph validation rejects cycles, missing inputs,
host-to-board substitutions, or a source/board mismatch. A flat bag of hashes
was rejected because it cannot prove which artifact supported which claim.

### 6. Promotion and rollback share the verifier

Candidates and retained rollback targets pass the same offline manifest,
artifact, evidence-graph, compatibility, and revocation verifier before the
active selector changes. Promotion creates evidence before selection; rollback
creates a new event but never edits the old release. Two independent scripts
were rejected because their validation rules would drift.

### 7. Diagnostics are bounded and allowlisted

Health is a read-only snapshot. The in-memory failure ledger uses fixed
capacity and stable categories. Support bundles are canonical archives built
only from allowlisted normalized records; environment variables, raw tensors,
logs, and arbitrary paths are excluded. General log collection was rejected as
too likely to leak credentials and produce unreproducible evidence.

## Risks / Trade-offs

- **Phase 2B remains unavailable** → Keep all apply tasks dependency-gated and
  record the exact evidence root required to unblock them.
- **Long soak time consumes protected board capacity** → Use declared count and
  duration profiles, with a shorter non-production qualification profile that
  cannot satisfy the production gate.
- **Temperature or process metrics vary by platform** → Mark availability
  explicitly; require only metrics supported by the accepted board profile and
  never invent values.
- **Reset injection can stress hardware** → Use deterministic bounded campaign
  points and stop at the first failure to recover.
- **Evidence schemas become complex** → Keep small typed nodes, canonical JSON,
  strict keys, and one graph verifier shared by promotion and rollback.

## Migration Plan

1. Merge and archive Phase 2A, then merge Phase 2B with protected-board evidence.
2. Record the accepted Phase 2B evidence-root digest in the Phase 2C tracking PR.
3. Land compatibility/reset and diagnostics changes with host and RTL tests.
4. Land soak and failure-campaign behavior with deterministic fake evidence.
5. Land release-graph verification, promotion, retention, revocation, and rollback.
6. Rebuild on trusted Vivado, execute the approved board soak, and promote only
   after all graph nodes verify.
7. Roll back by selecting the prior retained release through the same verifier;
   never overwrite either immutable version.
