# Unreleased v1.0.0 Upload

This file records the exact `dev` integration proposed for the current
`main` pull request. Changelog-only commits made after the integration commit
do not change this implementation boundary.

## Change commits

- Change-before commit: `665faf309d69d162e90962d40812899ac6524153`
  (`main` when release preparation started)
- Change-after commit: `81c7394fbd70fcccbf93f017b3b76b905216696e`
  (release branch integration of `dev` at
  `49f14a6bb0be4fed0c749777965a25b28552ba03`, followed by removal of the
  superseded pre-#48 delivery test contract)

## Roadmap progress reached

| Roadmap phase | Proposed state | Evidence |
| --- | --- | --- |
| Phase 2A — ResNet enablement | Complete | Issues #6 and #33–#36; PRs #37, #38, #40–#42 |
| Phase 2B — ResNet-18 acceptance | Complete with recorded provenance limitation | Issues #7 and #47; PRs #46 and #48; physical notebook run on 2026-09-04 |
| Phase 2C — Production hardening | Not included | Issue #8 closed without a merged implementation PR |

The physical notebook run completed 2,104,040 physical jobs in 28,031.949
seconds and matched the independent host digests for all three recorded
captures. The run deliberately allowed an artifact/source commit mismatch, so
it proves physical behavior in development mode but is not strict same-commit
release provenance.

## Issues and corresponding pull requests

### feat — batch 1

| Issue | Pull request | Relationship | Result proposed for `main` |
| --- | --- | --- | --- |
| [#33](https://github.com/yenhao-huang/npu_in_pynq/issues/33) | [#37](https://github.com/yenhao-huang/npu_in_pynq/pull/37) | Direct | Quantized graph and operator contracts |
| [#34](https://github.com/yenhao-huang/npu_in_pynq/issues/34) | [#38](https://github.com/yenhao-huang/npu_in_pynq/pull/38) | Direct | Deterministic package export and memory planning |
| [#35](https://github.com/yenhao-huang/npu_in_pynq/issues/35) | [#40](https://github.com/yenhao-huang/npu_in_pynq/pull/40) | Integrated dependency | Bounded matrix lowering was integrated with runtime execution |
| [#36](https://github.com/yenhao-huang/npu_in_pynq/issues/36) | [#40](https://github.com/yenhao-huang/npu_in_pynq/pull/40) | Direct | Validated package execution |
| [#7](https://github.com/yenhao-huang/npu_in_pynq/issues/7) | [#46](https://github.com/yenhao-huang/npu_in_pynq/pull/46) | Direct | ResNet-18 host and board acceptance workflow |

### feat — batch 2

| Issue | Pull request | Relationship | Result proposed for `main` |
| --- | --- | --- | --- |
| [#47](https://github.com/yenhao-huang/npu_in_pynq/issues/47) | [#48](https://github.com/yenhao-huang/npu_in_pynq/pull/48) | Direct | Pinned TorchVision ResNet-18 import, export, and transparent board demo |

### docs — batch 1

| Issue | Pull request | Relationship | Result proposed for `main` |
| --- | --- | --- | --- |
| [#6](https://github.com/yenhao-huang/npu_in_pynq/issues/6) | [#41](https://github.com/yenhao-huang/npu_in_pynq/pull/41) | Direct | Phase 2A OpenSpec definition and implementation graph |
| [#33](https://github.com/yenhao-huang/npu_in_pynq/issues/33) | [#42](https://github.com/yenhao-huang/npu_in_pynq/pull/42) | Follow-up | Production model contract manual |

### no merged implementation PR — batch 1

| Issue | Pull request | Relationship | Result proposed for `main` |
| --- | --- | --- | --- |
| [#8](https://github.com/yenhao-huang/npu_in_pynq/issues/8) | — | Closed without implementation | Phase 2C is not part of this upload |

### release — batch 1

| Issue | Pull request | Relationship | Result proposed for `main` |
| --- | --- | --- | --- |
| [#47](https://github.com/yenhao-huang/npu_in_pynq/issues/47) | [#51](https://github.com/yenhao-huang/npu_in_pynq/pull/51) | Promotion and integration follow-up | Promote Phase 2A/2B and remove the delivery test superseded by #48 |
