# Example alignment acceptance

Validated on 2026-10-25 in the issue-119 integration worktree.

## Source organization

Each public example now contains only README.md, README.macos.md and its
output-free notebook. Existing ignored model files were preserved.

| Responsibility | Owner |
| --- | --- |
| Matrix tiling | src/runtime/core/matrix.py |
| Matrix board cases | src/test/matrix_acceptance.py |
| Image CLI | src/runtime/image_cli.py |
| Image delivery verification | src/runtime/core/delivery.py |
| Standalone builders | src/export/delivery/{matrix,resnet18}.py |
| Pinned ResNet sources and downloaders | src/export/graphs/resnet18/ |
| Deployment automation | .github/cd/matrix_deploy.ps1 and existing ResNet CD scripts |
| Regressions | src/test/tests/test_matrix_*.py and test_resnet18_*.py |

Model products default to dist/resnet18/model. Test discovery, CD cache/upload
paths, package assembly, downstream experiment metadata readers and the root
README use the new owners. Canonical source JSON is pinned to LF in Git attributes.

## Host checks

- `python -X utf8 -m unittest discover -s src/test/tests -q`: 170 tests,
  150 passed, 20 skipped by their existing environment/toolchain gates.
  Ignored log: build/example-alignment/software-tests.log.
- Focused runtime, delivery, pinned download and dependency boundaries: 48 passed.
- Isolated matrix package imports and acceptance module help work with PYTHONPATH
  removed. Isolated ResNet ZIP verification runs its source-owned CLI without
  export or JIT source. Deterministic archive, tamper, path traversal and geometry
  rejection tests pass. PowerShell deployment dry run passes without networking.
- Exact example file lists, 14 local documentation links, nine Bash syntax blocks,
  both notebook ASTs and empty outputs, CD YAML and CD PowerShell syntax pass.
- Source-owned download and builder CLI help entry points pass.
- Strict OpenSpec validation and diff whitespace checks pass.

## Physical checks

The send-to-pynq-board skill performed a dry run followed by transfer to
`/home/xilinx/jupyter_notebooks/npu-example-alignment-20261025/`.
The previously exercised 16 x 16, MAX_K=256 BIT/HWH pair was reused:

- BIT SHA-256: bd6b8f44f2b25445aef61e7692beda62d85a35a84ccbf6448ce40ca4cc12f3db
- HWH SHA-256: 8ba3849b16c63faba539511c997a400f8b1f64c4f7f8707fae0f4a5af9d46748

The new image CLI used an existing compiled board model and sample image.
Top-1 was Samoyed, probability 0.898022755, with 21 NPU calls and 35,174
physical jobs. The revised six-stage ResNet notebook then executed with nbclient
in the root PYNQ kernel: Samoyed, CORRECT, 35,174 jobs and 40,077,077 cycles.
The summed six block durations were 28.809 s, including preprocessing/plotting;
inference alone was 4.851 s. No gallery images were available in the reused
model directory; the bundled sample path was exercised. Interactive uploads
and gallery downloads were not exercised.

The relocated matrix runtime and execute_cases helper passed all three physical
cases: 16x5 by 5x16 (one tile), 17x5 by 5x17 (four tiles), and a repeated signed
job (four tiles), all exactly equal to NumPy INT32 results. This is runtime
smoke evidence, not a complete provenance-checking package acceptance: the
reused pair's original build manifest was unavailable locally. No replacement
source commit or manifest was invented. The complete matrix notebook and
manifest-verifying CLI require the original manifest or a new clean build.

Remote evidence: image-acceptance.json, matrix-runtime-acceptance.json,
resnet18.executed.ipynb and resnet18-notebook.log. Local ignored evidence lives
under build/example-alignment/.

## Limits

No fresh Vivado build, model recompilation, release/CD run or native macOS
execution was performed. Arithmetic, RTL, vendor driver and numeric contracts
were unchanged, so RTL lint/simulation and synthesis were not applicable to this
relocation; no push was requested. No commit, push, PR, archive or human-owned
document update was performed. Pre-existing staged and unstaged work was retained.
