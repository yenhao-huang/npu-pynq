"""The quantization schemes the NPU accepts.

The array multiplies INT8 by INT8 into INT32. The schemes themselves are defined
in src/export/frontend/quantization/schemes.py; this names the ones it can run,
and the export refuses any other when the package targets the NPU.

Copied from src/vendors/npu_pynq/spec/quant.py: the CVA6 build has the same
matrix core, so it accepts the same schemes (plan 2026-1112-003 section 6.2).
"""

ACCEPTED_SCHEMES = ('w8a8', 'w4a8')
