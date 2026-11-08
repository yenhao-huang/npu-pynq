"""The quantization schemes the NPU accepts.

The array multiplies INT8 by INT8 into INT32. The schemes themselves are defined
in src/export/frontend/quantization/schemes.py; this names the ones it can run,
and the export refuses any other when the package targets the NPU.
"""

ACCEPTED_SCHEMES = ('w8a8', 'w4a8')
