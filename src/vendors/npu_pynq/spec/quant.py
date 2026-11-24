"""The quantization schemes the NPU accepts.

The array multiplies INT8 by INT8 into INT32. The schemes themselves are defined
by the export's quantization front end; this names the ones the NPU can run,
and the export refuses any other when the package targets the NPU.
"""

ACCEPTED_SCHEMES = ('w8a8', 'w4a8')
