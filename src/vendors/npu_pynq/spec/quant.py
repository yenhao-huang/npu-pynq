"""Quantization formats the NPU accepts.

The array multiplies INT8 by INT8 into INT32. A model reaches it only through
these schemes; the export front end quantizes to them and rejects anything
else (src/export/frontend/load/check.py). The rounding and saturation rules
are those of src/test/model, the numeric contract the RTL implements.
"""

EPS = 1e-8   # smallest scale denominator, before dividing by QMAX

# W8A8: per-output-channel INT8 weights, per-row (per-token) INT8 activations,
# both symmetric, values in [-QMAX, QMAX], round half to even.
W8A8 = dict(name='w8a8', weight_bits=8, weight_qmax=127, weight_granularity='per_channel',
            activation_bits=8, activation_qmax=127, activation_granularity='per_token',
            symmetric=True, rounding='half_even', eps=EPS)

# W4A8: group-wise INT4 weights in [-8, 7] (one scale per K group and output
# channel, scale = amax / 7), the same INT8 activations. Each group's INT32
# partial product comes back separately, so the per-group scales stay exact.
W4A8 = dict(name='w4a8', weight_bits=4, weight_qmax=7, weight_qmin=-8, weight_granularity='per_group',
            groups=(8, 64, 128), activation_bits=8, activation_qmax=127,
            activation_granularity='per_token', symmetric=True, rounding='half_even', eps=EPS)

SCHEMES = {s['name']: s for s in (W8A8, W4A8)}
