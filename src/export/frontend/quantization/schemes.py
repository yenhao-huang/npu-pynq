"""The quantization schemes, as data: their parameters and the ops each one quantizes.

Values lie in [-qmax, qmax] unless qmin is given. ``ops`` names the layers a
scheme rewrites, framework independent: linear, conv2d, embedding."""

EPS = 1e-8   # smallest scale denominator, before dividing by qmax

# W8A8: per-output-channel INT8 weights, quantized offline; static per-tensor
# INT8 activations, one scale per layer input, fixed by calibration (the
# largest |input| seen); both symmetric with zero point 0, round half to even.
W8A8 = dict(name='w8a8', weight_bits=8, weight_qmax=127, weight_granularity='per_channel',
            activation_bits=8, activation_qmax=127, activation_granularity='per_tensor',
            activation_static=True, activation_calibration='max', zero_point=0,
            symmetric=True, rounding='half_even', eps=EPS,
            ops=('linear', 'conv2d', 'embedding'))

# W4A8: group-wise INT4 weights in [-8, 7] (one scale per K group and output
# channel, scale = amax / 7) and dynamic per-token INT8 activations. Each group's INT32
# partial product is rescaled on its own, so the per-group scales stay exact.
W4A8 = dict(name='w4a8', weight_bits=4, weight_qmax=7, weight_qmin=-8, weight_granularity='per_group',
            groups=(8, 64, 128), activation_bits=8, activation_qmax=127,
            activation_granularity='per_token', symmetric=True, rounding='half_even', eps=EPS)

SCHEMES = {s['name']: s for s in (W8A8, W4A8)}
