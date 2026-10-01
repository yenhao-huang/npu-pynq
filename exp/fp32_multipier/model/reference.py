"""Exact integer FP12 E5M6 oracle. No floating-point arithmetic."""
def multiply(a: int, b: int) -> int:
    sign = (a ^ b) & 0x800
    ea, eb = (a >> 6) & 31, (b >> 6) & 31
    fa, fb = a & 63, b & 63
    if (ea == 31 and fa) or (eb == 31 and fb):
        return 0x7E0
    if ea == 31 or eb == 31:
        return 0x7E0 if (a & 0x7FF) == 0 or (b & 0x7FF) == 0 else sign | 0x7C0
    if (a & 0x7FF) == 0 or (b & 0x7FF) == 0:
        return sign
    p = (fa + (64 if ea else 0)) * (fb + (64 if eb else 0))
    scale = max(ea, 1) + max(eb, 1) - 42
    exponent = p.bit_length() - 1 + scale
    quantum = max(exponent - 6, -20)
    shift = quantum - scale
    if shift > 0:
        q, rem = divmod(p, 1 << shift)
        q += rem > (1 << (shift - 1)) or (rem == (1 << (shift - 1)) and q & 1)
    else:
        q = p << -shift
    if q == 128:
        q >>= 1
        exponent += 1
    if exponent > 15:
        return sign | 0x7C0
    if exponent < -14:
        return sign | q
    return sign | ((exponent + 15) << 6) | (q & 63)
