import torch as _torch

import decimal as _decimal

def pe_vec(minimum, maximum, num):
    minimum = _decimal.Decimal(minimum)
    maximum = _decimal.Decimal(maximum)
    _num = _decimal.Decimal(num)
    d_range = minimum / maximum
    vec = [int((1 << 63) * (d_range ** (x / (_num - 1))) / minimum) for x in range(num)]
    _torch.tensor(vec, dtype=_torch.uint64)
    return vec
