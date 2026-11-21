"""Math and layer mapping; standalone PyTorch unit-testable helpers."""
from __future__ import annotations

import math


def skip_pair(value: float, maximum: int):
    if maximum < 1:
        raise ValueError('CLIP encoder has no selectable layers')
    v = float(value)
    if not math.isfinite(v) or not (1.0 <= v <= maximum):
        raise ValueError(f'CLIP skip {value} is outside the supported range 1..{maximum}')
    lower = math.floor(v)
    upper = min(lower + 1, maximum)
    return lower, upper, v - lower


def interpolate(a, b, t: float, method: str = 'lerp'):
    """Safe FP32 intermediate, preserves output dtype, and handles zero vectors.

    Token vectors are interpolated over the last dimension; no CLIP-L/G mixing.
    """
    import torch

    if a.shape != b.shape:
        raise ValueError(f'Incompatible conditioning shapes {a.shape} != {b.shape}')
    if method not in ('lerp', 'slerp'):
        raise ValueError(f'Unknown interpolation method {method!r}')
    if not 0 <= t <= 1:
        raise ValueError(f'Interpolation fraction {t} is outside [0,1]')
    if t == 0:
        return a
    if t == 1:
        return b
    if method == 'lerp':
        return torch.lerp(a, b, float(t))

    lo, hi = a.float(), b.float()
    eps = 1e-6
    norm_lo = torch.linalg.vector_norm(lo, dim=-1, keepdim=True)
    norm_hi = torch.linalg.vector_norm(hi, dim=-1, keepdim=True)
    unit_lo = lo / norm_lo.clamp_min(eps)
    unit_hi = hi / norm_hi.clamp_min(eps)
    dot = (unit_lo * unit_hi).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    spherical = (
        torch.sin((1.0 - t) * omega) / sin_omega.clamp_min(eps) * lo
        + torch.sin(t * omega) / sin_omega.clamp_min(eps) * hi
    )
    linear = torch.lerp(lo, hi, t)
    degenerate = (norm_lo <= eps) | (norm_hi <= eps) | (sin_omega.abs() <= eps)
    out = torch.where(degenerate, linear, spherical)
    return out.to(dtype=a.dtype)
