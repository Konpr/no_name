"""WP-00 CPU/GPU-safe *pure* comparisons; no instrumentation or A1111 hooks.

This helper does not prove a native-equivalence experiment was conducted. Both
inputs must have been acquired at the same point of the pipeline by the caller.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TensorProof:
    exact: bool
    shape_equal: bool
    dtype_equal: bool
    device_equal: bool
    max_abs_diff: float | None
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    warning: str = ''
    status: str = 'UNKNOWN'
    mean_abs_diff: float | None = None
    max_relative_diff: float | None = None
    reference_finite: bool | None = None
    candidate_finite: bool | None = None


def compare_tensors(reference, candidate) -> TensorProof:
    """Return a diagnostic result rather than attempting unsafe broadcasts/casts.

    In particular EMPTY and NONFINITE are *not* accepted as exact proof even
    when torch.equal happens to return True. No hidden CPU/GPU device transfer.
    """
    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise TypeError('TensorProof only accepts real torch tensors')

    shape_equal = reference.shape == candidate.shape
    dtype_equal = reference.dtype == candidate.dtype
    device_equal = reference.device == candidate.device
    kw = dict(shape_equal=shape_equal, dtype_equal=dtype_equal,
              device_equal=device_equal, reference_shape=tuple(reference.shape),
              candidate_shape=tuple(candidate.shape))

    def result(status, warning='', *, max_abs=None, mean_abs=None, rel=None,
               ref_finite=None, cand_finite=None, exact=False):
        return TensorProof(exact=exact, max_abs_diff=max_abs,
                           warning=warning, status=status, mean_abs_diff=mean_abs,
                           max_relative_diff=rel, reference_finite=ref_finite,
                           candidate_finite=cand_finite, **kw)

    if not shape_equal:
        return result('SHAPE_MISMATCH', 'Shape mismatch; do not compare values')
    if not device_equal:
        return result('DEVICE_MISMATCH', 'Device mismatch; no implicit device transfer')
    if not dtype_equal:
        return result('DTYPE_MISMATCH', 'Dtype mismatch; no implicit cast for equivalence')
    if reference.numel() == 0:
        return result('EMPTY', 'Empty tensors do not provide conditioning equivalence evidence')
    if reference.device.type == 'meta':
        return result('UNMATERIALIZED', 'Meta tensors contain no materialized values')

    if reference.is_complex() or candidate.is_complex():
        return result('UNSUPPORTED_DTYPE', 'Complex tensors not supported by this proof')

    if reference.dtype.is_floating_point:
        finite_ref = bool(torch.isfinite(reference).all().item())
        finite_cand = bool(torch.isfinite(candidate).all().item())
        if not (finite_ref and finite_cand):
            return result('NONFINITE', 'Non-finite values present; no numeric proof',
                          ref_finite=finite_ref, cand_finite=finite_cand)
        # Preserve float64 accuracy; use float32 intermediate for fp16/bfloat16.
        working = torch.float64 if reference.dtype == torch.float64 else torch.float32
        r = reference.detach().to(working)
        c = candidate.detach().to(working)
        difference = (r - c).abs()
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        # Stabilized relative error: near-zero values use absolute error / 1.
        relative = difference / torch.maximum(r.abs(), torch.ones_like(r))
        max_relative = float(relative.max().item())
        values_equal = bool(torch.equal(reference, candidate))
        return result('EXACT' if values_equal else 'DIFFERENT',
                      max_abs=max_abs, mean_abs=mean_abs, rel=max_relative,
                      ref_finite=True, cand_finite=True, exact=values_equal)

    if reference.dtype == torch.bool or reference.dtype in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        values_equal = bool(torch.equal(reference, candidate))
        return result('EXACT' if values_equal else 'DIFFERENT', exact=values_equal,
                      warning='Non-floating tensor: no numerical difference statistics')

    return result('UNSUPPORTED_DTYPE', f'Unsupported tensor dtype: {reference.dtype}')
