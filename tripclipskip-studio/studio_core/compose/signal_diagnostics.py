"""WP-03A-r1: read-only, reference-relative CLIP layer signal measurements.

This module NEVER certifies that a tensor is a native A1111 output. Matching
model/source/tokenizer strings is metadata consistency, NOT runtime provenance
attestation. Native capture tickets are reserved for WP-03B1.

Metrics describe numerical deviation only; they cannot predict image quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .errors import ComposeError
from .layers import LayerSample, _check_pair, _scalar


@dataclass(frozen=True)
class LayerSignalDiagnostics:
    """Numerical measurements for [batch, tokens, features] CLIP states.

    `reference_assurance` deliberately never says VERIFIED_NATIVE: this pure
    module has no way to inspect which A1111 runtime invocation produced it.
    All scalar metrics use the input tensor's token axis. Undefined values are
    None (e.g. cosine at zero vectors), never fabricated as zero or one.
    """
    status: str
    reference_assurance: str
    encoder: str
    branch: str
    total_tokens: int
    reference_zero_tokens: int | None
    candidate_zero_tokens: int | None
    cosine_defined_tokens: int | None
    nonfinite_fraction: float
    low_signal_fraction: float | None
    min_cosine: float | None
    mean_cosine: float | None
    min_rms_ratio: float | None
    mean_rms_ratio: float | None
    max_rms_ratio: float | None
    max_abs_delta: float | None
    mean_abs_delta: float | None
    warnings: tuple[str, ...]
    high_signal_fraction: float | None = None
    comparison_mode: str = 'same_representation'


def _measure_threshold(value: float, name: str, *, low: float, high: float) -> float:
    return _scalar(value, name=name, lower=low, upper=high)


def diagnose_layer_signal(
    candidate: torch.Tensor,
    *,
    candidate_source: LayerSample,
    reference: LayerSample,
    low_rms_ratio: float = 0.10,
    high_rms_ratio: float = 4.0,
    low_cosine: float = 0.80,
    zero_tolerance: float = 1e-12,
    comparison_mode: str = 'same_representation',
    require_capture_metadata: bool = False,
) -> LayerSignalDiagnostics:
    """Compare a candidate with a *claimed* same-source reference, without mutation.

    Both LayerSample objects are verified only for shape/type/metadata.
    Cross-stage diagnostics require comparison_mode='cross_stage_observational',
    raw candidate_source, native_normalized reference, and matching *asserted*
    layout/invocation metadata; this does not authorize cross-stage MIXING.
    High/low RMS thresholds are alerts, NOT quality or security gates. Even if
    `reference.representation == 'native_normalized'`, no call here proves the
    reference was taken after A1111's correct native LayerNorm and selection.

    Thresholds control diagnostic warnings only: never gate semantic quality or
    modify the candidate. This call does NOT retain tensors in the report.
    """
    low_rms_ratio = _measure_threshold(low_rms_ratio, 'low_rms_ratio', low=1e-9, high=1.0)
    high_rms_ratio = _measure_threshold(high_rms_ratio, 'high_rms_ratio', low=1.0, high=1e12)
    if comparison_mode not in ('same_representation', 'cross_stage_observational'):
        raise ComposeError('Unsupported comparison mode')
    if comparison_mode == 'cross_stage_observational':
        if not isinstance(candidate_source, LayerSample) or not isinstance(reference, LayerSample):
            raise ComposeError('Cross-stage observation requires LayerSample sources')
        if candidate_source.representation != 'raw' or reference.representation != 'native_normalized':
            raise ComposeError('Cross-stage observation requires raw candidate and native_normalized reference')
        # Require at least asserted same-invocation + position alignment. These
        # strings are NOT proof of the native Stage or an actual CLIP capture.
        require_capture_metadata = True
    low_cosine = _measure_threshold(low_cosine, 'low_cosine', low=-1.0, high=1.0)
    zero_tolerance = _measure_threshold(zero_tolerance, 'zero_tolerance', low=0.0, high=1.0)
    _check_pair(candidate_source, reference, capabilities=None,
                require_capture_metadata=require_capture_metadata,
                allow_cross_representation=comparison_mode == 'cross_stage_observational')
    source = candidate_source.tensor
    if not isinstance(candidate, torch.Tensor) or candidate.is_meta or candidate.layout != torch.strided:
        raise ComposeError('Candidate must be a materialized dense torch.Tensor')
    if candidate.shape != source.shape or candidate.device != source.device or candidate.dtype != source.dtype:
        raise ComposeError('Candidate shape/device/dtype differs from source')

    total_tokens = int(source.shape[0] * source.shape[1])
    assurance = 'METADATA_ONLY_NOT_NATIVE_VERIFIED'
    with torch.no_grad():
        # Float64 inputs retain their precision. Lower-precision CLIP states
        # are promoted to FP32 for diagnostic reductions.
        work_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
        c = candidate.detach().to(dtype=work_dtype)
        r = reference.tensor.detach().to(dtype=work_dtype)
        finite = torch.isfinite(c)
        nonfinite_fraction = float((~finite).sum().item()) / candidate.numel()
        if nonfinite_fraction:
            return LayerSignalDiagnostics(
                status='INVALID_NONFINITE', reference_assurance=assurance,
                encoder=candidate_source.encoder, branch=candidate_source.branch,
                total_tokens=total_tokens,
                reference_zero_tokens=None, candidate_zero_tokens=None,
                cosine_defined_tokens=None,
                nonfinite_fraction=nonfinite_fraction,
                low_signal_fraction=None, min_cosine=None, mean_cosine=None,
                min_rms_ratio=None, mean_rms_ratio=None, max_rms_ratio=None,
                max_abs_delta=None, mean_abs_delta=None,
                warnings=('NONFINITE_CANDIDATE', 'NOT_NATIVE_VERIFIED') +
                    (('CROSS_STAGE_NUMERIC_ONLY',) if comparison_mode == 'cross_stage_observational' else ()),
                comparison_mode=comparison_mode,
            )
        # linalg.vector_norm avoids avoidable overflow from x*x on large FP32
        # values. RMS = L2 / sqrt(feature width). Values are detached.
        feature_scale = math.sqrt(source.shape[-1])
        c_norm = torch.linalg.vector_norm(c, ord=2, dim=-1)
        r_norm = torch.linalg.vector_norm(r, ord=2, dim=-1)
        if not bool(torch.isfinite(c_norm).all() and torch.isfinite(r_norm).all()):
            raise ComposeError('Nonfinite intermediate while computing norms')
        c_rms = c_norm / feature_scale
        r_rms = r_norm / feature_scale
        r_nonzero = r_rms > zero_tolerance
        c_nonzero = c_rms > zero_tolerance
        valid_cosine = r_nonzero & c_nonzero
        r_zero = int((~r_nonzero).sum().item())
        c_zero = int((~c_nonzero).sum().item())
        valid_count = int(valid_cosine.sum().item())
        ratios = c_rms[r_nonzero] / r_rms[r_nonzero]
        # Avoid calculating dot products on vectors where cosine is undefined.
        # torch.sum(c*r) may overflow on enormous but finite float32 vectors:
        # cosine is calculated with norm-normalized vectors instead.
        if valid_count:
            unit_c = c[valid_cosine] / c_norm[valid_cosine].unsqueeze(-1)
            unit_r = r[valid_cosine] / r_norm[valid_cosine].unsqueeze(-1)
            cosines = (unit_c * unit_r).sum(dim=-1).clamp(-1.0, 1.0)
            if not bool(torch.isfinite(cosines).all()):
                raise ComposeError('Nonfinite cosine diagnostic')
            min_cosine, mean_cosine = float(cosines.min().item()), float(cosines.mean().item())
        else:
            min_cosine = mean_cosine = None
        if ratios.numel():
            if not bool(torch.isfinite(ratios).all()):
                raise ComposeError('Nonfinite RMS ratio diagnostic')
            min_ratio = float(ratios.min().item())
            mean_ratio = float(ratios.mean().item())
            max_ratio = float(ratios.max().item())
            low_fraction = float((ratios < low_rms_ratio).sum().item()) / ratios.numel()
            high_fraction = float((ratios > high_rms_ratio).sum().item()) / ratios.numel()
        else:
            min_ratio = mean_ratio = max_ratio = low_fraction = high_fraction = None
        delta = torch.abs(c-r)
        if not bool(torch.isfinite(delta).all()):
            raise ComposeError('Nonfinite difference diagnostic')
        warnings = ['NOT_NATIVE_VERIFIED']
        if r_zero:
            warnings.append('REFERENCE_ZERO_TOKENS')
        if c_zero:
            warnings.append('CANDIDATE_ZERO_TOKENS')
        if low_fraction is not None and low_fraction > 0.0:
            warnings.append('LOW_SIGNAL_RELATIVE_TO_REFERENCE')
        if high_fraction is not None and high_fraction > 0.0:
            warnings.append('HIGH_SIGNAL_RELATIVE_TO_REFERENCE')
        if bool((~r_nonzero & c_nonzero).any()):
            warnings.append('CANDIDATE_NONZERO_WHERE_REFERENCE_ZERO')
        if comparison_mode == 'cross_stage_observational':
            warnings.append('CROSS_STAGE_NUMERIC_ONLY')
        if min_cosine is not None and min_cosine < low_cosine:
            warnings.append('DIRECTION_SHIFT_RELATIVE_TO_REFERENCE')
        if valid_count == 0:
            warnings.append('COSINE_UNDEFINED')
        return LayerSignalDiagnostics(
            status=('MEASURED_CROSS_STAGE_NOT_QUALITY_PROOF' if comparison_mode == 'cross_stage_observational'
                    else 'MEASURED_NOT_QUALITY_PROOF'), reference_assurance=assurance,
            encoder=candidate_source.encoder, branch=candidate_source.branch,
            total_tokens=total_tokens, reference_zero_tokens=r_zero,
            candidate_zero_tokens=c_zero, cosine_defined_tokens=valid_count,
            nonfinite_fraction=0.0, low_signal_fraction=low_fraction,
            min_cosine=min_cosine, mean_cosine=mean_cosine,
            min_rms_ratio=min_ratio, mean_rms_ratio=mean_ratio, max_rms_ratio=max_ratio,
            max_abs_delta=float(delta.max().item()),
            mean_abs_delta=float(delta.mean().item()), warnings=tuple(warnings),
            high_signal_fraction=high_fraction, comparison_mode=comparison_mode,
        )
