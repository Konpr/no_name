"""Multi-phase STEP schedule for A1111 native conditioning.

Pure arithmetic and validation, without A1111 imports or GPU allocations.
One phase is active at a time; no conditioning recomputation during denoising.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Phase:
    start: float
    clip_l: float
    clip_g: float


@dataclass(frozen=True)
class PhaseSchedule:
    phases: tuple[Phase, ...]
    allow_legacy_endpoints: bool = False

    def __post_init__(self):
        if not 2 <= len(self.phases) <= 4:
            raise ValueError('Preview supports 2 to 4 phases')
        previous = -1.0
        for i, p in enumerate(self.phases):
            if any(not math.isfinite(float(v)) for v in (p.start, p.clip_l, p.clip_g)):
                raise ValueError(f'Phase {i+1} has a non-finite value')
            if i == 0 and p.start != 0:
                raise ValueError('First phase must start at 0%')
            legacy_endpoint = (self.allow_legacy_endpoints and len(self.phases) == 2
                               and i == 1 and p.start in (0.0, 1.0))
            if not legacy_endpoint and (not 0 <= p.start < 1 or p.start <= previous):
                raise ValueError('Phase boundaries must strictly increase within 0..100%')
            if p.clip_l < 1 or p.clip_g < 1:
                raise ValueError('CLIP depth must be at least 1')
            previous = p.start

    def phase_index(self, step: int, total_calls: int) -> int:
        """Index by sampler denoiser calls; 100% is not an in-pass phase."""
        if total_calls is None or total_calls <= 0:
            return 0
        result = 0
        for i, phase in enumerate(self.phases[1:], 1):
            if phase.start >= 1.0:
                break  # Legacy Dual-Phase at 100% never switches.
            if step >= math.ceil(total_calls * phase.start):
                result = i
            else:
                break
        return result

    @property
    def boundaries(self) -> tuple[float, ...]:
        return tuple(phase.start for phase in self.phases)


def schedule_from_ui(first_l, first_g, late_l, late_g, fraction, *,
                     multi_count=0, third_l=2.5, third_g=2.5,
                     fourth_l=3.0, fourth_g=3.0,
                     start2=0.25, start3=0.5, start4=0.75,
                     phase2_l=None, phase2_g=None) -> PhaseSchedule:
    """Legacy Dual-Phase is exactly a two-phase schedule; extra phases are opt-in."""
    base = Phase(0.0, float(first_l), float(first_g))
    if int(multi_count) == 0:
        return PhaseSchedule((base, Phase(float(fraction), float(late_l), float(late_g))),
                             allow_legacy_endpoints=True)
    if int(multi_count) not in (3, 4):
        raise ValueError('Multi-Phase count must be 3 or 4')
    # Multi-Phase has its own Phase 2 controls.  Old presets/callers fall
    # back to the historical Dual-Phase values.
    p2_l = late_l if phase2_l is None else phase2_l
    p2_g = late_g if phase2_g is None else phase2_g
    phases = [base, Phase(float(start2), float(p2_l), float(p2_g)),
              Phase(float(start3), float(third_l), float(third_g))]
    if int(multi_count) == 4:
        phases.append(Phase(float(start4), float(fourth_l), float(fourth_g)))
    return PhaseSchedule(tuple(phases))


def effective_phase2_g(*, independent: bool, phase2_l, phase2_g, legacy_l, legacy_g) -> float:
    """Apply the same independent-G policy to Phase 2 as every other phase.

    None represents an older preset that has no dedicated Multi-Phase Phase 2.
    Never use the hidden G slider when independent control is disabled.
    """
    chosen_l = legacy_l if phase2_l is None else phase2_l
    chosen_g = legacy_g if phase2_g is None else phase2_g
    return float(chosen_g if independent else chosen_l)


def conditioning_signature(obj, depth=0):
    """Conservative, shallow structural checks without copying large tensors.

    Includes SDXL crossattn/pooled tensors and prompt-parser containers, but
    intentionally skips arbitrary opaque internals (A1111 has dynamic classes).
    """
    if depth > 8:
        raise ValueError('Conditioning structure is too deeply nested')
    try:
        import torch
        if torch.is_tensor(obj):
            return ('tensor', tuple(obj.shape), str(obj.dtype),
                    conditioning_signature(obj.pooled, depth+1) if hasattr(obj, 'pooled') else None)
    except ImportError:
        pass
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(conditioning_signature(x, depth+1) for x in obj))
    if isinstance(obj, dict):
        return ('dict', tuple((k, conditioning_signature(obj[k], depth+1)) for k in sorted(obj)))
    if obj is None or isinstance(obj, (int, str, bool, float)):
        return type(obj).__name__
    fields = []
    for attr in ('crossattn', 'vector', 'pooled', 'conds'):
        if hasattr(obj, attr):
            fields.append((attr, conditioning_signature(getattr(obj, attr), depth+1)))
    return (type(obj).__name__, tuple(fields))


def validate_conditioning_pair(reference, candidate):
    """Reject incompatible positive/negative structures before switching."""
    a, b = reference, candidate
    if conditioning_signature(a[0]) != conditioning_signature(b[0]) or \
       conditioning_signature(a[1]) != conditioning_signature(b[1]):
        raise ValueError('Phase conditioning structures differ (prompt schedule/SDXL shape/pooled); cannot switch safely')
