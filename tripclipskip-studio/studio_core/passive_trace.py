"""Opt-in, read-only diagnostics for sampler/denoiser progress and sigma.

The recorder does not choose phases or mutate incoming tensors. Its full per-
forward output belongs in the terminal; PNG Info only contains bounded summary.
Reading CUDA sigma values may synchronize the GPU, so this is OFF by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


def _number(value):
    """Convert optional progress counters without manufacturing missing values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def sigma_summary(value):
    """Read small sigma vector for diagnostics, never edit it.

    For tensors this *may* synchronize CUDA. The caller must be opted in and
    exceptions are caught at the denoiser wrapper (diagnostics must not break jobs).
    """
    if value is None:
        return 'unavailable'
    if isinstance(value, (int, float)):
        x = float(value)
        return f'{x:.6g}' if math.isfinite(x) else 'nonfinite'
    if hasattr(value, 'detach') and hasattr(value, 'numel'):
        z = value.detach()
        count = int(z.numel())
        if count == 0:
            return 'empty'
        if count > 64:
            # Bounded GPU transfer; the representative value is not a range.
            x = float(z.reshape(-1)[0].item())
            return f'{x:.6g} (first of {count})' if math.isfinite(x) else 'nonfinite'
        values = [float(v) for v in z.reshape(-1).float().cpu().tolist()]
    elif isinstance(value, (tuple, list)):
        if not value:
            return 'empty'
        if len(value) > 64:
            return sigma_summary(value[0]) + f' (first of {len(value)})'
        values = [float(v) for v in value]
    else:
        try:
            return sigma_summary(float(value))
        except (TypeError, ValueError, OverflowError):
            return 'unsupported'
    if not all(math.isfinite(v) for v in values):
        return 'nonfinite'
    small, large = min(values), max(values)
    return f'{small:.6g}' if small == large else f'{small:.6g}..{large:.6g}'


def _fmt(value):
    return '-' if value is None else str(value)


@dataclass
class PassiveForwardTrace:
    """One recorder for the current job; base/Hires are labelled separately.

    A bounded terminal log avoids unbounded text for long jobs. Do not claim
    these aggregated counts are per-image for batch or n_iter > 1.
    """
    max_log_calls: int = 160
    sampler: str = 'unknown'
    calls: int = 0
    base_calls: int = 0
    hires_calls: int = 0
    first: str = ''
    last: str = ''
    samples: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.max_log_calls, int) or self.max_log_calls < 1:
            raise ValueError('max_log_calls must be a positive integer')

    def record(self, *, hr, denoiser_step, denoiser_total, sampler_step,
               sampler_total, sigma) -> str | None:
        number = self.calls + 1
        pass_name = 'hires' if hr else 'base'
        ds, dt = _number(denoiser_step), _number(denoiser_total)
        ss, st = _number(sampler_step), _number(sampler_total)
        row = (f'call={number} pass={pass_name} sampler_name={self.sampler} '
               f'denoiser={_fmt(ds)}/{_fmt(dt)} '
               f'sampler={_fmt(ss)}/{_fmt(st)} sigma={sigma_summary(sigma)}')
        self.calls = number
        self.hires_calls += int(bool(hr))
        self.base_calls += int(not hr)
        if not self.first:
            self.first = row
        self.last = row
        if len(self.samples) < 8:
            self.samples.append(row)
        return row if number <= self.max_log_calls else None

    def metadata(self):
        result = {
            'TCS Passive trace': f'ON (per forward, job aggregate; console cap {self.max_log_calls})',
            'TCS Passive sampler': self.sampler,
            'TCS Passive forwards': self.calls,
            'TCS Passive base/hires': f'{self.base_calls}/{self.hires_calls}',
            'TCS Passive last call': self.last[:160] if self.last else 'none',
        }
        if self.calls > self.max_log_calls:
            result['TCS Passive trace warning'] = (f'Console output truncated after {self.max_log_calls} calls; '
                                                   f'{self.calls - self.max_log_calls} further calls counted')
        return result
