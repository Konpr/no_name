"""Small, tensor-free per-job diagnostics for the STEP scheduler.

The trace records *observed* denoiser calls, not requested steps. It does
not infer the true number of future calls or try to move boundaries mid-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class PhaseTrace:
    phase_count: int
    mode: str = 'denoiser'
    # Requested schedule boundaries; used ONLY for diagnostics.
    # The actual phase selection is always performed by PhaseSchedule.
    starts: tuple[float, ...] = ()
    counts: list[int] = field(init=False)
    transitions: list[tuple[int, int, int, int]] = field(default_factory=list)
    totals_seen: set[int] = field(default_factory=set)
    call_count: int = 0
    last_step: int | None = None
    last_total: int | None = None
    max_step: int | None = None
    nonmonotonic: int = 0
    unavailable: int = 0
    fallbacks: int = 0

    def __post_init__(self):
        if not 2 <= self.phase_count <= 4:
            raise ValueError('PhaseTrace supports 2–4 phases')
        if self.mode not in ('denoiser', 'sampler'):
            raise ValueError('Unknown phase progress source')
        if self.starts and (len(self.starts) != self.phase_count or
                            any(not math.isfinite(float(x)) for x in self.starts) or
                            self.starts[0] != 0 or
                            (any(a >= b for a, b in zip(self.starts, self.starts[1:])) and
                             not (self.phase_count == 2 and self.starts == (0.0, 0.0)))):
            raise ValueError('PhaseTrace boundaries do not match the requested phases')
        self.counts = [0] * self.phase_count

    def record(self, phase: int, step: int | None, total: int | None) -> None:
        if not 0 <= phase < self.phase_count:
            raise ValueError('Invalid observed phase')
        if step is None or total is None or total <= 0 or \
                not math.isfinite(float(step or 0)) or not math.isfinite(float(total or 0)):
            self.unavailable += 1
            self.counts[phase] += 1
            self.call_count += 1
            return
        step, total = int(step), int(total)
        if self.last_step is not None and step < self.last_step:
            self.nonmonotonic += 1
        self.last_step, self.last_total = step, total
        self.max_step = step if self.max_step is None else max(self.max_step, step)
        self.totals_seen.add(total)
        if not self.transitions or self.transitions[-1][1] != phase:
            # call number, phase index, progress counter, expected progress total
            self.transitions.append((self.call_count, phase, step, total))
        self.counts[phase] += 1
        self.call_count += 1

    @property
    def missing(self):
        return [i + 1 for i, n in enumerate(self.counts) if not n]

    def compact_transitions(self, max_items=16):
        entries = self.transitions[:max_items]
        result = '; '.join(f'P{p+1}@{step}/{total} (call {call+1})'
                           for call, p, step, total in entries)
        if len(self.transitions) > max_items:
            result += '; ...'
        return result or 'none'

    def planned_thresholds(self):
        """Readable requested phase thresholds for the last known total.

        These are *not* proof that every phase was reached.
        """
        if not self.starts or not self.last_total or self.last_total <= 0:
            return 'unknown until first valid denoiser progress'
        return '; '.join(
            f'P{i+1}@{math.ceil(self.last_total * start)}/{self.last_total}'
            for i, start in enumerate(self.starts)
        )

    def warnings(self):
        out = []
        # Legacy 2-phase endpoints deliberately select only one phase.
        # Their zero-call phase is not a failed or missed transition.
        missing = self.missing
        if self.phase_count == 2 and self.starts == (0.0, 0.0):
            missing = [number for number in missing if number != 1]
        elif self.phase_count == 2 and self.starts == (0.0, 1.0):
            missing = [number for number in missing if number != 2]
        if missing:
            out.append('phases never observed: ' + ', '.join(map(str, missing)))
            if self.starts and self.last_total and self.max_step is not None:
                for number in missing:
                    if number == 1:
                        continue
                    target = math.ceil(self.last_total * self.starts[number - 1])
                    if self.max_step < target:
                        out.append(f'P{number} starts at {target}/{self.last_total}, '
                                   f'but maximum observed progress was {self.max_step}/{self.last_total}')
        if len(self.totals_seen) > 1:
            out.append('expected progress total changed: ' + ','.join(map(str, sorted(self.totals_seen))))
        if self.nonmonotonic:
            out.append(f'progress counter decreased {self.nonmonotonic} time(s)')
        if self.unavailable:
            out.append(f'progress counter unavailable {self.unavailable} time(s)')
        if self.fallbacks:
            out.append(f'sampler progress unavailable: used denoiser counter {self.fallbacks} time(s)')
        if self.last_total and self.last_step is not None and self.last_step + 1 < self.last_total:
            # In the user's real SDXL/Euler run, sampler callback state was
            # 0, 0, 1, ..., 18 for 20 denoiser forwards.  The callback is
            # updated *after* the forward: 18/20 at the last call is NOT an
            # early stop when all 20 forwards were actually observed.
            one_step_sampler_callback_lag = (
                self.mode == 'sampler' and self.last_step == self.last_total - 2
                and self.call_count == self.last_total and not self.nonmonotonic
                and not self.fallbacks and not self.unavailable
                and self.totals_seen == {self.last_total}
            )
            if not one_step_sampler_callback_lag:
                context = ('counter mismatch; expected number of forwards observed'
                           if self.call_count >= self.last_total
                           else 'may be early stop or counter mismatch')
                out.append(f'last observed progress {self.last_step+1}/{self.last_total} ({context})')
        return out

    def metadata(self):
        last = ('unknown' if self.last_step is None else f'{self.last_step}/{self.last_total}')
        return {
            'TCS Phase progress source': self.mode,
            'TCS Observed phase calls': ','.join(map(str, self.counts)),
            'TCS Phase last progress': last,
            'TCS Observed transitions': self.compact_transitions(),
            'TCS Planned phase thresholds': self.planned_thresholds(),
            'TCS Actual forward calls': self.call_count,
            'TCS Phase trace warning': '; '.join(self.warnings())[:350] if self.warnings() else 'none',
        }
