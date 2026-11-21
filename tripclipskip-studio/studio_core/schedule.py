"""Pure timing arithmetic; independent of a specific sampler."""
import math


def switched(step: int, total_calls: int, threshold: float) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('Switch fraction must be in 0..1')
    if not total_calls or total_calls <= 0:
        return False
    # At threshold 1.0, never switch inside this pass.
    if threshold >= 1.0:
        return False
    return step >= math.ceil(total_calls * threshold)
