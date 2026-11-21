"""Read-only mapping of requested relative TCS depth to actual CLIP layer outputs.

No changes to tensors or model options.  Used for PNG Info diagnostics and as
an initial shared mapping for a future visual/absolute layer timeline.
"""
from __future__ import annotations

from studio_core.interpolation import skip_pair


def _blend(a: str, b: str, fraction: float) -> str:
    if fraction == 0:
        return a
    return f'{1 - fraction:.0%} {a} + {fraction:.0%} {b}'


def clip_l_layer_map(reference, *, depth: int, relative: float, family: str) -> str:
    """Report HF hidden_states indexes and explicitly identify normalization."""
    lower, upper, t = skip_pair(relative, reference.index + 1)
    def label(relative_integer):
        index = reference.index - (relative_integer - 1)
        if index == depth and reference.use_final_hidden_state:
            return 'last_hidden_state (final LayerNorm)'
        if index == 0:
            token = 'hidden_states[0] (input embedding)'
        else:
            token = f'hidden_states[-{depth + 1 - index}]'
        if family == 'sd1' and reference.sd1_native_skip + relative_integer - 1 > 1:
            token += ' (final LayerNorm)'
        return token
    return _blend(label(lower), label(upper), t)


def clip_g_layer_map(*, blocks: int, model_anchor: str, relative: float) -> str:
    """Report one-based OpenCLIP-G transformer block numbers, not HF indices."""
    base = blocks - (1 if model_anchor == 'penultimate' else 0)
    lower, upper, t = skip_pair(relative, base)
    return _blend(f'block {base - lower + 1}/{blocks}',
                  f'block {base - upper + 1}/{blocks}', t) + '; pooled=native'
