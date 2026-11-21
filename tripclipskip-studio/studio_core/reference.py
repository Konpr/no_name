"""Per-job reference-layer mapping for native A1111 CLIP settings.

This never changes global shared.opts. A fractional depth of 1.0 means the
reference selected by this module (not necessarily model-train defaults).
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def native_integer_skip(value, *, origin='Custom native A1111 Clip Skip') -> int:
    """Validate integer-only *reference* depth, not fractional TCS depth.

    Never round 2.5 to 2: that silently changes the selected reference layer.
    """
    try:
        raw = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f'{origin} must be a whole number (1, 2, 3, ...); got {value!r}. '
            'For fractional values use TCS CLIP-L relative depth instead.'
        ) from exc
    if not math.isfinite(raw) or not raw.is_integer():
        raise ValueError(
            f'{origin} must be a whole number (1, 2, 3, ...); got {value!r}. '
            'For fractional values use TCS CLIP-L relative depth instead.'
        )
    return int(raw)


@dataclass(frozen=True)
class LayerReference:
    index: int                         # zero-based HF hidden_states index
    sd1_native_skip: int = 1           # SD1 needs layer_norm on skipped layers
    use_final_hidden_state: bool = False  # HF final layer_norm for model 'last'
    description: str = ''


def reference_layer(mode: str, wrapped, *, source='a1111', manual_skip=1,
                    apply_to_sdxl_l=False, opts=None, depth=12) -> LayerReference:
    """Return a reference anchored to the *real* upstream A1111 semantics.

    Source: 'a1111' reads shared.opts, 'model' uses model's original config,
    'custom' uses a per-job integer and (for SDXL) its apply-to-L toggle.
    For SDXL the upstream A1111 override selects hidden_states[-skip], i.e.
    native layer_norm is NOT applied; this intentionally differs from SD1.
    """
    if source not in ('a1111', 'model', 'custom'):
        raise ValueError(f'Unknown CLIP reference source: {source!r}')
    if mode not in ('sd1', 'sdxl'):
        raise ValueError(f'Unsupported CLIP model family: {mode!r}')
    if source == 'a1111':
        if opts is None:
            raise ValueError('A1111 reference requires shared.opts')
        apply = bool(getattr(opts, 'sdxl_clip_l_skip', False))
        # For SDXL CLIP-L with override disabled, A1111 ignores global skip;
        # do not reject a global value which this encoder never consumes.
        if mode == 'sdxl' and not apply:
            skip = 1
        else:
            skip = native_integer_skip(
                getattr(opts, 'CLIP_stop_at_last_layers', 1),
                origin='Global A1111 CLIP_stop_at_last_layers',
            )
    elif source == 'model':
        skip, apply = 1, False
    else:
        apply = bool(apply_to_sdxl_l)
        # Native SDXL ignores the integer reference unless its apply-to-L
        # toggle is enabled.  A stale/inapplicable custom slider value must
        # not disable TCS fractional CLIP-L for this job.
        if mode == 'sdxl' and not apply:
            skip = 1
        else:
            skip = native_integer_skip(manual_skip)
    if not 1 <= skip <= depth:
        raise ValueError(f'Native CLIP Skip {skip} outside 1..{depth}')

    if mode == 'sd1':
        # HF CLIP hidden_states has depth+1 entries, index depth is final.
        effective_skip = 1 if source == 'model' else skip
        return LayerReference(depth - effective_skip + 1, effective_skip,
                              effective_skip == 1, f'SD1 Clip Skip {effective_skip}')

    if apply:
        # HF hidden_states [-1] == raw output of final transformer layer.
        idx = depth - skip + 1
        return LayerReference(idx, 1, False,
                              f'SDXL A1111 override: hidden_states[-{skip}]')

    if getattr(wrapped, 'layer', None) == 'last':
        return LayerReference(depth, 1, True, 'SDXL model last hidden state')
    if getattr(wrapped, 'layer', None) != 'hidden':
        raise ValueError(f'Unsupported SDXL CLIP-L layer: {getattr(wrapped, "layer", None)!r}')
    idx = int(getattr(wrapped, 'layer_idx', -2))
    if idx < 0:
        idx += depth + 1
    if not 0 <= idx <= depth:
        raise ValueError(f'Unsupported SDXL hidden-state index {idx}')
    return LayerReference(idx, 1, False, f'SDXL model hidden_states[{idx}]')
