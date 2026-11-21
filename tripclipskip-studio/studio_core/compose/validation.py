"""Tensor shape/dtype/provenance guards for future Composer operations."""
from __future__ import annotations

import torch
from .errors import ComposeError, PooledLayoutUnknown
from .types import ConditioningChannels
from .capabilities import ModelCapabilities

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_CHANNELS = ('l', 'g', 'pooled_g')
_FINGERPRINTS = ('model_fingerprint', 'tokenizer_fingerprint', 'lora_ti_fingerprint')


def _require_tensor(value, *, name, rank, width, full_finite=True):
    if not isinstance(value, torch.Tensor):
        raise ComposeError(f'{name} requires a named torch.Tensor')
    if value.ndim != rank or value.shape[-1] != width or any(d <= 0 for d in value.shape):
        raise ComposeError(f'{name}: expected rank {rank}, width {width}, positive dimensions; got {tuple(value.shape)}')
    if value.dtype not in _FLOAT_DTYPES:
        raise ComposeError(f'{name}: expected floating tensor, got {value.dtype}')
    if full_finite and not bool(torch.isfinite(value).all()):
        raise ComposeError(f'{name}: NaN/Inf not allowed')


def _live_channels(value: ConditioningChannels, cap: ModelCapabilities):
    return ('l',) if cap.family == 'sd15' else ('l', 'g') + (('pooled_g',) if value.pooled_g is not None else ())


def _verify_source_records(value: ConditioningChannels, cap: ModelCapabilities):
    # Even a shadowed legacy source cannot silently refer to another branch.
    for field_name in ('source_ref', 'l_source', 'g_source', 'pooled_source'):
        declared = getattr(value, field_name)
        if declared is not None:
            if not hasattr(declared, 'branch') or declared.branch != value.branch:
                raise ComposeError(f'{field_name}: runtime provenance branch mismatch')
    refs = []
    for channel in _live_channels(value, cap):
        ref = value.source_for(channel)
        if ref is None:
            continue  # Native observation without editable source metadata.
        if ref.branch != value.branch:
            raise ComposeError(f'{channel}: runtime provenance branch mismatch')
        refs.append((channel, ref))
    if refs:
        first = refs[0][1]
        for channel, ref in refs[1:]:
            for name in _FINGERPRINTS:
                if getattr(first, name) != getattr(ref, name):
                    raise ComposeError(f'{channel}: inconsistent {name} across channel sources')


def validate_channels(value: ConditioningChannels, cap: ModelCapabilities, *, full_finite=True):
    """Pure validation; never normalize, cast, slice ADM, or clone tensors.

    Full finite scans synchronize CUDA, so only use during pre-sampling checks.
    This does not authorize mutation/injection or establish ADM layout.
    """
    if not isinstance(value, ConditioningChannels) or not isinstance(cap, ModelCapabilities):
        raise ComposeError('Expected typed ConditioningChannels and ModelCapabilities')
    if value.branch not in ('positive', 'negative'):
        raise ComposeError('Unknown conditioning branch')
    _verify_source_records(value, cap)
    if cap.model_fingerprint is not None:
        for channel in _live_channels(value, cap):
            ref = value.source_for(channel)
            if ref is not None and ref.model_fingerprint != cap.model_fingerprint:
                raise ComposeError(f'{channel}: source fingerprint disagrees with active checkpoint')
    _require_tensor(value.l, name='CLIP-L', rank=3, width=cap.width_l, full_finite=full_finite)
    b, t = value.l.shape[:2]
    if t % cap.chunk_length:
        raise ComposeError(f'CLIP-L token axis {t} is not a multiple of {cap.chunk_length}')
    if cap.family == 'sd15':
        if value.g is not None or value.pooled_g is not None:
            raise ComposeError('SD1.x does not have a CLIP-G/pooled-G channel')
        if value.g_source is not None or value.pooled_source is not None:
            raise ComposeError('SD1.x cannot claim G/pooled-G source provenance')
    else:
        _require_tensor(value.g, name='CLIP-G', rank=3, width=cap.width_g, full_finite=full_finite)
        if value.g.shape[:2] != (b, t):
            raise ComposeError('CLIP-L/G batch or token-axis shape mismatch')
        if value.g.device != value.l.device or value.g.dtype != value.l.dtype:
            raise ComposeError('CLIP-L/G must have matching device and dtype before concat')
        if value.pooled_g is not None:
            if not cap.named_pooled_available:
                raise PooledLayoutUnknown('Named G pooled was not captured from a verified source')
            _require_tensor(value.pooled_g, name='pooled-G', rank=2, width=cap.pooled_width, full_finite=full_finite)
            if value.pooled_g.shape[0] != b or value.pooled_g.device != value.l.device or value.pooled_g.dtype != value.l.dtype:
                raise ComposeError('pooled-G batch/device/dtype mismatch')
        elif value.pooled_source is not None:
            raise ComposeError('Pooled source supplied without pooled-G tensor')
    return value


def validate_compatible(a: ConditioningChannels, b: ConditioningChannels,
                        cap: ModelCapabilities, *, full_finite=True,
                        operation='bundle', channel=None):
    """Check candidates for a specific same-branch operation.

    ``bundle`` means a whole conditioning bundle can be selected/interchanged;
    its opaque microconditioning must have the SAME owner/reference. ``channel``
    means only one named text channel can be selected while the caller preserves
    all other target channels and native geometry unchanged. Neither mode
    implements a modification by itself.
    """
    if operation not in ('bundle', 'channel'):
        raise ComposeError('Unknown compatibility operation')
    if operation == 'channel' and channel not in _CHANNELS:
        raise ComposeError('Channel operation requires explicit l/g/pooled_g target')
    if operation == 'channel' and cap.family == 'sd15' and channel != 'l':
        raise ComposeError('SD1.x only supports CLIP-L')
    validate_channels(a, cap, full_finite=full_finite)
    validate_channels(b, cap, full_finite=full_finite)
    if a.branch != b.branch:
        raise ComposeError('Refusing positive/negative cross-branch mixing')
    relevant = _live_channels(a, cap) if operation == 'bundle' else (channel,)
    if operation == 'channel' and (getattr(a, channel) is None or getattr(b, channel) is None):
        raise ComposeError(f'{channel}: selected channel missing')
    for name in relevant:
        ra, rb = a.source_for(name), b.source_for(name)
        if ra is None or rb is None:
            raise ComposeError(f'{name}: both sources require runtime provenance')
        for key in _FINGERPRINTS:
            if getattr(ra, key) != getattr(rb, key):
                raise ComposeError(f'Incompatible source provenance: {key} ({name})')
        x, y = getattr(a, name), getattr(b, name)
        if (x is None) != (y is None):
            raise ComposeError(f'{name}: one source is missing')
        if x is not None and (x.shape != y.shape or x.device != y.device or x.dtype != y.dtype):
            raise ComposeError(f'{name}: incompatible shapes/devices/dtypes')
    if operation == 'bundle' and a.microcond is not b.microcond:
        raise ComposeError('Opaque microconditioning has different owners; whole-bundle operation refused')
    if operation == 'bundle' and (a.pooled_g is None) != (b.pooled_g is None):
        raise ComposeError('pooled_g: one bundle lacks native pooled')
    return a, b


def assemble_cross(channels: ConditioningChannels, cap: ModelCapabilities):
    """Construct token-wise context only; never touch pooled/ADM/geometry."""
    validate_channels(channels, cap)
    if cap.family == 'sd15':
        return channels.l
    return torch.cat((channels.l, channels.g), dim=-1)
