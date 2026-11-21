"""WP-03A: pure, fail-closed mixing of two *captured* CLIP hidden states.

No CLIP forward, monkey-patches, Gradio inputs, or integration with denoising.
The caller must capture native hidden states and establish their provenance.

Endpoints intentionally return the selected original tensor object: a borrowed
reference, not a mutable snapshot. Treat returned and input tensors as read-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .capabilities import ModelCapabilities
from .errors import ComposeError
from .types import ConditioningChannels
from .validation import validate_channels

_SUPPORTED_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))
_VALID_REPRESENTATIONS = frozenset(('raw', 'native_normalized'))
_VALID_POLICIES = frozenset(('input_as_is', 'output_rms_match'))
_WIDTHS = {'l': 768, 'g': 1280}


@dataclass(frozen=True)
class LayerSample:
    """A borrowed [batch, tokens, features] tensor plus *asserted* capture metadata.

    `native_normalized` is an assertion by the capture owner, NOT proof that
    native A1111 final LayerNorm was correctly performed. WP-03A never calls LN.
    `source_id` distinguishes a prompt/capture source; cross-prompt composition
    belongs in a separate, explicitly validated future operation.
    `token_layout_digest` and `capture_invocation_id` are ASSERTED tags, not
    trusted proofs. Strict mode rejects their absence, but only B0/B1 can bind
    them to an actual forward (token IDs/order/chunks/edit timeline).
    """
    tensor: torch.Tensor
    encoder: str
    layer_index: int
    source_id: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    lora_ti_fingerprint: str
    branch: str = 'positive'
    representation: str = 'raw'
    # Optional ASSERTED metadata; only WP-03B0/B1 can establish real provenance.
    # Token-layout digest must account for actual chunks, order, special/TI tokens,
    # weights and any prompt-editing timeline used for the capture.
    token_layout_digest: str | None = None
    capture_invocation_id: str | None = None


def _scalar(value: float, *, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise ComposeError(f'{name} must be a finite number, not bool')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ComposeError(f'{name} must be a finite number') from exc
    if not math.isfinite(result) or not (lower <= result <= upper):
        raise ComposeError(f'{name} must be finite and within [{lower}, {upper}]')
    return result


def _validate_sample(sample: LayerSample, *, capabilities: ModelCapabilities | None) -> None:
    if not isinstance(sample, LayerSample):
        raise ComposeError('Expected LayerSample')
    if sample.encoder not in ('l', 'g'):
        raise ComposeError('Only CLIP-L and CLIP-G token channels are supported')
    if isinstance(sample.layer_index, bool) or not isinstance(sample.layer_index, int) or sample.layer_index < 0:
        raise ComposeError('layer_index must be a nonnegative integer (hidden_states index)')
    if not isinstance(sample.source_id, str) or not sample.source_id.strip():
        raise ComposeError('source_id must identify a captured prompt/source')
    if not isinstance(sample.model_fingerprint, str) or not sample.model_fingerprint.strip():
        raise ComposeError('model_fingerprint is required')
    for name in ('tokenizer_fingerprint', 'lora_ti_fingerprint'):
        v = getattr(sample, name)
        if not isinstance(v, str) or not v.strip():
            raise ComposeError(f'{name} is required for a valid CLIP capture')
    if sample.branch not in ('positive', 'negative'):
        raise ComposeError('Invalid positive/negative branch')
    if sample.representation not in _VALID_REPRESENTATIONS:
        raise ComposeError('Unknown hidden-state representation; no implicit LayerNorm')
    for name in ('token_layout_digest', 'capture_invocation_id'):
        value = getattr(sample, name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ComposeError(f'{name} must be a nonempty string or None')
    t = sample.tensor
    if not isinstance(t, torch.Tensor):
        raise ComposeError('Captured state must be a torch.Tensor')
    if t.layout != torch.strided or t.is_meta:
        raise ComposeError('Only dense, materialized tensors are supported')
    if t.ndim != 3 or any(dim <= 0 for dim in t.shape):
        raise ComposeError('Expected nonempty [B,T,C] tensor')
    if t.shape[1] % 77:
        raise ComposeError('Token axis must be a multiple of A1111 77-position chunk length')
    if t.shape[-1] != _WIDTHS[sample.encoder]:
        raise ComposeError(f'CLIP-{sample.encoder.upper()} requires width {_WIDTHS[sample.encoder]}, got {t.shape[-1]}')
    if t.dtype not in _SUPPORTED_DTYPES:
        raise ComposeError('Only floating point CLIP states are supported')
    if capabilities is not None:
        if not isinstance(capabilities, ModelCapabilities):
            raise ComposeError('Expected ModelCapabilities for checked model constraints')
        if sample.encoder == 'g' and capabilities.family != 'sdxl':
            raise ComposeError('CLIP-G is not available in SD1.x')
        if sample.encoder == 'l' and t.shape[-1] != capabilities.width_l:
            raise ComposeError('CLIP-L width does not match capabilities')
        if sample.encoder == 'g' and t.shape[-1] != capabilities.width_g:
            raise ComposeError('CLIP-G width does not match capabilities')
        if t.shape[1] % capabilities.chunk_length:
            raise ComposeError('Token length must be a multiple of the native chunk length')
        if capabilities.model_fingerprint is not None and capabilities.model_fingerprint != sample.model_fingerprint:
            raise ComposeError('Layer source does not match the active checkpoint')
    try:
        if not bool(torch.isfinite(t).all()):
            raise ComposeError('NaN/Inf hidden state is not permitted')
    except (RuntimeError, NotImplementedError) as exc:
        raise ComposeError('Unable to validate finite hidden-state values') from exc


def _check_pair(
    a: LayerSample,
    b: LayerSample,
    *,
    capabilities: ModelCapabilities | None,
    require_capture_metadata: bool = False,
    allow_cross_representation: bool = False,
) -> None:
    """Check asserted source alignment; never certify origin based on strings.

    Legacy CPU callers without capture metadata remain usable. Runtime-facing
    observers MUST enable require_capture_metadata before treating positions
    as aligned. Missing metadata are not proof of same tokens/invocation.
    """
    if not isinstance(require_capture_metadata, bool) or not isinstance(allow_cross_representation, bool):
        raise ComposeError('Capture metadata and representation policies must be boolean')
    _validate_sample(a, capabilities=capabilities)
    _validate_sample(b, capabilities=capabilities)
    fields = ('encoder', 'source_id', 'model_fingerprint', 'tokenizer_fingerprint',
              'lora_ti_fingerprint', 'branch', 'token_layout_digest', 'capture_invocation_id')
    if not allow_cross_representation:
        fields += ('representation',)
    for field in fields:
        if getattr(a, field) != getattr(b, field):
            raise ComposeError(f'Layer pair requires equal {field} (unverified source/position alignment)')
    if require_capture_metadata and (not a.token_layout_digest or not a.capture_invocation_id):
        raise ComposeError('Strict capture alignment requires token_layout_digest and capture_invocation_id')
    if a.tensor.shape != b.tensor.shape:
        raise ComposeError('Hidden-state shapes differ')
    if a.tensor.device != b.tensor.device:
        raise ComposeError('Hidden-state devices differ')
    if a.tensor.dtype != b.tensor.dtype:
        raise ComposeError('Hidden-state dtypes differ')


def validate_observed_layer_index(sample: LayerSample, available_indices: tuple[int, ...]) -> None:
    """Check layer index against *caller-observed* physical indices.

    The pure library cannot observe physical CLIP depth. This is only a
    constraint helper for WP-03B0; passing invented indices is NOT attestation.
    """
    _validate_sample(sample, capabilities=None)
    if not isinstance(available_indices, tuple) or not available_indices or any(
        not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in available_indices
    ) or len(set(available_indices)) != len(available_indices):
        raise ComposeError('available_indices must be a nonempty unique tuple of nonnegative observed indices')
    if sample.layer_index not in available_indices:
        raise ComposeError('Layer index is not in the caller-observed depth/coordinate system')


def snapshot_layer_sample(sample: LayerSample) -> LayerSample:
    """Take an owned detached copy, never a native-capture certificate.

    Use only for bounded CPU diagnostics; copying full live GPU states may
    increase VRAM and should not be introduced into inference by default.
    """
    _validate_sample(sample, capabilities=None)
    with torch.no_grad():
        return replace(sample, tensor=sample.tensor.detach().clone())


def _checked_reference(reference: torch.Tensor | None, sample: LayerSample) -> torch.Tensor:
    if not isinstance(reference, torch.Tensor):
        raise ComposeError('output_rms_match requires an explicit native reference tensor')
    if reference.layout != torch.strided or reference.is_meta or reference.shape != sample.tensor.shape:
        raise ComposeError('RMS reference shape/layout is incompatible')
    if reference.device != sample.tensor.device or reference.dtype != sample.tensor.dtype:
        raise ComposeError('RMS reference device/dtype is incompatible')
    if not bool(torch.isfinite(reference).all()):
        raise ComposeError('RMS reference contains NaN/Inf')
    return reference


def _rms_match(result_working: torch.Tensor, reference: torch.Tensor, gain_limit: float) -> torch.Tensor:
    """Per-token feature RMS matching, never silently clipping the gain.

    Both-zero tokens remain zero. One-sided zero tokens cannot be matched
    without creating/removing features, and are rejected instead of guessed.
    """
    # For FP64 input, the working dtype is FP64 throughout the operation.
    # Intermediate overflow is rejected rather than silently renormalized.
    target = reference.to(dtype=result_working.dtype)
    input_rms = torch.sqrt(torch.mean(result_working.square(), dim=-1, keepdim=True))
    target_rms = torch.sqrt(torch.mean(target.square(), dim=-1, keepdim=True))
    if not bool(torch.isfinite(input_rms).all() and torch.isfinite(target_rms).all()):
        raise ComposeError('RMS intermediate is not finite')
    source_zero = input_rms == 0
    target_zero = target_rms == 0
    if bool(torch.logical_xor(source_zero, target_zero).any()):
        raise ComposeError('Cannot RMS-match a zero vector to a nonzero vector')
    # For zeros on both sides the gain is 1: the zero tensor stays zero.
    gain = torch.where(source_zero, torch.ones_like(input_rms), target_rms / input_rms.clamp_min(torch.finfo(result_working.dtype).tiny))
    if not bool(torch.isfinite(gain).all()):
        raise ComposeError('Nonfinite RMS gain')
    if bool(((gain > gain_limit) | (gain < (1.0 / gain_limit))).any()):
        raise ComposeError('RMS gain exceeds the explicit safety limit')
    return result_working * gain


def mix_two_layers(
    a: LayerSample,
    b: LayerSample,
    alpha: float,
    *,
    policy: str = 'input_as_is',
    native_reference: torch.Tensor | None = None,
    rms_gain_limit: float = 4.0,
    capabilities: ModelCapabilities | None = None,
    require_capture_metadata: bool = False,
) -> torch.Tensor:
    """Mix 2 non-necessarily-adjacent layers of the SAME CLIP encoder/source.

    `input_as_is`: raw or already-native-normalized states, without applying LN.
    `output_rms_match`: explicit experimental per-token RMS to native_reference.
    FP16/BF16/FP32 use a FP32 intermediate, FP64 uses FP64 throughout.

    Endpoints are exact object identity with a/b and skip RMS, BUT all supplied
    inputs and reference metadata must still pass validation first.
    require_capture_metadata=True enforces matched nonempty layout/invocation
    tags, NOT runtime attestation; the legacy default remains False for CPU.
    Raw vs native_normalized MIXING is always forbidden in this API.
    """
    value = _scalar(alpha, name='alpha', lower=0.0, upper=1.0)
    if policy not in _VALID_POLICIES:
        raise ComposeError('Unsupported layer normalization policy')
    _check_pair(a, b, capabilities=capabilities, require_capture_metadata=require_capture_metadata)
    if policy == 'output_rms_match':
        _scalar(rms_gain_limit, name='rms_gain_limit', lower=1.0, upper=32.0)
        _checked_reference(native_reference, a)
    elif native_reference is not None:
        raise ComposeError('native_reference can be supplied only for output_rms_match')
    if value == 0.0:
        return a.tensor
    if value == 1.0:
        return b.tensor

    # WP-03A is a read-only INFERENCE utility. Do not retain TE autograd graphs.
    with torch.no_grad():
        working_dtype = torch.float64 if a.tensor.dtype == torch.float64 else torch.float32
        out = torch.lerp(a.tensor.to(dtype=working_dtype), b.tensor.to(dtype=working_dtype), value)
        if not bool(torch.isfinite(out).all()):
            raise ComposeError('Mixed hidden state contains NaN/Inf in working dtype')
        if policy == 'output_rms_match':
            out = _rms_match(out, native_reference, float(rms_gain_limit))
        out = out.to(dtype=a.tensor.dtype)
        if not bool(torch.isfinite(out).all()):
            raise ComposeError('Mixed hidden state overflowed in the output dtype')
        return out


def preview_bundle_layer_mix(
    bundle: ConditioningChannels,
    a: LayerSample,
    b: LayerSample,
    alpha: float,
    *,
    capabilities: ModelCapabilities,
    policy: str = 'input_as_is',
    native_reference: torch.Tensor | None = None,
    rms_gain_limit: float = 4.0,
    require_capture_metadata: bool = False,
) -> ConditioningChannels:
    """Pure model-aware preview: replace one token channel without mutating rest.

    Does NOT assemble/inject an A1111 conditioner. Pooled-G, microconditioning,
    native references and all other borrowed channels keep object identity.
    The selected capture source must match the channel's declared source.
    """
    validate_channels(bundle, capabilities)
    if a.encoder != b.encoder:
        raise ComposeError('Cannot select two different token channels')
    if bundle.branch != a.branch:
        raise ComposeError('Bundle/capture branch mismatch')
    declared = bundle.source_for(a.encoder)
    if declared is None or declared.id != a.source_id:
        raise ComposeError('Unknown or mismatched token-channel provenance in bundle')
    for field in ('model_fingerprint', 'tokenizer_fingerprint', 'lora_ti_fingerprint'):
        if getattr(declared, field) != getattr(a, field):
            raise ComposeError(f'Bundle/{field} fingerprint mismatch')
    out = mix_two_layers(a, b, alpha, policy=policy, native_reference=native_reference,
                         rms_gain_limit=rms_gain_limit, capabilities=capabilities,
                         require_capture_metadata=require_capture_metadata)
    updated = replace(bundle, **{a.encoder: out})
    validate_channels(updated, capabilities)
    return updated
