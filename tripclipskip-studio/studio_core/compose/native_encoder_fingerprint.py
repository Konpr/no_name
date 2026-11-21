"""WP-03B1a: bounded, read-only metadata for *observed* A1111 encoders.

This is a capability DESCRIPTOR, not a model/weights/tokenizer fingerprint or
trusted WP-03B1 positive/negative provenance. It never calls forward(), reads
input token values, serializes prompts or retains tensors. Hook counts are
observed *before* the temporary B0 observer hooks are installed.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re

import torch

_ALLOWED_CLASS_NAME = re.compile(r'^[A-Za-z0-9_.]+$')
_ALLOWED_LAYER = frozenset(('last', 'penultimate', 'hidden'))


def _class_name(value):
    if value is None:
        return None
    cls = type(value)
    name = str(getattr(cls, '__module__', '')) + '.' + str(getattr(cls, '__qualname__', ''))
    # Do not pass through arbitrary user-defined repr or a dynamic class name.
    return name if len(name) <= 180 and _ALLOWED_CLASS_NAME.fullmatch(name) else 'UNRECOGNIZED_CLASS'


def _hook_counts(module):
    """Read registration counts only; never invoke or inspect hook callbacks."""
    if not isinstance(module, torch.nn.Module):
        return None
    out = {}
    for name, attr in (('forward_pre', '_forward_pre_hooks'),
                       ('forward', '_forward_hooks'),
                       ('backward', '_backward_hooks')):
        hooks = getattr(module, attr, None)
        out[name] = len(hooks) if isinstance(hooks, dict) else None
    return out


def _norm_description(module):
    if not isinstance(module, torch.nn.Module):
        return {'status': 'NO_MODULE_OBSERVED', 'class': None, 'hooks': None,
                'weight_dtype': None}
    weight = getattr(module, 'weight', None)
    return {'status': 'MODULE_PRESENT_EXECUTION_UNVERIFIED',
            'class': _class_name(module),
            'hooks': _hook_counts(module),
            'weight_dtype': str(weight.dtype) if isinstance(weight, torch.Tensor) else None}


def _safe_layer_configuration(wrapped):
    """Use fixed allowlists; never record a free-form custom config string."""
    layer = getattr(wrapped, 'layer', None)
    if layer is None:
        declared_layer = None
    elif type(layer) is str and layer in _ALLOWED_LAYER:
        declared_layer = layer
    else:
        declared_layer = 'UNRECOGNIZED_VALUE'
    layer_idx = getattr(wrapped, 'layer_idx', None)
    if type(layer_idx) is not int or not (-128 <= layer_idx <= 128):
        layer_idx = None
    legacy = getattr(wrapped, 'legacy', None)
    return {'layer': declared_layer,
            'layer_idx': layer_idx,
            'legacy': legacy if type(legacy) is bool else None,
            'evidence': 'DECLARED_FIELDS_ONLY_NOT_EXECUTED_SELECTION'}


def describe_encoder_capabilities(wrapper, encoder, targets, discovery_status):
    """Produce bounded JSON-safe structural evidence without running a model.

    The metadata digest describes *these fields only*, not weights, LoRA,
    model identity, token IDs or what branch A1111 actually encoded.
    """
    targets = tuple(targets)
    wrapped = getattr(wrapper, 'wrapped', None)
    if encoder == 'l':
        transformer = getattr(wrapped, 'transformer', None)
        model = getattr(transformer, 'text_model', None)
        final_norm = getattr(model, 'final_layer_norm', None)
    elif encoder == 'g':
        model = getattr(wrapped, 'model', None)
        transformer = getattr(model, 'transformer', None)
        final_norm = getattr(model, 'ln_final', None)
    else:
        model = transformer = final_norm = None
    block_targets = [t for t in targets if t.stage == 'block_output']
    # Layers are already bounded by discover_layer_targets() to 64.
    preexisting_hooks = {'forward_pre': 0, 'forward': 0, 'backward': 0}
    hook_count_known = True
    for t in targets:
        counts = _hook_counts(t.module)
        for kind in preexisting_hooks:
            if counts is None or counts[kind] is None:
                hook_count_known = False
            else:
                preexisting_hooks[kind] += counts[kind]
    if not hook_count_known:
        preexisting_hooks = None
    descriptor = {
        'schema': 'tcs-encoder-capability-descriptor-v1',
        'status': 'STRUCTURE_OBSERVED_NOT_TRUSTED_PROVENANCE',
        'encoder': encoder if encoder in ('l', 'g') else 'unknown',
        'wrapper_class': _class_name(wrapper),
        'wrapped_class': _class_name(wrapped),
        'transformer_class': _class_name(transformer),
        'model_class': _class_name(model),
        'discovery': discovery_status,
        'block_count_discovered': len(block_targets),
        'block_coordinate': block_targets[0].coordinate if block_targets else None,
        'block_indices_zero_based': [t.index for t in block_targets],
        'configuration': _safe_layer_configuration(wrapped),
        'final_norm': _norm_description(final_norm),
        'preexisting_layer_and_norm_hooks': preexisting_hooks,
        'preexisting_hooks_ownership': 'UNKNOWN',
        'token_axis_contract': 'NOT_VERIFIED_BY_STATIC_INSPECTION',
        'pooled_origin': 'NOT_VERIFIED_BY_STATIC_INSPECTION',
        'trusted_branch': 'UNKNOWN',
        'trusted_token_layout': 'UNKNOWN',
        'model_weights_fingerprint': 'NOT_COMPUTED',
    }
    stable = json.dumps(descriptor, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    descriptor['metadata_sha256'] = sha256(stable.encode('utf-8')).hexdigest()
    return descriptor


def observe_encoder_output_contract(result, token_arg):
    """Describe actual output shape / attached pooled and tentative axis relation.

    Input token values are never read; matching dimensions is not proof of
    a complete [B,T,C] layout, chunk identity, or pooling algorithm.
    """
    root = result if isinstance(result, torch.Tensor) else None
    pooled = getattr(root, 'pooled', None) if root is not None else None
    pooled = pooled if isinstance(pooled, torch.Tensor) else None
    tokens = token_arg if isinstance(token_arg, torch.Tensor) else None
    relation = 'UNKNOWN'
    if tokens is not None and tokens.ndim == 2 and root is not None and root.ndim == 3:
        b, t = tokens.shape
        matches_btc = root.shape[0] == b and root.shape[1] == t
        matches_tbc = root.shape[0] == t and root.shape[1] == b
        if matches_btc and matches_tbc:
            relation = 'AMBIGUOUS_EQUAL_AXES'
        elif matches_btc:
            relation = 'B_T_C_SHAPE_MATCH_ONLY'
        elif matches_tbc:
            relation = 'T_B_C_SHAPE_MATCH_ONLY'
    return {
        'evidence': 'SAME_FORWARD_SHAPES_ONLY_NOT_TRUSTED_LAYOUT',
        'token_input_shape': list(tokens.shape) if tokens is not None else None,
        'token_input_dtype': str(tokens.dtype) if tokens is not None else None,
        'native_root_shape': list(root.shape) if root is not None else None,
        'native_root_dtype': str(root.dtype) if root is not None else None,
        'root_axis_relation_to_input': relation,
        'pooled_attached_to_root': pooled is not None,
        'pooled_shape': list(pooled.shape) if pooled is not None else None,
        'pooled_dtype': str(pooled.dtype) if pooled is not None else None,
        'pooled_selection_and_projection': 'NOT_VERIFIED',
        'token_ids_or_text': 'NOT_RECORDED',
    }
