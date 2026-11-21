"""WP-03B0.2: interpret bounded *metadata*, never tensors.

Byte equality with a wrapper output is evidence only of byte equality for
that forward. It does NOT prove the layer was selected by A1111, that raw and
normalized representations have equivalent semantics, or GPU image identity.
"""
from __future__ import annotations


def summarize_layer_metadata(encoder: str, layers: list, wrapper_outputs: list):
    """Return JSON-only summary and numeric warnings from an observed call.

    This read-only function never hashes new values or holds PyTorch tensors.
    In shapes mode it cannot establish finiteness or digest equality.
    """
    native = next((x for x in wrapper_outputs if x.get('path') == 'root'), None)
    native = native if isinstance(native, dict) else None
    native_digest = native.get('digest_status', 'NOT_OBSERVED') if native else 'NOT_OBSERVED'
    nonfinite_layers = []
    matches = []
    for entry in layers:
        tensor = entry.get('tensor', {})
        if tensor.get('digest_status') == 'NONFINITE' or tensor.get('finite') is False:
            nonfinite_layers.append({
                'layer_index': entry.get('layer_index'),
                'stage': entry.get('stage'),
                'coordinate': entry.get('coordinate'),
            })
        if (native is not None and native_digest == 'COMPLETE'
                and tensor.get('digest_status') == 'COMPLETE'
                and native.get('sha256') and native.get('sha256') == tensor.get('sha256')
                and tensor.get('dtype') == native.get('dtype')
                and tensor.get('element_count') == native.get('element_count')):
            matches.append({
                'layer_index': entry.get('layer_index'),
                'stage': entry.get('stage'),
                'coordinate': entry.get('coordinate'),
                'comparison': 'EXACT_SHA256_SAME_SHAPE' if tensor.get('shape') == native.get('shape')
                              else 'EXACT_FLAT_BYTES_DIFFERENT_SHAPE',
            })
    nonfinite_native = [
        {'path': x.get('path'), 'shape': x.get('shape')}
        for x in wrapper_outputs
        if x.get('digest_status') == 'NONFINITE'
    ]
    comparison = ('NOT_COMPARABLE' if native_digest != 'COMPLETE'
                  else 'MATCHED' if matches else 'NO_MATCHING_EXECUTED_OUTPUT')
    if nonfinite_native:
        root_interpretation = 'NATIVE_OUTPUT_NONFINITE_OBSERVED'
    elif nonfinite_layers and native_digest == 'COMPLETE':
        root_interpretation = 'NONFINITE_INTERMEDIATE_WITH_FINITE_WRAPPER_ROOT'
    elif nonfinite_layers:
        root_interpretation = 'NONFINITE_INTERMEDIATE_WRAPPER_FINITE_UNKNOWN'
    else:
        root_interpretation = 'NO_NONFINITE_OBSERVED' if native_digest == 'COMPLETE' else 'NOT_ASSESSED'

    alerts = []
    if nonfinite_layers:
        # Grouped by location; hard-bounded by the discovered module count.
        for entry in nonfinite_layers:
            idx = entry['layer_index']
            kind = 'FINAL_NORM' if entry['stage'] == 'final_norm_if_executed' else 'BLOCK'
            alerts.append(f'NONFINITE_OBSERVED_{kind}:{encoder}:{idx}')
    for entry in nonfinite_native:
        alerts.append(f'NONFINITE_WRAPPER_OUTPUT:{encoder}:{entry["path"]}')
    if nonfinite_layers and native_digest == 'COMPLETE':
        alerts.append(f'NONFINITE_INTERMEDIATE_BUT_WRAPPER_ROOT_FINITE:{encoder}')

    return ({
        'evidence': 'SAME_FORWARD_METADATA_ONLY_NOT_TRUSTED_PROVENANCE',
        'wrapper_root_digest_status': native_digest,
        'wrapper_output_byte_comparison': comparison,
        'matching_executed_outputs': matches,
        'nonfinite_executed_outputs': nonfinite_layers,
        'nonfinite_wrapper_outputs': nonfinite_native,
        'interpretation': root_interpretation,
    }, alerts)


def assess_observed_layer_rows(rows):
    """Pure, read-only fail-closed numeric source assessment for Layer Explorer.

    This does not select or inject a state. A valid digest and finite statistic
    do NOT establish trusted token/chunk/branch provenance. In shapes mode
    numeric safety is unverified, never inferred from an expected layer index.
    """
    assessed = []
    for entry in rows:
        entry_copy = dict(entry)
        tensor = entry.get('tensor', {})
        status = tensor.get('digest_status')
        if status == 'NONFINITE' or tensor.get('finite') is False:
            numeric_gate = 'REJECT_NONFINITE'
        elif status == 'COMPLETE' and tensor.get('finite') is True:
            numeric_gate = 'FINITE_OBSERVED_NOT_TRUSTED'
        else:
            numeric_gate = 'UNVERIFIED'
        entry_copy['source_guard'] = {
            'numeric_gate': numeric_gate,
            'injection_eligible': False,
            'reason': 'TRUSTED_TOKEN_AND_LAYER_PROVENANCE_NOT_ESTABLISHED',
        }
        assessed.append(entry_copy)
    return assessed
