"""Offline exact-digest comparison of two opt-in native-proof JSON reports.

Never claims exact tensor equivalence when a value was not fully captured.
"""
from __future__ import annotations

import json
from pathlib import Path

from .live_proof import PROOF_SCHEMA


class InvalidProofReport(ValueError):
    pass


def _index(report):
    result = {}
    for event in report.get('events', []):
        key = (event.get('stage'), event.get('branch'), event.get('pass'), event.get('ordinal'))
        if key in result:
            raise InvalidProofReport('Duplicated event identity')
        result[key] = {item['path']: item for item in event.get('tensors', [])}
    return result


def compare_reports(off, on):
    """Return an evidence report, never just a success boolean.

    Exact match means identical captured tensor bytes for the aligned events,
    NOT verified image quality, U-Net equivalence, or absence of instrumentation.
    """
    if off.get('schema') != PROOF_SCHEMA or on.get('schema') != PROOF_SCHEMA:
        raise InvalidProofReport('Unknown proof report schema')
    if off.get('tcs') != 'OFF' or on.get('tcs') != 'ON':
        raise InvalidProofReport('Expected first report OFF and second report ON')
    if off.get('mode') != 'sha256' or on.get('mode') != 'sha256':
        raise InvalidProofReport('Both proof reports require sha256 mode')
    if off.get('model_family') != on.get('model_family'):
        raise InvalidProofReport('Different model families')
    # Comparing incompatible sampler/seed/model settings would be misleading.
    settings_off, settings_on = off.get('settings', {}), on.get('settings', {})
    required = ('checkpoint', 'seed', 'steps', 'sampler', 'width', 'height', 'cfg_scale')
    mismatched = [name for name in required if settings_off.get(name) is None or
                  settings_on.get(name) is None or settings_off.get(name) == '' or
                  settings_on.get(name) == '' or settings_off[name] != settings_on[name]]
    if mismatched:
        return {'status': 'INVALID_BASELINE', 'different_settings': mismatched,
                'evidence': [], 'summary': {'exact': 0, 'different': 0, 'incomplete': 0}}

    a, b = _index(off), _index(on)
    rows = []
    exact = different = incomplete = 0
    for event in sorted(a.keys() | b.keys(), key=lambda x: tuple(map(str, x))):
        x, y = a.get(event, {}), b.get(event, {})
        for path in sorted(x.keys() | y.keys()):
            rx, ry = x.get(path), y.get(path)
            key = {'stage': event[0], 'branch': event[1], 'pass': event[2],
                   'ordinal': event[3], 'path': path}
            if rx is None or ry is None:
                state = 'MISSING'
            elif (rx['shape'] != ry['shape'] or rx['dtype'] != ry['dtype'] or
                  rx['device_type'] != ry['device_type']):
                state = 'METADATA_MISMATCH'
            elif rx.get('digest_status') != 'COMPLETE' or ry.get('digest_status') != 'COMPLETE':
                state = 'DIGEST_UNAVAILABLE'
            else:
                state = 'EXACT' if rx['sha256'] == ry['sha256'] else 'DIFFERENT'
            if state == 'EXACT':
                exact += 1
            elif state in ('DIFFERENT', 'METADATA_MISMATCH'):
                different += 1
            else:
                incomplete += 1
            rows.append({**key, 'status': state})
    # Denoiser-forward calls are counted even after the bounded event capture
    # stops. A missing or different count must never pass as exact evidence.
    calls_off = off.get('forward_calls_seen')
    calls_on = on.get('forward_calls_seen')
    forward_counts = {'off': calls_off, 'on': calls_on}
    valid_counts = all(type(count) is int and count > 0 for count in (calls_off, calls_on))
    if not valid_counts:
        incomplete += 1
        rows.append({'stage': 'denoiser_forward_count', 'branch': 'both',
                     'status': 'COUNT_UNAVAILABLE'})
    elif calls_off != calls_on:
        different += 1
        rows.append({'stage': 'denoiser_forward_count', 'branch': 'both',
                     'status': 'COUNT_MISMATCH'})
    else:
        rows.append({'stage': 'denoiser_forward_count', 'branch': 'both',
                     'status': 'COUNT_EXACT'})

    # An experimental phase/absolute run needs a full forward-input trace.
    # For ordinary G0 native equivalence, a bounded sample is still useful;
    # do not confuse it with a complete denoiser-trajectory proof.
    phase_enabled = any(bool(report.get('settings', {}).get('multi_count', 0)) or
                        bool(report.get('settings', {}).get('absolute_enabled', False))
                        for report in (off, on))
    if phase_enabled:
        for side, report in (('off', off), ('on', on)):
            captured = len([event for event in report.get('events', [])
                            if event.get('stage') == 'denoiser_input' and
                            event.get('branch') == 'positive'])
            seen = report.get('forward_calls_seen')
            if type(seen) is not int or captured < seen:
                incomplete += 1
                rows.append({'stage': 'denoiser_forward_trace', 'branch': side,
                             'status': 'FORWARD_TRACE_TRUNCATED',
                             'captured': captured})

    # A report with unhandled capture errors is never proof of equality.
    for key, report in (('off', off), ('on', on)):
        if report.get('warnings') or report.get('status') != 'CAPTURED_NOT_GPU_ACCEPTED':
            incomplete += 1
            rows.append({'stage': 'report', 'branch': key, 'status': 'CAPTURE_INCOMPLETE'})
    required_stages = {('prepared', 'positive'), ('prepared', 'negative'),
                       ('denoiser_input', 'positive'), ('denoiser_input', 'negative'),
                       ('encoder_l', 'unknown')}
    if off['model_family'] == 'sdxl':
        required_stages.add(('encoder_g', 'unknown'))
    observed = {(item.get('stage'), item.get('branch')) for item in rows}
    missing = sorted(required_stages - observed)
    if missing:
        incomplete += len(missing)
    status = 'DIFFERENT' if different else ('INCOMPLETE' if incomplete or not exact else 'EXACT_CAPTURED_VALUES')
    return {'status': status, 'forward_calls': forward_counts, 'summary': {'exact': exact, 'different': different,
                                        'incomplete': incomplete},
            'missing_required_stages': missing, 'evidence': rows,
            'caveat': 'EXACT_CAPTURED_VALUES is not GPU/native pipeline certification.'}


def compare_files(off_path, on_path):
    with Path(off_path).open(encoding='utf-8') as stream:
        off = json.load(stream)
    with Path(on_path).open(encoding='utf-8') as stream:
        on = json.load(stream)
    return compare_reports(off, on)
