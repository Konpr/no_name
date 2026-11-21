"""Versioned, bounded TCS configuration for A1111 PNG Info -> UI restore.

No prompt text, tensor, model weights or execution payload is serialized.
The readable TCS fields remain separately in PNG Info.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import re

KEY = 'TCS State'
PREFIX = 'v1.'
MAX_TOKEN_CHARS = 8192
MAX_CONFIG_BYTES = 6144

# Same order as the 28 Script.ui generation controls. Never reorder existing keys.
FIELD_ORDER = (
    'enabled', 'l', 'interpolation', 'separate_g', 'g', 'dual', 'late_l', 'switch',
    'reset_cache', 'late_g', 'reference_source', 'native_skip', 'apply_sdxl_clip_l',
    'multi_count', 'third_l', 'third_g', 'fourth_l', 'fourth_g', 'start2', 'start3',
    'start4', 'phase2_l', 'phase2_g', 'progress_mode', 'passive_trace',
    'absolute_enabled', 'absolute_text', 'conditioning_inspect',
)
DEFAULTS = dict(zip(FIELD_ORDER, (
    False, 1., 'lerp', True, 1., False, 2., .5, True, 2., 'a1111', 1,
    False, 0, 2.5, 2.5, 3., 3., .25, .5, .75, 2., 2., 'denoiser',
    False, False, '0.00 | -2 | 31\n0.70 | -8 | 24', 'Выключено',
)))
BOOL_FIELDS = {'enabled','separate_g','dual','reset_cache','apply_sdxl_clip_l',
               'passive_trace','absolute_enabled'}
L_FIELDS = {'l','late_l','third_l','fourth_l','phase2_l'}
G_FIELDS = {'g','late_g','third_g','fourth_g','phase2_g'}
FRACTIONS = {'switch','start2','start3','start4'}
CHOICES = {
    'interpolation': {'lerp','slerp'},
    'reference_source': {'a1111','model','custom'},
    'progress_mode': {'denoiser','sampler'},
    'conditioning_inspect': {'Выключено','Только формы','Нормы и сравнение фаз'},
}


def _validated(key, value):
    if key in BOOL_FIELDS:
        if type(value) is not bool:
            raise ValueError(f'{key}: expected boolean')
        return value
    if key == 'multi_count':
        if type(value) is not int or value not in (0,3,4):
            raise ValueError('invalid multi_count')
        return value
    if key in L_FIELDS | G_FIELDS | FRACTIONS | {'native_skip'}:
        if type(value) not in (int,float):
            raise ValueError(f'{key}: invalid number')
        x = float(value)
        if not math.isfinite(x):
            raise ValueError(f'{key}: non-finite')
        lo, hi = ((1.,12.) if key in L_FIELDS or key == 'native_skip'
                  else (1.,31.) if key in G_FIELDS else (0.,1.))
        if not lo <= x <= hi:
            raise ValueError(f'{key}: out of range')
        if key == 'native_skip' and x.is_integer():
            return int(x)
        return x
    if key in CHOICES:
        if type(value) is not str or value not in CHOICES[key]:
            raise ValueError(f'{key}: unsupported choice')
        return value
    if key == 'absolute_text':
        if type(value) is not str or len(value) > 2048 or '\x00' in value:
            raise ValueError('invalid absolute_text')
        return value
    raise ValueError(f'unknown TCS field: {key}')


def validate(data, *, require_all=True):
    if not isinstance(data, dict):
        raise ValueError('settings must be an object')
    keys = FIELD_ORDER if require_all else tuple(x for x in FIELD_ORDER if x in data)
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f'incomplete TCS State: missing {missing[0]}')
    return {k: _validated(k, data[k]) for k in keys}


def encode(settings):
    values = validate(settings)
    raw = json.dumps(values, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError('settings too large for PNG Info')
    result = PREFIX + base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')
    if len(result) > MAX_TOKEN_CHARS:
        raise ValueError('settings token exceeds limit')
    return result


def decode(token):
    if not isinstance(token, str) or len(token) > MAX_TOKEN_CHARS or not token.startswith(PREFIX):
        raise ValueError('unsupported or oversized TCS State')
    encoded = token[len(PREFIX):]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', encoded):
        raise ValueError('invalid base64 alphabet')
    try:
        raw = base64.b64decode(encoded + '=' * (-len(encoded) % 4), altchars=b'-_', validate=True)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ValueError('oversized TCS State')
        data = json.loads(raw.decode('utf8'))
        return validate(data)
    except (binascii.Error, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('corrupt TCS State') from exc


def from_infotext(params):
    """Return None on missing or untrusted/incompatible PNG metadata.

    Legacy PNGs have no complete, unambiguous configuration. We deliberately
    avoid silently guessing how their hidden phase sliders were set.
    """
    token = params.get(KEY) if isinstance(params, dict) else None
    if not token:
        return None
    try:
        return decode(token)
    except (ValueError, KeyError):
        return None


def legacy_from_infotext(params):
    """Restore only fields explicitly present in pre-v1 TCS PNGs.

    Older images did not encode the 28-control state, so hidden switches,
    independent G and absolute text cannot be reconstructed reliably.
    """
    if not isinstance(params, dict) or not params.get('TripClipSkip Studio'):
        return {}
    result = {}
    for label, key in (('TCS CLIP-L','l'), ('TCS CLIP-G','g'),
                       ('TCS Interpolation','interpolation'),
                       ('TCS Phase progress source','progress_mode')):
        if label in params:
            try:
                val = params[label]
                if key in ('l','g'):
                    val = float(val)
                result[key] = _validated(key, val)
            except (ValueError, TypeError, OverflowError):
                pass
    if 'l' in result:
        result['enabled'] = True
    count = params.get('TCS Phase count')
    try:
        if count is not None:
            num = int(count)
            if num in (3,4):
                result['multi_count'] = num
            elif num == 2:
                result['multi_count'] = 0
                result['dual'] = True
    except (ValueError, TypeError, OverflowError):
        pass
    if 'TCS Phases L/G' in params:
        try:
            rows = [part.strip() for part in params['TCS Phases L/G'].split(';')]
            for i, row in enumerate(rows[:4]):
                ls, gs = (float(piece.strip()) for piece in row.split('/'))
                if i == 0:
                    continue  # CLIP-L and CLIP-G already separately serialized.
                if i == 1:
                    if result.get('multi_count') in (3,4):
                        result['phase2_l'] = _validated('phase2_l',ls)
                        result['phase2_g'] = _validated('phase2_g',gs)
                    else:
                        result['late_l'] = _validated('late_l',ls)
                        result['late_g'] = _validated('late_g',gs)
                elif i == 2:
                    result['third_l'] = _validated('third_l',ls)
                    result['third_g'] = _validated('third_g',gs)
                elif i == 3:
                    result['fourth_l'] = _validated('fourth_l',ls)
                    result['fourth_g'] = _validated('fourth_g',gs)
        except (ValueError, TypeError, OverflowError):
            pass
    # Very old PNGs may record multiple phase rows but omit Phase count.
    # Do not infer enabled: it cannot be proven from a partial legacy record.
    if ('late_l' in result and 'late_g' in result and
            'dual' not in result and 'multi_count' not in result and
            count is None and 'TCS Phases L/G' in params and
            len(params['TCS Phases L/G'].split(';')) == 2):
        result['dual'] = True
    if 'TCS Phase boundaries' in params:
        try:
            boundaries = [float(x) for x in params['TCS Phase boundaries'].split(',')]
            for i, key in enumerate(('start2','start3','start4'), 1):
                if len(boundaries) > i:
                    result[key] = _validated(key,boundaries[i])
        except (ValueError, TypeError, OverflowError):
            pass
    return result


def normalize_preset(data):
    """Accept older named presets with missing keys; reject corrupted values."""
    if not isinstance(data, dict):
        raise ValueError('preset must contain JSON settings')
    merged = dict(DEFAULTS)
    merged.update({key:data[key] for key in FIELD_ORDER if key in data})
    # Before Multi-Phase had dedicated Phase 2 controls, it shared late_l/g.
    if 'phase2_l' not in data and 'late_l' in data:
        merged['phase2_l'] = data['late_l']
    if 'phase2_g' not in data and 'late_g' in data:
        merged['phase2_g'] = data['late_g']
    return validate(merged)
