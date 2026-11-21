"""Bounded, inert recipe-envelope validation (NOT wired into PNG Info).

Never serialize SourceRef.text, runtime tensors or model inputs implicitly.
This is a *config validator*, not an executable recipe compiler.
"""
from __future__ import annotations
import json
from .errors import ComposeError

MAX_BYTES = 4096
MAX_OPS = 16
MAX_SOURCES = 8
_ALLOWED_OPS = {'select', 'native'}  # WP-01 only; mix/replace added with their implementations
_ALLOWED_CHANNELS = {'l', 'g', 'pooled_g'}


def validate_inert_recipe(config):
    if not isinstance(config, dict):
        raise ComposeError('Recipe must be a JSON object')
    if set(config) - {'schema', 'enabled', 'model_family', 'sources', 'operations', 'fallback'}:
        raise ComposeError('Unknown recipe field')
    if config.get('schema') != 'tcs-recipe-v1' or type(config.get('enabled')) is not bool:
        raise ComposeError('Unknown schema or missing enabled boolean')
    if config.get('model_family') not in ('sd15', 'sdxl') or config.get('fallback') != 'fail_closed':
        raise ComposeError('Unknown family or unsafe fallback')
    sources = config.get('sources', [])
    operations = config.get('operations', [])
    if not isinstance(sources, list) or len(sources) > MAX_SOURCES or not isinstance(operations, list) or len(operations) > MAX_OPS:
        raise ComposeError('Recipe collection exceeds limits')
    names = set()
    for s in sources:
        if not isinstance(s, dict) or set(s) != {'id', 'branch', 'role'}:
            raise ComposeError('Sources must only contain id, branch, role (no prompt text)')
        name = s['id']
        if type(name) is not str or not (1 <= len(name) <= 64) or name in names:
            raise ComposeError('Invalid/duplicate source id')
        if s['branch'] not in ('positive', 'negative') or s['role'] not in _ALLOWED_CHANNELS:
            raise ComposeError('Invalid source branch/role')
        names.add(name)
    for o in operations:
        if not isinstance(o, dict) or set(o) != {'op', 'channel', 'inputs', 'params'}:
            raise ComposeError('Unknown operation fields')
        if type(o['op']) is not str or type(o['channel']) is not str or o['op'] not in _ALLOWED_OPS or o['channel'] not in _ALLOWED_CHANNELS:
            raise ComposeError('Unsupported operation (inert WP-01 accepts select/native only)')
        inputs = o['inputs']
        if not isinstance(inputs, list) or any(type(i) is not str or i not in names for i in inputs):
            raise ComposeError('Unknown input source')
        if o['op'] == 'native' and inputs or o['op'] == 'select' and len(inputs) != 1:
            raise ComposeError('Invalid number of inputs')
        if not isinstance(o['params'], dict) or o['params'] != {}:
            raise ComposeError('WP-01 does not execute operation params')
        if o['op'] == 'select' and next(s['role'] for s in sources if s['id'] == inputs[0]) != o['channel']:
            raise ComposeError('Selected source role does not match target channel')
    if config['model_family'] == 'sd15' and any(x['role'] in ('g','pooled_g') for x in sources):
        raise ComposeError('SD1.x cannot use G or pooled-G')
    raw = json.dumps(config, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf8')
    if len(raw) > MAX_BYTES:
        raise ComposeError('Recipe config oversized')
    return config


def recipe_json(config):
    validate_inert_recipe(config)
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
