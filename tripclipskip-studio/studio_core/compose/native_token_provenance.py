"""WP-03B1c: privacy-preserving *observation* of A1111 prompt chunks.

No tokenizer calls, no encoder calls, no raw token IDs, prompt strings, TI
embedding names/weights, or tensor references are serialized. The HMAC key is
job-local and intentionally never written to the report. This proves equality
of *pre-pad-rewrite* token ID matrices only inside that job. It does NOT prove
final content provenance through custom composition or U-Net consumption.
"""
from __future__ import annotations

from hashlib import sha256
from hmac import new as hmac_new
import re
import struct

MAX_BATCH = 16
MAX_TOKENS_PER_CHUNK = 256
MAX_FIXES_PER_CHUNK = 64
MAX_OPERATORS_SCAN = 16
MAX_SCAN_CHARS = 16384
OPERATOR_MARKERS = ('ASSEMBLE', 'BLEND', 'MORPH', 'POOL', 'CHUNK', 'BIND', 'REGION')
# Lexical hints deliberately do not claim that A1111's actual parser accepted
# or executed an operator. REGION has a different scanner from other operators.
_PREFIX = r'(?<![\w\\])'
_MARKERS = {
    'ASSEMBLE': re.compile(_PREFIX + r'ASSEMBLE\s*\{'),
    'POOL': re.compile(_PREFIX + r'POOL\s*\{'),
    'CHUNK': re.compile(_PREFIX + r'CHUNK(?:\s*\[[^\]]*\])?\s*\{'),
    'BIND': re.compile(_PREFIX + r'BIND(?:\s*\^[^{]*)?\s*\{'),
    'BLEND': re.compile(_PREFIX + r'BLEND(?:\s*\^[^\[{]*)?(?:\s*\[[^\]]*\])?\s*\{'),
    'MORPH': re.compile(_PREFIX + r'MORPH(?:\s*\^[^\[{@]*)?(?:\s*@[^\[{]*)?(?:\s*\[[^\]]*\])?\s*\{'),
    'REGION': re.compile(r'REGION\{', re.IGNORECASE),
}


def observe_prompt_markers(required_prompts):
    """Return bounded marker presence, never source text or a parsed AST."""
    if not isinstance(required_prompts, list):
        return {'status': 'UNAVAILABLE', 'markers': [], 'evidence': 'NOT_PARSED'}
    found = set()
    truncated = len(required_prompts) > MAX_OPERATORS_SCAN
    for prompt in required_prompts[:MAX_OPERATORS_SCAN]:
        if type(prompt) is not str:
            return {'status': 'UNAVAILABLE', 'markers': [], 'evidence': 'NOT_PARSED'}
        if len(prompt) > MAX_SCAN_CHARS:
            truncated = True
        fragment = prompt[:MAX_SCAN_CHARS]
        found.update(name for name, regex in _MARKERS.items() if regex.search(fragment))
    return {
        'status': 'PARTIAL_SCAN' if truncated else 'LEXICAL_SCAN_ONLY',
        'markers': sorted(found),
        'evidence': 'SYNTAX_HINT_ONLY_NOT_EXECUTION_OR_OUTPUT_PROVENANCE',
    }


def _is_int(value):
    return type(value) is int and -2**63 <= value < 2**63


def chunk_input_summary(tokens, multipliers, *, id_start, id_end, id_pad, key):
    """Read exactly the A1111 process_tokens arguments before native encoding.

    Only bounded plain-list matrices of ordinary Python ints/floats accepted.
    Digest uses a random, non-persisted per-job HMAC key to avoid offline
    dictionary attacks on short tokenized prompts.
    """
    unknown = {'status': 'UNSUPPORTED_INPUT', 'semantics': 'PRE_PADDING_REWRITE'}
    if type(tokens) not in (list, tuple) or not 1 <= len(tokens) <= MAX_BATCH:
        return unknown
    if type(multipliers) not in (list, tuple) or len(multipliers) != len(tokens):
        return unknown
    if any(type(row) not in (list, tuple) or not 1 <= len(row) <= MAX_TOKENS_PER_CHUNK
           for row in tokens):
        return unknown
    size = len(tokens[0])
    if any(len(row) != size for row in tokens):
        return unknown
    if any(type(row) not in (list, tuple) or len(row) != size for row in multipliers):
        return unknown
    if not all(_is_int(tok) for row in tokens for tok in row):
        return unknown
    if not all(type(weight) in (int, float) for row in multipliers for weight in row):
        return unknown
    digest = hmac_new(key, digestmod=sha256)
    digest.update(struct.pack('<II', len(tokens), size))
    for row in tokens:
        for tok in row:
            digest.update(struct.pack('<q', tok))
    start = id_start if _is_int(id_start) else None
    end = id_end if _is_int(id_end) else None
    pad = id_pad if _is_int(id_pad) else None
    return {
        'status': 'OBSERVED_PRE_PADDING_REWRITE',
        'semantics': 'A1111_PROCESS_TOKENS_INPUT_NOT_TRANSFORMER_INPUT',
        'batch_size': len(tokens), 'tokens_per_chunk': size,
        'token_ids_hmac': digest.hexdigest(),
        'digest_scope': 'JOB_LOCAL_KEY_NOT_PERSISTED_NOT_CROSS_JOB_COMPARABLE',
        'all_rows_start_with_declared_bos': all(row[0] == start for row in tokens) if start is not None else None,
        'first_eos_position': [next((i for i, x in enumerate(row) if x == end), None)
                               for row in tokens] if end is not None else None,
        'eos_equals_padding': end == pad if end is not None and pad is not None else None,
        'weight_modified_positions': [sum(weight != 1.0 for weight in row) for row in multipliers],
    }


def ti_fixes_summary(fixes):
    """Metadata for TI fixes *before* the encoder consumes hijack.fixes."""
    if type(fixes) not in (list, tuple) or len(fixes) > MAX_BATCH:
        return {'status': 'UNAVAILABLE'}
    out = []
    for batch_fixes in fixes:
        if type(batch_fixes) not in (tuple, list) or len(batch_fixes) > MAX_FIXES_PER_CHUNK:
            return {'status': 'UNAVAILABLE'}
        rows = []
        for fix in batch_fixes:
            try:
                offset, embedding = fix
                vectors = getattr(embedding, 'vectors', None)
                if type(offset) is not int or not (0 <= offset <= MAX_TOKENS_PER_CHUNK):
                    return {'status': 'UNAVAILABLE'}
                if type(vectors) is not int or not (0 < vectors <= MAX_TOKENS_PER_CHUNK):
                    return {'status': 'UNAVAILABLE'}
                rows.append({'offset_in_chunk_without_bos': offset, 'vectors': vectors})
            except Exception:
                return {'status': 'UNAVAILABLE'}
        out.append(rows)
    return {'status': 'OBSERVED_A1111_PRE_ENCODER_FIXES',
            'fix_count_by_batch_item': [len(row) for row in out],
            'fixes_by_batch_item': out,
            'embedding_names_and_values': 'NOT_RECORDED',
            'effective_embedding_application': 'NOT_VERIFIED'}
