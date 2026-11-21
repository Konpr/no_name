"""WP-02A: read-only native A1111 token-slot snapshots (no CLIP forward).

This is deliberately NOT a text-span alignment engine. A1111's tokenize_line()
returns token IDs, weights and TI fixes, but no original-character offsets. The
EOS ID is also commonly used for padding, so a trailing run of EOS tokens can
be genuinely ambiguous, particularly after BREAK. The uncertainty is retained
rather than inferred from text length, regexes or a separate tokenizer.

Only the exported native structural slots are exact. No generated map authorizes
per-token conditioning edits; WP-02B must establish source spans separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any


class TokenSnapshotError(ValueError):
    """Invalid or internally inconsistent native A1111 tokenization snapshot."""


@dataclass(frozen=True)
class TokenSlot:
    chunk_index: int                 # zero-based chunk in A1111 stream
    slot_index: int                  # raw slot, 0=BOS; chunk_length+1=EOS
    token_id: int                    # native ID; TI placeholders are NOT BPE IDs
    multiplier: float
    role: str                        # bos | eos | token | ti | eot_or_padding | padding
    padding: bool | None             # None means not determinable from native snapshot
    source_start: None = None        # WP-02A NEVER invents character offsets
    source_end: None = None
    source_certainty: str = 'unknown'
    ti_name: str | None = None
    ti_vector_index: int | None = None   # zero based
    ti_vector_count: int | None = None


@dataclass(frozen=True)
class ChunkSnapshot:
    index: int
    slots: tuple[TokenSlot, ...]
    occupied_lower_bound: int
    occupied_upper_bound: int
    exact_content_length: int | None

    @property
    def ambiguous_slots(self) -> int:
        return sum(s.padding is None for s in self.slots)


@dataclass(frozen=True)
class NativeTokenMap:
    source_id: str
    branch: str
    encoder: str
    prompt_sha256: str               # hash of manually supplied text; not runtime trace
    chunk_length: int
    native_token_count: int
    chunks: tuple[ChunkSnapshot, ...]
    structural_exact: bool = True    # native ID/weight/position, NOT char offsets
    source_spans_exact: bool = False

    @property
    def ambiguous_slots(self) -> int:
        return sum(c.ambiguous_slots for c in self.chunks)

    @property
    def slot_count(self) -> int:
        return sum(len(c.slots) for c in self.chunks)


def _int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TokenSnapshotError(f'{label} must be an integer')
    return value


def _native_ti(fixes: Any, *, chunk_length: int, chunk_index: int) -> dict[int, tuple[str, int, int]]:
    result: dict[int, tuple[str, int, int]] = {}
    for fix in fixes:
        offset = _int(getattr(fix, 'offset', None), 'TI offset')
        embedding = getattr(fix, 'embedding', None)
        vectors = _int(getattr(embedding, 'vectors', None), 'TI vectors')
        name = getattr(embedding, 'name', None)
        if not isinstance(name, str) or not name:
            raise TokenSnapshotError('TI embedding has no name')
        if vectors < 1 or vectors > chunk_length or offset < 0 or offset + vectors > chunk_length:
            raise TokenSnapshotError(f'TI fix outside native chunk {chunk_index}')
        for vector_index in range(vectors):
            content_index = offset + vector_index
            if content_index in result:
                raise TokenSnapshotError(f'Overlapping TI fixes in chunk {chunk_index}')
            result[content_index] = (name, vector_index, vectors)
    return result


def snapshot_from_native_chunks(*, chunks: Any, token_count: int, wrapper: Any,
                                source_id: str, branch: str, encoder: str,
                                prompt: str) -> NativeTokenMap:
    """Snapshot already tokenized A1111 chunks; never re-tokenizes or encodes.

    `chunk.tokens` are the pre-encode native IDs. With SD2/OpenCLIP, the encoder
    may subsequently replace tokens following EOT with id_pad; this function
    explicitly does not claim to capture those post-processing changes.
    """
    if branch not in ('positive', 'negative') or encoder not in ('l', 'g'):
        raise TokenSnapshotError('Unknown branch or encoder')
    if not isinstance(source_id, str) or not source_id or len(source_id) > 64:
        raise TokenSnapshotError('source_id must be a nonempty <=64-character string')
    if not isinstance(prompt, str):
        raise TokenSnapshotError('prompt must be a string')
    chunk_length = _int(getattr(wrapper, 'chunk_length', None), 'chunk_length')
    if not 1 <= chunk_length <= 2048:
        raise TokenSnapshotError('Unsupported native chunk_length')
    bos = _int(getattr(wrapper, 'id_start', None), 'BOS')
    eos = _int(getattr(wrapper, 'id_end', None), 'EOS')
    if bos == eos:
        raise TokenSnapshotError('BOS and EOS cannot have the same ID')
    native_count = _int(token_count, 'token_count')
    chunk_items = tuple(chunks)
    if not chunk_items or len(chunk_items) > 256:
        raise TokenSnapshotError('Unsupported native chunk count')
    if not (chunk_length * (len(chunk_items) - 1) <= native_count <= chunk_length * len(chunk_items)):
        raise TokenSnapshotError('Native token_count disagrees with chunk count')

    snapshots: list[ChunkSnapshot] = []
    for ci, chunk in enumerate(chunk_items):
        ids = tuple(_int(v, 'token_id') for v in getattr(chunk, 'tokens', ()))
        weights = tuple(getattr(chunk, 'multipliers', ()))
        if len(ids) != chunk_length + 2 or len(weights) != len(ids):
            raise TokenSnapshotError(f'Invalid native chunk {ci} length')
        if ids[0] != bos or ids[-1] != eos:
            raise TokenSnapshotError(f'Invalid BOS/EOS in chunk {ci}')
        parsed_weights = []
        for w in weights:
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w):
                raise TokenSnapshotError(f'Invalid token weight in chunk {ci}')
            parsed_weights.append(float(w))
        ti = _native_ti(getattr(chunk, 'fixes', ()), chunk_length=chunk_length, chunk_index=ci)
        content = ids[1:-1]
        last_non_eot = max((i for i, tid in enumerate(content) if tid != eos), default=-1)
        lower = last_non_eot + 1
        is_last = ci == len(chunk_items) - 1
        last_count = native_count - chunk_length * (len(chunk_items) - 1)
        # This is exact only when the final chunk was flushed with is_last=True.
        # A1111's explicit trailing BREAK flush uses is_last=False -> 75 even
        # if the actual last chunk had only a few content tokens.
        exact_length = last_count if is_last and last_count < chunk_length else None
        if exact_length is not None and lower > exact_length:
            raise TokenSnapshotError('Native token_count conflicts with visible content')
        upper = exact_length if exact_length is not None else chunk_length
        slots: list[TokenSlot] = []
        for slot_idx, (token_id, weight) in enumerate(zip(ids, parsed_weights)):
            is_padding: bool | None = False
            ti_name: str | None = None
            ti_index: int | None = None
            ti_count: int | None = None
            if slot_idx == 0:
                role = 'bos'
            elif slot_idx == chunk_length + 1:
                role = 'eos'
            else:
                pos = slot_idx - 1
                if pos in ti:
                    ti_name, ti_index, ti_count = ti[pos]
                    if token_id != 0:
                        raise TokenSnapshotError('TI slot must contain native zero placeholder')
                    role = 'ti'
                elif token_id != eos or pos <= last_non_eot:
                    role = 'token'
                elif exact_length is not None:
                    if pos >= exact_length:
                        role, is_padding = 'padding', True
                    else:
                        role = 'token'  # actual user EOT at a known occupied slot
                else:
                    role, is_padding = 'eot_or_padding', None
                if is_padding is True and weight != 1.0:
                    raise TokenSnapshotError('Native padding has a non-default weight')
            slots.append(TokenSlot(ci, slot_idx, token_id, weight, role, is_padding,
                                   ti_name=ti_name, ti_vector_index=ti_index,
                                   ti_vector_count=ti_count))
        snapshots.append(ChunkSnapshot(ci, tuple(slots), lower, upper, exact_length))
    return NativeTokenMap(source_id, branch, encoder, hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                          chunk_length, native_count, tuple(snapshots))


def snapshot_native_token_map(*, wrapper: Any, prompt: str, source_id: str,
                              branch: str, encoder: str) -> NativeTokenMap:
    """Call only the loaded wrapper's native tokenize_line(), exactly once."""
    tokenize = getattr(wrapper, 'tokenize_line', None)
    if not callable(tokenize):
        raise TokenSnapshotError('A1111 wrapper has no native tokenize_line')
    chunks, token_count = tokenize(prompt)
    return snapshot_from_native_chunks(chunks=chunks, token_count=token_count,
                                       wrapper=wrapper, source_id=source_id,
                                       branch=branch, encoder=encoder, prompt=prompt)


def summarize_native_token_map(snapshot: NativeTokenMap) -> str:
    """Short safe-to-render Markdown excerpt; no raw prompt, ids or TIFF values."""
    lines = ['**WP-02A · карта фактических позиций (`tokenize_line`)**',
             f'Чанков: **{len(snapshot.chunks)}**, позиций: **{snapshot.slot_count}**, '
             f'неразличимых EOS/padding: **{snapshot.ambiguous_slots}**.',
             'Для всех токенов исходные символьные offsets — **unknown**. '
             'Позиции, ID, веса и TI-флаги взяты из нативных чанков; '
             'это **не** трасса runtime-макросов и не разрешение на редактирование токенов.']
    if snapshot.ambiguous_slots:
        lines.append('Части завершающих EOS совпадают с padding по ID: '
                     'не выдаём предполагаемую длину за точно известную.')
    return '\n\n'.join(lines)
