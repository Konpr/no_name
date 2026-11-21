"""WP-02B: conservative, read-only parser-segment -> native CLIP slot trace.

This is not a semantic binder or an editable offset map. Parser text, BPE IDs,
weights and chunk slots are independently cross-checked against the *already*
produced tokenize_line() chunks. No CLIP forward and no monkey patches.

The parser may merge adjacent equal-weight fragments, so a parser segment is
NOT necessarily an attention group, word, or concept. Character spans are only
certified when their exact source spelling has a unique monotone assignment.
Individual BPE offsets are deliberately unknown without validated tokenizer
character-offset support. TI vectors and literal EOT content fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Any, Callable

from .tokenizer_map import NativeTokenMap

MAX_SEGMENTS = 512
MAX_MATCHES_PER_SEGMENT = 64
MAX_BPE_TOKENS = 8192


@dataclass(frozen=True)
class SegmentTrace:
    index: int
    weight: float
    token_count: int
    source_start: int | None
    source_end: int | None
    source_certainty: str                 # exact | unknown
    slots: tuple[tuple[int, int], ...]   # (chunk zero-based, slot including BOS)
    token_certainty: str                  # exact | unknown
    reason: str = ''


@dataclass(frozen=True)
class TokenSpanTrace:
    source_id: str
    branch: str
    encoder: str
    prompt_sha256: str
    segments: tuple[SegmentTrace, ...]
    status: str                          # exact_segments | partial | unavailable
    reason: str = ''

    @property
    def exact_count(self) -> int:
        return sum(x.source_certainty == 'exact' and x.token_certainty == 'exact'
                   for x in self.segments)


def _unavailable(snapshot: NativeTokenMap, reason: str) -> TokenSpanTrace:
    return TokenSpanTrace(snapshot.source_id, snapshot.branch, snapshot.encoder,
                          snapshot.prompt_sha256, (), 'unavailable', reason)


def _candidate_spans(prompt: str, pieces: list[str]) -> list[list[tuple[int, int]]] | None:
    """Find all ordered exact source spellings, not fuzzy/normalized matches."""
    candidates = []
    for piece in pieces:
        if not piece:
            return None
        spans = []
        start = 0
        while True:
            index = prompt.find(piece, start)
            if index < 0:
                break
            spans.append((index, index + len(piece)))
            if len(spans) > MAX_MATCHES_PER_SEGMENT:
                return None
            start = index + 1
        if not spans:
            return None
        candidates.append(spans)
    return candidates


def _unique_monotone_spans(prompt: str, pieces: list[str]) -> tuple[tuple[int, int], ...] | None:
    """Require ONE nonoverlapping ordered mapping for the entire parsed stream.

    Ambiguous repetitions return None, rather than choosing first/last match.
    Short fragments coinciding with punctuation in markup cannot be certified
    when they admit multiple ordered source placements.
    """
    if not pieces:
        return ()
    cands = _candidate_spans(prompt, pieces)
    if cands is None:
        return None

    @lru_cache(maxsize=65536)
    def count(i: int, previous_end: int) -> int:
        if i == len(cands):
            return 1
        total = 0
        for start, end in cands[i]:
            if start < previous_end:
                continue
            total += count(i + 1, end)
            if total >= 2:
                return 2
        return total

    if count(0, 0) != 1:
        return None
    result = []
    previous_end = 0
    for i, matches in enumerate(cands):
        selected = [(a, b) for a, b in matches
                    if a >= previous_end and count(i + 1, b) == 1]
        if len(selected) != 1:
            return None
        result.append(selected[0])
        previous_end = selected[0][1]
    return tuple(result)


def _pairs(parsed: Any) -> list[tuple[str, float]]:
    if not isinstance(parsed, (list, tuple)) or len(parsed) > MAX_SEGMENTS:
        raise ValueError('Unsupported parsed segment count')
    pairs = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError('Invalid native parsed attention item')
        text, weight = item
        if not isinstance(text, str) or isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError('Invalid attention text or weight')
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError('Non-finite parsed attention weight')
        pairs.append((text, weight))
    return pairs


def trace_native_segments(*, prompt: str, snapshot: NativeTokenMap,
                          wrapper: Any, native_chunks: Any,
                          parse_attention: Callable[[str], Any] | None,
                          emphasis_disabled: bool = False) -> TokenSpanTrace:
    """Inspect parser output and check it AGAINST native tokenized chunks.

    Pass the real A1111 parse_prompt_attention when emphasis is on. No extra
    tokenize_line() call. wrapper.tokenize() is called once with the same parsed
    segments A1111 feeds to its tokenizer; it does NOT invoke the transformer.

    Fail closed for TI, literal EOT tokens, missing parser, unsupported sizes,
    mismatched weights/IDs, unsupported BREAK placement or ambiguous text spans.
    """
    if not isinstance(prompt, str):
        return _unavailable(snapshot, 'invalid prompt')
    if hashlib.sha256(prompt.encode('utf-8')).hexdigest() != snapshot.prompt_sha256:
        return _unavailable(snapshot, 'source fingerprint mismatch')
    if not callable(getattr(wrapper, 'tokenize', None)):
        return _unavailable(snapshot, 'native tokenize() unavailable')
    if not emphasis_disabled and not callable(parse_attention):
        return _unavailable(snapshot, 'native attention parser unavailable')
    try:
        parsed = [(prompt, 1.0)] if emphasis_disabled else parse_attention(prompt)
        pairs = _pairs(parsed)
        if not pairs:
            return _unavailable(snapshot, 'empty parsed stream')
        tokens_by_piece = wrapper.tokenize([piece for piece, _ in pairs])
        if not isinstance(tokens_by_piece, (list, tuple)) or len(tokens_by_piece) != len(pairs):
            return _unavailable(snapshot, 'native tokenize output count mismatch')
        pieces = []
        breaks = []
        for (piece, weight), ids in zip(pairs, tokens_by_piece):
            is_break = piece == 'BREAK' and weight == -1
            breaks.append(is_break)
            if is_break:
                pieces.append(())
                continue
            if not isinstance(ids, (list, tuple)):
                return _unavailable(snapshot, 'native tokenizer returned unsupported IDs')
            if len(ids) > MAX_BPE_TOKENS:
                return _unavailable(snapshot, 'tokenized segment too large')
            if any(type(t) is not int or t < 0 or t == wrapper.id_end for t in ids):
                return _unavailable(snapshot, 'BPE includes invalid or literal EOS IDs')
            pieces.append(tuple(ids))
        if sum(len(p) for p in pieces) > MAX_BPE_TOKENS:
            return _unavailable(snapshot, 'tokenized prompt too large')
        native = tuple(native_chunks)
        if len(native) != len(snapshot.chunks):
            return _unavailable(snapshot, 'native chunk count changed')
        # The supplied map and raw native chunks must belong to the same capture.
        # This protects standalone consumers from accidental cross-prompt mixing.
        for src_chunk, mapped_chunk in zip(native, snapshot.chunks):
            if (len(src_chunk.tokens) != len(mapped_chunk.slots)
                    or len(src_chunk.multipliers) != len(mapped_chunk.slots)):
                return _unavailable(snapshot, 'native snapshot length mismatch')
            if any(src_chunk.tokens[k] != slot.token_id or
                   float(src_chunk.multipliers[k]) != slot.multiplier
                   for k, slot in enumerate(mapped_chunk.slots)):
                return _unavailable(snapshot, 'native snapshot content mismatch')
        if any(getattr(chunk, 'fixes', ()) for chunk in native):
            return _unavailable(snapshot, 'Textual Inversion requires a dedicated trace')

        # Parser-segment offsets are intentionally not assigned at BPE granularity.
        # BREAK is a parser sentinel, not an ordinary source substring.
        normal_pieces = [piece for (piece, _), is_break in zip(pairs, breaks) if not is_break and piece]
        spans = _unique_monotone_spans(prompt, normal_pieces)
        span_i = 0
        c_i, pos = 0, 1
        traces = []
        for seg_i, ((piece, weight), ids, is_break) in enumerate(zip(pairs, pieces, breaks)):
            if is_break:
                # Native BREAK discards unused positions and emits a chunk. Even
                # when current chunk is empty, this is an explicit chunk boundary.
                if c_i >= len(native):
                    return _unavailable(snapshot, 'BREAK beyond native chunk')
                tail = native[c_i].tokens[pos:1+snapshot.chunk_length]
                tail_w = native[c_i].multipliers[pos:1+snapshot.chunk_length]
                if any(t != wrapper.id_end for t in tail) or any(float(w) != 1.0 for w in tail_w):
                    return _unavailable(snapshot, 'BREAK leaves unverified native tokens')
                c_i += 1
                pos = 1
                # A1111 does not append an empty final chunk after trailing BREAK.
                traces.append(SegmentTrace(seg_i, weight, 0, None, None,
                                           'unknown', (), 'exact', 'BREAK separator'))
                continue
            if piece:
                match = spans[span_i] if spans is not None else (None, None)
                span_i += 1
            else:
                match = (None, None)
            positions = []
            for token in ids:
                while True:
                    if c_i >= len(native):
                        return _unavailable(snapshot, 'native token stream ended early')
                    if pos <= snapshot.chunk_length:
                        actual_id = native[c_i].tokens[pos]
                        actual_weight = float(native[c_i].multipliers[pos])
                        if actual_id == token and actual_weight == weight:
                            positions.append((c_i, pos))
                            pos += 1
                            break
                        if actual_id != wrapper.id_end:
                            return _unavailable(snapshot, 'native token/weight mismatch')
                    remaining = native[c_i].tokens[pos:1+snapshot.chunk_length]
                    remaining_weights = native[c_i].multipliers[pos:1+snapshot.chunk_length]
                    if any(t != wrapper.id_end for t in remaining) or any(float(w) != 1.0 for w in remaining_weights):
                        return _unavailable(snapshot, 'native chunk has unmatched content')
                    c_i += 1
                    pos = 1
            traces.append(SegmentTrace(seg_i, weight, len(ids), match[0], match[1],
                                       'exact' if spans is not None and piece else 'unknown',
                                       tuple(positions), 'exact',
                                       '' if spans is not None else 'source spelling ambiguous or transformed'))
        # The entire remaining native stream must consist solely of EOS padding,
        # including explicit BREAK-created empty chunks.
        for tail_ci in range(c_i, len(native)):
            start = pos if tail_ci == c_i else 1
            remaining = native[tail_ci].tokens[start:1+snapshot.chunk_length]
            weights = native[tail_ci].multipliers[start:1+snapshot.chunk_length]
            if any(t != wrapper.id_end for t in remaining) or any(float(w) != 1.0 for w in weights):
                return _unavailable(snapshot, 'unmatched tokens after parsed stream')
        exact = sum(t.source_certainty == 'exact' and t.token_certainty == 'exact' for t in traces)
        status = 'exact_segments' if exact == sum(1 for t in traces if t.token_count) and exact > 0 else 'partial'
        return TokenSpanTrace(snapshot.source_id, snapshot.branch, snapshot.encoder,
                              snapshot.prompt_sha256, tuple(traces), status,
                              'parser segments are not semantic concepts')
    except Exception as exc:
        return _unavailable(snapshot, f'verification failed: {type(exc).__name__}')


def render_span_trace(trace: TokenSpanTrace, *, max_rows: int = 24) -> str:
    """No raw prompt/token contents are rendered or serialized."""
    if trace.status == 'unavailable':
        return ('**WP-02B · привязка сегментов:** не подтверждена (' +
                trace.reason.replace('`', '').replace('|', '/')[:120] + ').')
    lines = ['**WP-02B · сегменты парсера и исходные фрагменты**',
             f'Статус: **{trace.status}** · подтверждено: **{trace.exact_count}** '
             f'из **{sum(t.token_count > 0 for t in trace.segments)}** непустых сегментов.',
             'Сегмент парсера — **не обязательно смысловая группа**: соседние фрагменты '
             'с одинаковыми весами могут объединиться. Исходные координаты относятся '
             'к целому сегменту; **не к отдельным BPE-токенам**. Генерация не меняется.',
             '', '| Сегмент | Символы Python [start:end) | Вес | BPE | Позиции (чaнк:слот) |',
             '|---:|:---|---:|---:|:---|']
    shown = 0
    for seg in trace.segments:
        if not seg.token_count:
            continue
        if shown >= max_rows:
            break
        shown += 1
        source = f'{seg.source_start}:{seg.source_end}' if seg.source_certainty == 'exact' else 'unknown'
        positions = ', '.join(f'{ci+1}:{slot}' for ci, slot in seg.slots[:8])
        if len(seg.slots) > 8:
            positions += ' …'
        lines.append(f'| {seg.index+1} | `{source}` | {seg.weight:.4g} | {seg.token_count} | {positions} |')
    if shown == 0:
        lines.append('| — | — | — | 0 | — |')
    return '\n'.join(lines)
