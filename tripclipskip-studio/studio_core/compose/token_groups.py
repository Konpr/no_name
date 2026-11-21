"""WP-02C: conservative, read-only selection of WHOLE verified source segments.

A parser segment is not a learned concept and matching weight is not semantic
binding. Sub-segment BPE offsets cannot be inferred from the segment-level WP02B
trace, so partial matches MUST NOT receive token slots. No model inference,
conditioning edits, tokenizer calls, hooks, or mutable global state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .token_trace import TokenSpanTrace

MAX_GROUP_QUERY = 256


@dataclass(frozen=True)
class GroupSelection:
    status: str   # exact_whole_segments | partial_segment | ambiguous | not_found | unavailable | empty
    source_start: int | None = None
    source_end: int | None = None
    segment_indices: tuple[int, ...] = ()
    slots: tuple[tuple[int, int], ...] = ()
    reason: str = ''

    @property
    def safe_whole_segment_match(self) -> bool:
        return self.status == 'exact_whole_segments'


def select_verified_group(*, prompt: str, query: str, trace: TokenSpanTrace) -> GroupSelection:
    """Match ONE unique literal spelling against proven whole parser segments.

    This API never authorizes edits. `safe_whole_segment_match` only describes
    a read-only mapping, not permission to alter conditioning or to assume a
    semantic concept. All offsets are Python str indexes, not BPE byte offsets.
    """
    if not isinstance(prompt, str) or not isinstance(query, str):
        return GroupSelection('unavailable', reason='text and query must be strings')
    if not query:
        return GroupSelection('empty', reason='no group requested')
    if len(query) > MAX_GROUP_QUERY:
        return GroupSelection('unavailable', reason='group query too long')
    if (trace.status != 'exact_segments' or
            hashlib.sha256(prompt.encode('utf-8')).hexdigest() != trace.prompt_sha256):
        return GroupSelection('unavailable', reason='native segment trace is not fully verified for this prompt')
    occurrences = []
    start = 0
    while True:
        i = prompt.find(query, start)
        if i < 0:
            break
        occurrences.append((i, i + len(query)))
        if len(occurrences) > 1:
            return GroupSelection('ambiguous', reason='group spelling occurs more than once; no location selected')
        start = i + 1
    if not occurrences:
        return GroupSelection('not_found', reason='literal group spelling absent from original prompt')
    a, b = occurrences[0]
    segments = sorted((s for s in trace.segments if s.token_count),
                      key=lambda s: (s.source_start if s.source_start is not None else -1, s.index))
    fully_inside = [s for s in segments if s.source_start is not None and
                    s.source_end is not None and a <= s.source_start and s.source_end <= b]
    # Only a continuous union of complete segments is eligible: no markup,
    # skipped punctuation, BREAK marker, or partial BPE-derived groups.
    if (not fully_inside or fully_inside[0].source_start != a or
            fully_inside[-1].source_end != b or
            any(x.source_end != y.source_start for x, y in zip(fully_inside, fully_inside[1:])) or
            any(s.source_certainty != 'exact' or s.token_certainty != 'exact' or not s.slots
                for s in fully_inside)):
        return GroupSelection('partial_segment', a, b,
                              reason='selection is not a continuous union of whole verified parser segments')
    slots = tuple(slot for segment in fully_inside for slot in segment.slots)
    if len(slots) != len(set(slots)) or sorted(slots) != list(slots):
        return GroupSelection('unavailable', reason='overlapping or out-of-order native slots')
    return GroupSelection('exact_whole_segments', a, b,
                          tuple(s.index for s in fully_inside), slots,
                          'verified parser-segment mapping; NOT semantic binding')


def render_group_selection(selection: GroupSelection, *, query_supplied: bool) -> str:
    if not query_supplied:
        return ('**Ручная группа:** не выбрана. Укажи фрагмент из исходного промпта, '
                'например `long hair`; это **только просмотр**.')
    title = '**Ручная группа · проверка совпадения сегментов:** '
    if selection.safe_whole_segment_match:
        slots = ', '.join(f'{c+1}:{p}' for c, p in selection.slots[:24])
        if len(selection.slots) > 24:
            slots += ', …'
        return (title + '**exact_whole_segments** · Python offsets '
                f'`{selection.source_start}:{selection.source_end}` · '
                f'{len(selection.slots)} BPE позиций (`{slots}`). '
                'Это подтверждение **целого сегмента**, НЕ нового понятия и НЕ per-BPE offsets.')
    return (title + '**' + selection.status + '**. '
            + selection.reason.replace('`', 'ˋ').replace('|', '¦')[:180] + '. '
            'Слоты не назначены; разрешения менять conditioning нет.')
