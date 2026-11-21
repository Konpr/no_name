"""WP-02C: read-only, quantitative BREAK/chunk efficiency preview.

Counterfactuals are *estimates* based on observed parser-segment token lengths.
They do not re-tokenize a rewritten prompt, change CLIP attention context,
assert visual quality, or construct a prompt to feed into generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from .token_trace import TokenSpanTrace
from .tokenizer_map import NativeTokenMap


@dataclass(frozen=True)
class BreakPlanPreview:
    status: str             # available | unavailable
    native_chunks: int = 0
    capacity_per_chunk: int = 0
    verified_content_tokens: int = 0
    native_unused_content_slots: int = 0
    forced_breaks: int = 0
    native_split_segments: int = 0
    theoretical_min_chunks: int | None = None
    whole_segment_greedy_chunks: int | None = None
    whole_segment_boundaries: tuple[int, ...] = ()  # before zero-based segment index
    reason: str = ''


def preview_break_layout(*, native_map: NativeTokenMap,
                         trace: TokenSpanTrace) -> BreakPlanPreview:
    if (trace.status != 'exact_segments' or trace.prompt_sha256 != native_map.prompt_sha256 or
            trace.source_id != native_map.source_id or trace.branch != native_map.branch or
            trace.encoder != native_map.encoder):
        return BreakPlanPreview('unavailable', reason='native map and full segment trace not verified together')
    cap = native_map.chunk_length
    if type(cap) is not int or cap <= 0:
        return BreakPlanPreview('unavailable', reason='invalid native chunk capacity')
    pieces = [s for s in trace.segments if s.token_count > 0]
    count = sum(s.token_count for s in pieces)
    native_slots = cap * len(native_map.chunks)
    if count == 0 or count > native_slots:
        return BreakPlanPreview('unavailable', reason='no verified content or token count exceeds native capacity')
    flat = [slot for s in pieces for slot in s.slots]
    if (len(flat) != count or len(set(flat)) != count or
            any(c < 0 or c >= len(native_map.chunks) or p < 1 or p > cap for c, p in flat)):
        return BreakPlanPreview('unavailable', reason='invalid or duplicate native segment slots')
    # Verify that each reported slot has a matching native map token role.
    if any(native_map.chunks[ci].slots[pos].role in ('bos', 'eos', 'padding', 'eot_or_padding')
           for ci, pos in flat):
        return BreakPlanPreview('unavailable', reason='segment positions contain uncertain EOS/padding')
    forced = sum(s.reason == 'BREAK separator' for s in trace.segments)
    split_native = sum(len(set(ci for ci, _ in s.slots)) > 1 for s in pieces)
    theoretical = (count + cap - 1) // cap
    # Next-fit is a hypothetical packing of WHOLE parser segments. They need
    # not be semantic concepts, and removing BREAK can alter native parsing.
    if any(s.token_count > cap for s in pieces):
        greedy = None
        boundaries: tuple[int, ...] = ()
    else:
        used = 0
        chunk_count = 1
        candidate_boundaries = []
        for s in pieces:
            if used + s.token_count > cap:
                chunk_count += 1
                used = 0
                candidate_boundaries.append(s.index)
            used += s.token_count
        greedy = chunk_count
        boundaries = tuple(candidate_boundaries)
    return BreakPlanPreview('available', len(native_map.chunks), cap, count,
                            native_slots - count, forced, split_native,
                            theoretical, greedy, boundaries,
                            'counterfactual counts only; NOT a validated rewritten prompt')


def render_break_preview(plan: BreakPlanPreview) -> str:
    if plan.status != 'available':
        return '**Smart BREAK · предварительный анализ:** недоступен (' + plan.reason[:170] + ').'
    theoretical = str(plan.theoretical_min_chunks) if plan.theoretical_min_chunks is not None else 'unknown'
    greedy = str(plan.whole_segment_greedy_chunks) if plan.whole_segment_greedy_chunks is not None else 'нельзя сохранить все сегменты целиком'
    return ('**Smart BREAK · аналитика (не модифицирует промпт)**\n\n'
            f'- Нативно: **{plan.native_chunks} чанка(ов)** × {plan.capacity_per_chunk} '
            f'= {plan.native_chunks * plan.capacity_per_chunk} содержательных слотов; '
            f'**{plan.verified_content_tokens} занято** и **{plan.native_unused_content_slots} свободно**.\n'
            f'- Принудительных `BREAK`: **{plan.forced_breaks}**; '
            f'сегментов, пересекающих границу чанка: **{plan.native_split_segments}**.\n'
            f'- Нижняя граница без принудительных разрывов: **{theoretical} чанка(ов)**; '
            f'гипотетическая укладка целых сегментов: **{greedy}**.\n\n'
            '**Ограничение:** это оценка по уже подтверждённым длинам сегментов. '
            'Она **не повторяет нативную токенизацию изменённого промпта**, '
            'не учитывает возможное изменение BPE/пунктуации/контекста, '
            'не обещает качество изображения и не создаёт новую команду `BREAK`.')
