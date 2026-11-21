"""Experimental SDXL absolute hidden-state/block timeline.

Pure conversion from explicit layer selectors to the existing tested
PhaseSchedule/ClipController. No A1111 imports, weights, or hooks here.

L[-k] is a zero-based HF hidden_states[-k] *raw* tensor (not necessarily
last_hidden_state or post-final-LayerNorm). G is a one-based OpenCLIP block.
Native pooled-G stays native. The experimental penultimate selector intentionally
restricts token-wise G to blocks 1..(N-1). The SDXL FrozenOpenCLIPEmbedder2
can still execute block N to compute its pooled vector: it is deliberately
not exposed as a selectable token-wise layer in this preview.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import re

from studio_core.multiphase import Phase, PhaseSchedule

_LINE = re.compile(r'^\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\s*\|\s*(-[1-9]\d*)\s*\|\s*([1-9]\d*)\s*$')


@dataclass(frozen=True)
class AbsoluteRow:
    start: float
    l_hidden_negative: int
    g_block: int
    l_relative: int
    g_relative: int


@dataclass(frozen=True)
class AbsolutePlan:
    rows: tuple[AbsoluteRow, ...]
    schedule: PhaseSchedule

    def summary(self):
        return '; '.join(f'P{i}@{row.start:.2f}: L[-{row.l_hidden_negative}] G{row.g_block}'
                         for i, row in enumerate(self.rows, 1))


def absolute_schedule(text: str, *, l_blocks: int, g_blocks: int, g_anchor: str) -> AbsolutePlan:
    """Parse 2–4 lines: ``0.00 | -2 | 31`` (fraction | L[-N] | G block).

    Unlike relative TCS mode, the L anchor is the final raw hidden state:
    relative=1 -> hidden_states[-1], relative=2 -> hidden_states[-2].
    This deliberately bypasses the A1111 native reference for this opt-in mode.
    """
    if not isinstance(text, str) or len(text) > 1024:
        raise ValueError('Абсолютное расписание должно быть текстом до 1024 символов')
    if not (isinstance(l_blocks, int) and 1 <= l_blocks <= 128):
        raise ValueError('Неизвестная архитектура CLIP-L')
    if not (isinstance(g_blocks, int) and 2 <= g_blocks <= 128):
        raise ValueError('Неизвестная архитектура OpenCLIP-G')
    if g_anchor not in ('penultimate', 'last'):
        raise ValueError('Неизвестная опора CLIP-G; требуется penultimate или last')
    base_g = g_blocks - (g_anchor == 'penultimate')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not 2 <= len(lines) <= 4:
        raise ValueError('Укажи от двух до четырёх строк фаз')
    rows = []
    phases = []
    for i, line in enumerate(lines, 1):
        m = _LINE.fullmatch(line)
        if not m:
            raise ValueError(f'Строка {i}: формат «0.00 | -2 | 31» (доля | hidden_states[-N] | блок G)')
        start, l_raw, g_raw = float(m.group(1)), int(m.group(2)), int(m.group(3))
        l_negative = -l_raw
        if not math.isfinite(start) or not 0 <= start < 1:
            raise ValueError(f'Фаза {i}: начало должно быть в диапазоне [0,1)')
        if i == 1 and start != 0:
            raise ValueError('Фаза 1 должна начинаться с 0.00')
        if i > 1 and start <= rows[-1].start:
            raise ValueError('Границы фаз должны строго возрастать')
        if not 1 <= l_negative <= l_blocks + 1:
            raise ValueError(f'Фаза {i}: CLIP-L hidden_states[-{l_negative}] вне диапазона [-1..-{l_blocks+1}]')
        if not 1 <= g_raw <= base_g:
            label = (f' (G{g_blocks} недоступен в экспериментальном режиме penultimate; '
                     'штатный SDXL может выполнять этот блок для pooled)'
                     if g_anchor == 'penultimate' else '')
            raise ValueError(f'Фаза {i}: G block {g_raw} вне доступного диапазона 1..{base_g}{label}')
        l_rel = l_negative
        g_rel = base_g - g_raw + 1
        rows.append(AbsoluteRow(start, l_negative, g_raw, l_rel, g_rel))
        phases.append(Phase(start, float(l_rel), float(g_rel)))
    schedule = PhaseSchedule(tuple(phases))
    return AbsolutePlan(tuple(rows), schedule)


def absolute_preview(plan: AbsolutePlan, *, l_blocks: int, g_blocks: int, g_anchor: str):
    header = ('### Абсолютные слои · проверка до генерации\n\n'
              '| Фаза | Начало | CLIP-L (сырой HF hidden state) | OpenCLIP-G | Перевод в TCS |\n'
              '|---|---:|---|---|---|')
    lines = [header]
    for i, row in enumerate(plan.rows, 1):
        label = f'**входные embeddings**' if row.l_hidden_negative == l_blocks + 1 else f'выход блока {l_blocks + 1 - row.l_hidden_negative}'
        lines.append(f'| P{i} | {row.start:.0%} | `hidden_states[-{row.l_hidden_negative}]` ({label}) | '
                     f'блок {row.g_block}/{g_blocks} | L={row.l_relative}, G={row.g_relative} |')
    lines.append('\n**Только предпросмотр.** L выбирается из *raw `hidden_states`*, а не из `last_hidden_state`; '
                 'финальная LayerNorm не добавляется. Pooled CLIP-G остаётся штатным. '
                 f'Опора G: `{g_anchor}`; выбор G за пределами разрешённого диапазона запрещён. '
                 'При `penultimate` последний блок G не предлагается как token-wise выход, '
                 'хотя в обычном SDXL он может выполняться для native pooled. '
                 'Моменты переключения определяются выбранным источником прогресса; '
                 'это не гарантия конкретного UI-шага другого самплера. '
                 'REGION отключает переключение фаз, Hires остаётся на P1.')
    return '\n'.join(lines)
