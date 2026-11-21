"""Pure, read-only Markdown view of the *requested* CLIP layer selection.

No A1111 imports, model calls, tensors, hooks, global option changes or caches.
Runtime and this view intentionally share layer_map.py and multiphase.py.
"""
from __future__ import annotations

from studio_core.layer_map import clip_g_layer_map, clip_l_layer_map
from studio_core.interpolation import skip_pair
from studio_core.multiphase import effective_phase2_g, schedule_from_ui


def _phases(*, l, g, separate, dual, late_l, late_g, fraction, multi_count,
            phase2_l, phase2_g, third_l, third_g, fourth_l, fourth_g,
            start2, start3, start4):
    first_g = float(g) if separate else float(l)
    if int(multi_count) == 0 and not dual:
        return [(0.0, float(l), first_g)]
    p2_g = effective_phase2_g(
        independent=bool(separate), phase2_l=phase2_l, phase2_g=phase2_g,
        legacy_l=late_l, legacy_g=late_g)
    plan = schedule_from_ui(
        l, first_g, late_l, float(late_g) if separate else float(late_l), fraction,
        multi_count=multi_count,
        phase2_l=phase2_l, phase2_g=p2_g,
        third_l=third_l, third_g=float(third_g) if separate else float(third_l),
        fourth_l=fourth_l, fourth_g=float(fourth_g) if separate else float(fourth_l),
        start2=start2, start3=start3, start4=start4)
    return [(p.start, p.clip_l, p.clip_g) for p in plan.phases]


def _layer_strip(depth: int, indexes: tuple[int, ...], *, include_embedding=False) -> str:
    """Small no-HTML visual. Output positions are actual one-based blocks (plus E)."""
    cells = ['E' if include_embedding else '0'] if include_embedding else []
    cells.extend('·' for _ in range(depth))
    for index in indexes:
        pos = index if include_embedding else index - 1
        if not 0 <= pos < len(cells):
            raise ValueError(f'Layer position {index} is outside model history')
        cells[pos] = '●'
    return ''.join(cells)


def render_layer_map(*, mode, reference, depth_l, blocks_g=None, anchor_g=None,
                     interpolation='lerp', l=1.0, g=1.0, separate=True,
                     dual=False, late_l=2.0, late_g=2.0, fraction=0.5,
                     multi_count=0, phase2_l=2.0, phase2_g=2.0,
                     third_l=2.5, third_g=2.5, fourth_l=3.0, fourth_g=3.0,
                     start2=0.25, start3=0.5, start4=0.75, locale="en"):
    """Render the requested effective layer choices; do NOT claim they ran.

    Raises on bad inputs exactly where the runtime schedule/mapping rejects them.
    """
    if mode not in ('sd1', 'sdxl'):
        raise ValueError('This preview supports only SD1.x and SDXL')
    if interpolation not in ('lerp', 'slerp'):
        raise ValueError('Unknown interpolation method')
    if mode == 'sdxl' and (not isinstance(blocks_g, int) or blocks_g < 2 or
                           anchor_g not in ('penultimate', 'last')):
        raise ValueError('Unsupported CLIP-G layout or model anchor')
    # _phases calls schedule_from_ui() and therefore PhaseSchedule validation.
    phases = _phases(
        l=l, g=g, separate=separate, dual=dual, late_l=late_l, late_g=late_g,
        fraction=fraction, multi_count=multi_count, phase2_l=phase2_l,
        phase2_g=phase2_g, third_l=third_l, third_g=third_g,
        fourth_l=fourth_l, fourth_g=fourth_g, start2=start2,
        start3=start3, start4=start4)
    if locale not in ('ru', 'en'):
        raise ValueError(f'Unsupported Layer Map locale: {locale!r}')
    if locale == 'ru':
        rows = ['| Фаза | Начало | Выход CLIP-L | Выход OpenCLIP-G |',
                '|:---|---:|:---|:---|']
    else:
        rows = ['| Phase | Start | CLIP-L output | OpenCLIP-G output |',
                '|:---|---:|:---|:---|']
    diagrams = []
    for i, (start, lv, gv) in enumerate(phases, 1):
        l_text = clip_l_layer_map(reference, depth=depth_l, relative=lv, family=mode)
        g_text = (clip_g_layer_map(blocks=blocks_g, model_anchor=anchor_g, relative=gv)
                  if mode == 'sdxl' else 'Not present in SD1.x')
        if locale == 'ru':
            l_text = (l_text.replace(' (final LayerNorm)', ' (финальная LayerNorm)')
                      .replace(' (input embedding)', ' (входной эмбеддинг)'))
            g_text = (g_text.replace('block ', 'блок ')
                      .replace('pooled=native', 'pooled=штатный')
                      .replace('Not present in SD1.x', 'Отсутствует в SD1.x'))
        rows.append(f'| P{i} | {start:.0%} | `{l_text}` | `{g_text}` |')
        lo, hi, blend = skip_pair(lv, reference.index + 1)
        l_idxs = (reference.index - lo + 1,) if not blend else (reference.index - lo + 1, reference.index - hi + 1)
        diagrams.append(f'P{i} L [E, 1..{depth_l:02d}]  '
                        + _layer_strip(depth_l, l_idxs, include_embedding=True))
        if mode == 'sdxl':
            base = blocks_g - (1 if anchor_g == 'penultimate' else 0)
            glo, ghi, gblend = skip_pair(gv, base)
            g_idxs = (base - glo + 1,) if not gblend else (base - glo + 1, base - ghi + 1)
            diagrams.append(f'P{i} G [1..{blocks_g:02d}]     '
                            + _layer_strip(blocks_g, g_idxs))
    if locale == 'ru':
        # UI-only translation: numerical layer mapping, schedules and PNG Info
        # remain unchanged.  Preserve native/technical identifiers in code.
        description = reference.description
        if description.startswith('SDXL model hidden_states['):
            description = (f'Слой, заданный моделью SDXL: hidden_states[{reference.index}] '
                           f'(= hidden_states[-{depth_l + 1 - reference.index}])')
        elif description.startswith('SDXL A1111 override:'):
            description = (f'Переопределение A1111 для SDXL: '
                           f'hidden_states[-{depth_l + 1 - reference.index}]')
        elif description == 'SDXL model last hidden state':
            description = 'Слой модели SDXL: last_hidden_state (финальная LayerNorm)'
        elif description.startswith('SD1 Clip Skip'):
            description = f'SD1.x, штатный Clip Skip {reference.sd1_native_skip}'
        return ('### Карта слоёв CLIP — выбранные настройки\n\n'
                f'**Модель:** {"SDXL" if mode == "sdxl" else "SD1.x"} · '
                f'**Слоёв CLIP-L:** {depth_l} · '
                f'**Блоков OpenCLIP-G:** {blocks_g if mode == "sdxl" else "нет"} · '
                f'**Интерполяция:** `{interpolation}`\n\n'
                f'**Опорный слой CLIP-L:** `{description}`\n\n'
                + ('**Проверка границ:** корректно, фазы построены из PhaseSchedule.\n\n'
                   if int(multi_count) in (3, 4) or dual else
                   '**Проверка:** активна одна фаза.\n\n')
                + '\n'.join(rows) + '\n\n'
                + '**Карта блоков** (`●` — выбранный блок, `·` — не выбран, '
                  '`E` — входной эмбеддинг; две точки означают смешивание слоёв):\n\n'
                + '```text\n' + '\n'.join(diagrams) + '\n```\n\n'
                + '**Как читать карту:** показаны запрошенные слои и веса смешивания. '
                  '`1.0` — выбранная опора, а не обязательно глобальный Clip Skip 1. '
                  'Pooled-вектор CLIP-G остаётся **штатным**. При `slerp` проценты '
                  'описывают параметры интерполяции, а не линейные доли тензоров.\n\n'
                  '**Только просмотр:** текстовые эмбеддинги не пересчитываются; '
                  'глобальные настройки WebUI и хуки attention не меняются. '
                  'Эта карта **не подтверждает** фактическое выполнение фаз, '
                  'совместимость с REGION, работу Hires Fix или эквивалентность изображений на GPU. '
                  'Фактические переключения смотри в Phase Trace. '
                  'Обновляй карту после смены checkpoint или штатного Clip Skip в WebUI.')
    return ('### CLIP Layer Map — requested configuration\n\n'
            f'**Model:** {"SDXL" if mode == "sdxl" else "SD1.x"} · '
            f'**CLIP-L blocks:** {depth_l} · '
            f'**OpenCLIP-G blocks:** {blocks_g if mode == "sdxl" else "not present"} · '
            f'**Interpolation:** `{interpolation}`\n\n'
            f'**CLIP-L reference:** `{reference.description}`\n\n'
            + '\n'.join(rows) + '\n\n'
            + '**Block selection map** (`●` = selected block, `·` = not selected, '
              '`E` = input embedding; two dots selected = fractional blend):\n\n'
            + '```text\n' + '\n'.join(diagrams) + '\n```\n\n'
            '**Interpretation:** The table shows requested layer indices and blend weights. '
            '`1.0` is the selected anchor, not necessarily global Clip Skip 1. '
            'CLIP-G pooled stays **native**. For `slerp`, percentages are blending '
            'parameters, not linear tensor contributions.\n\n'
            '**Read-only:** no tensors are recomputed; no global WebUI options or '
            'attention hooks are changed. This preview does **not** confirm phase '
            'execution, REGION compatibility, Hires behavior or GPU image equivalence. '
            'Use Phase Trace for observed calls. Refresh after a checkpoint or WebUI '
            'Clip Skip change.')
