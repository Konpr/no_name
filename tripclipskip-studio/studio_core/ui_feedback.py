"""Pure read-only helpers for safe TCS UI restore and diagnostics (R2).

Only presentation is changed: these helpers never mutate a recipe, prompt,
model, Gradio component or saved PNG value.
"""
from __future__ import annotations

import html
import math


def visibility(state):
    """Compute visibility for (G, reference, multi, fourth, dual, disabled notice)."""
    count = state['multi_count']
    try:
        count = int(count)
    except (ValueError, TypeError, OverflowError):
        count = 0
    return (bool(state['separate_g']),
            state['reference_source'] == 'custom',
            count in (3, 4),
            count == 4,
            bool(state['dual']) and count == 0,
            not bool(state['enabled']))


def phase_boundary_warning(start2, start3, start4, multi_count):
    """Validate current active phase fractions without silently rewriting preset values."""
    try:
        count = int(multi_count)
        if count not in (3, 4):
            return 'Границы Multi-Phase не используются (режим Dual-Phase или одна фаза).'
        parts = (float(start2), float(start3), float(start4))
        if not all(math.isfinite(x) for x in parts):
            raise ValueError('нечисловая граница')
        a,b,c = parts
        if not 0.0 < a < b < 1.0 or (count == 4 and not b < c < 1.0):
            return '**Ошибка границ фаз:** требуется 0 < start2 < start3' + (' < start4' if count == 4 else '') + ' < 1. Исправь значения до генерации.'
        used = (a,b,c) if count == 4 else (a,b)
        return 'Расписание границ корректно: ' + ', '.join(f'{x:.0%}' for x in used) + '.'
    except (ValueError, TypeError, OverflowError):
        return '**Ошибка:** невозможно проверить границы фаз; исправь значения до генерации.'


def native_skip_warning(value):
    """Never silently round values from historical presets or metadata."""
    try:
        v = float(value)
        if not math.isfinite(v) or not v.is_integer():
            return '**Внимание:** штатный Clip Skip должен быть целым числом. При активной custom-опоре дробное значение приведёт к отказу TCS в этом job.'
    except (ValueError, TypeError, OverflowError):
        return '**Внимание:** штатный Clip Skip должен быть целым числом.'
    return ''


def markdown_exception(exc, limit=240):
    """Return inert single-line content suitable for a Markdown code span."""
    raw = f'{type(exc).__name__}: {str(exc)[:limit]}'
    raw = raw.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    raw = raw.replace('`', 'ˋ').replace('|', '¦')
    return html.escape(raw, quote=True)


def markdown_label(value, limit=96):
    text = str(value).replace('\r', ' ').replace('\n', ' ').replace('`', 'ˋ')[:limit]
    return html.escape(text, quote=True).replace('|', '¦')


def inspector_report_context(p, phase):
    """Display the observation's job context, not the currently selected checkpoint."""
    model = getattr(p, 'sd_model', None)
    info = getattr(model, 'sd_checkpoint_info', None)
    checkpoint = (getattr(p, 'sd_model_checkpoint', None)
                  or getattr(info, 'title', None)
                  or getattr(info, 'model_name', None)
                  or 'не указан')
    seeds = getattr(p, 'all_seeds', None)
    seed = (seeds[0] if isinstance(seeds, (list, tuple)) and seeds else
            getattr(p, 'seed', 'не указан'))
    return ('*Контекст измерения (снимок на момент генерации):* '
            + f'checkpoint `{markdown_label(checkpoint)}`, '
            + f'seed `{markdown_label(seed, 40)}`, '
            + f'фаза `{markdown_label(phase, 30)}`. '
            + 'При смене модели/seed обнови измерение новой генерацией.\n\n')
