"""WP-02C UX: conservative, human-readable read-only summary.

Rendering only. Consumes already verified GroupSelection and BreakPlanPreview;
never infers per-BPE offsets, reruns tokenization, or edits conditioning.
"""
from __future__ import annotations

from html import escape

from .smart_break import BreakPlanPreview
from .token_groups import GroupSelection


def _code(value: str) -> str:
    """User-provided text inside an inert Markdown inline-code span."""
    text = escape(value.replace('`', 'ˋ').replace('\n', ' ↵ ')
                  .replace('\r', ' ').replace('\t', ' '), quote=True)
    return '`' + text[:350] + '`'


def _bar(used: int, capacity: int, width: int = 24) -> str:
    """Illustrative bar only; printed numeric counts remain authoritative."""
    blocks = round(width * used / capacity) if capacity > 0 else 0
    if used > 0:
        blocks = max(1, blocks)
    blocks = max(0, min(width, blocks))
    return '▰' * blocks + '░' * (width - blocks)


def group_summary(selection: GroupSelection, query: str) -> str:
    if not query:
        return '**Проверка группы:** не выбрана. Можешь ввести фразу над кнопкой.'
    head = '**Проверка фрагмента ' + _code(query) + ':** '
    if selection.status == 'exact_whole_segments':
        slots = ', '.join(f'{ci+1}:{pos}' for ci, pos in selection.slots[:12])
        if len(selection.slots) > 12:
            slots += ', …'
        chunks = sorted({c+1 for c, _ in selection.slots})
        location = ('чанк ' + str(chunks[0])) if len(chunks) == 1 else ('чанки ' + ', '.join(map(str, chunks)))
        return (head + '**найден целый проверенный сегмент.** '
                f'{len(selection.slots)} токеновых позиций, {location}; позиции `{slots}`. '
                'Это **не** объединение токенов в новое понятие.')
    if selection.status == 'partial_segment':
        return (head + '**найден внутри более длинного сегмента**, но его отдельные BPE-позиции '
                'не доказаны. Точные слоты не назначены; изменение conditioning запрещено.')
    if selection.status == 'ambiguous':
        return (head + '**неоднозначно:** фраза встречается несколько раз. '
                'Нужен выбор конкретного появления; слоты не назначены.')
    if selection.status == 'not_found':
        return head + '**не найдена в исходном тексте.**'
    if selection.status == 'unavailable':
        return head + '**проверить не удалось** — надёжного сопоставления нет. Слоты не назначены.'
    return head + '**совпадение не подтверждено.** Слоты не назначены.'


def break_summary(plan: BreakPlanPreview, occupied_per_chunk: tuple[int, ...]) -> str:
    if plan.status != 'available':
        return ('**Анализ BREAK:** недоступен: нативная карта и сегменты не прошли '
                'совместную проверку. Ничего не изменено.')
    out = ['**BREAK и использование чанков**',
           f'Сейчас: **{plan.native_chunks} чанка(ов)**, '
           f'**{plan.verified_content_tokens}** занятых из '
           f'**{plan.native_chunks * plan.capacity_per_chunk}** содержательных позиций; '
           f'**{plan.native_unused_content_slots}** незанятых.', '']
    for i, occupied in enumerate(occupied_per_chunk[:6], start=1):
        out.append(f'Чанк {i}: **{occupied}/{plan.capacity_per_chunk}**  '
                   f'`{_bar(occupied, plan.capacity_per_chunk)}`')
    if len(occupied_per_chunk) > 6:
        out.append(f'*Показаны первые шесть из {len(occupied_per_chunk)} чанков.*')
    out.append(f'Принудительных `BREAK`: **{plan.forced_breaks}**; '
               f'сегментов на границе чанков: **{plan.native_split_segments}**.')
    if plan.theoretical_min_chunks is not None:
        out.append(f'Если не учитывать принудительные разрывы, математический минимум — '
                   f'**{plan.theoretical_min_chunks} чанка(ов)**. '
                   'Это **оценка, а не повторная токенизация** изменённого текста.')
    if plan.whole_segment_greedy_chunks is None:
        out.append('Сохранить все сегменты целиком в гипотетической раскладке невозможно.')
    else:
        out.append(f'Гипотетическая укладка целых сегментов: **{plan.whole_segment_greedy_chunks}** '
                   'чанка(ов).')
    out.append('**Важно:** удаление `BREAK` меняет контекст CLIP и может изменить изображение. '
               'Анализ **ничего не переписывает и не генерирует**.')
    return '\n\n'.join(out)


def readable_card(*, name: str, branch: str, selection: GroupSelection, query: str,
                  plan: BreakPlanPreview, occupied_per_chunk: tuple[int, ...]) -> str:
    prefix = 'Положительный' if branch == 'positive' else 'Негативный'
    return '\n\n'.join([
        f'### {prefix} · {name}',
        group_summary(selection, query),
        break_summary(plan, occupied_per_chunk),
    ])
