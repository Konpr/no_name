"""Pure read-only, explanatory views of native A1111 token/chunk results.

Works on already tokenized data; never invokes model conditioning or changes state.
All estimates refer to positional occupancy, not attention strength or image quality.
"""
from __future__ import annotations

import re
import unicodedata

BREAK_PATTERN = re.compile(r'\s*\bBREAK\b\s*', re.S)


def strip_break_for_comparison(prompt: str) -> str:
    """Produce an explicitly labelled counterfactual input, not a prompt rewrite.

    Mirrors A1111's BREAK word boundary detection, but only for a user-invoked
    diagnostic comparison; original text and A1111 settings are never changed.
    """
    return BREAK_PATTERN.sub(' ', prompt)


def has_break(prompt: str) -> bool:
    return BREAK_PATTERN.search(prompt) is not None


def occupancy_map(occupied: int, capacity: int, ti_offsets=(), weights=None, *, max_chars: int = 75) -> str:
    """One character = one content slot for CLIP's 75-slot chunks.

    For non-standard capacities above max_chars the view must not pretend to be
    a 1:1 map; return a textual summary rather than downsample silently.
    """
    if capacity > max_chars:
        return f'Карта для нестандартного чанка {capacity} позиций не выводится (занято {occupied}).'
    ti = set(ti_offsets)
    weights = weights or []
    def symbol(i):
        if i >= occupied:
            return '·'
        if i in ti:
            return '◆'
        if i < len(weights) and weights[i] > 1.000001:
            return '+'
        if i < len(weights) and weights[i] < .999999:
            return '-'
        return '■'
    slots = [symbol(i) for i in range(capacity)]
    return ' '.join(''.join(slots[i:i + 15]) for i in range(0, capacity, 15))


def overview_lines(*, name: str, lengths: list[int], capacity: int,
                   ti_offsets: list[set[int]], token_count: int,
                   weights: list[list[float]], prompt: str) -> list[str]:
    """Explain exact chunk allocation without assuming what a BREAK means artistically."""
    total = len(lengths)
    occupied = sum(lengths)
    padding = total * capacity - occupied
    bytes_total = total * (capacity + 2)
    weighted_up = sum(1 for ws in weights for w in ws if w > 1.000001)
    weighted_down = sum(1 for ws in weights for w in ws if w < .999999)
    ti_count = sum(len(x) for x in ti_offsets)
    lines = [
        f'**Карта позиций — {name}:** ■ = обычный токен · + = усиленный · - = ослабленный · ◆ = TI · · = EOT-паддинг. '
        'BOS/EOS находятся по краям каждого чанка.',
    ]
    for i, (n, t) in enumerate(zip(lengths[:12], ti_offsets[:12]), 1):
        pct = n / capacity * 100
        lines.append(f'**Чанк {i}** · {n}/{capacity} ({pct:.0f}%) · '
                     f'{capacity-n} позиций паддинга · '
                     f'BOS `{occupancy_map(n, capacity, t, weights[i-1])}` EOS')
    if total > 12:
        lines.append(f'Показаны карты первых 12 из {total} чанков; статистика рассчитана по всему тексту.')
    lines += [
        f'**Итог:** {occupied} содержательных позиций, {padding} паддинг-позиций, '
        f'{bytes_total} входных позиций CLIP ({total} × {capacity + 2}). '
        f'Штатный счётчик A1111: **{token_count}**.',
        f'**Веса A1111:** усилены {weighted_up} позиций (>1), ослаблены {weighted_down} (<1), '
        f'TI-векторов {ti_count}. Это *веса токенизации*, а не измерение силы признаков изображения.',
    ]
    if total > 1:
        early = [(i, n) for i, n in enumerate(lengths[:-1], 1) if n < capacity]
        if early:
            freed = sum(capacity - n for _, n in early)
            names = ', '.join(f'{i} ({n}/{capacity})' for i, n in early[:6])
            lines.append(
                f'**Внимание — раннее закрытие:** чанк(и) {names} '
                f'закрылись до заполнения; суммарно {freed} позиций занято паддингом '
                f'перед следующим чанком. Возможные причины — `BREAK` или '
                f'`comma_padding_backtrack`. Это не ошибка и не оценка качества.'
            )
            if any(n <= max(1, int(capacity * .2)) for _, n in early):
                lines.append('**Подсказка:** один из ранних чанков заполнен не более чем на 20%. '
                             'Если такое разбиение не было задумано, проверь расположение `BREAK`.')
        if has_break(prompt):
            lines.append('В тексте присутствует `BREAK`: он принудительно отделяет чанки; '
                         'запятая после него не нужна. Один перенос строки без `BREAK` чанки не разделяет.')
    if weighted_up or weighted_down:
        lines.append('**Как читать веса:** вес `0.9` рядом с запятой нормален: A1111 назначает '
                     'вес всему текстовому фрагменту, включая его знаки препинания. '
                     '`1.2` не означает «изображение будет на 20% лучше»; итоговое conditioning '
                     'может дополнительно нормализоваться.')
    if padding:
        lines.append('Паддинг остаётся частью 77-позиционной последовательности. '
                     '«Свободные позиции» здесь не означают, что CLIP полностью их игнорирует.')
    return lines


def ru_chunks(n: int) -> str:
    return f'{n} ' + ('чанк' if n % 10 == 1 and n % 100 != 11 else 'чанка' if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else 'чанков')


def compare_encoders(l: dict, g: dict, *, title: str) -> str:
    """Compare *tokenization*, never imply encoder semantic equivalence."""
    a, b = l['ids'], g['ids']
    if len(a) != len(b):
        return (f'**CLIP-L ↔ OpenCLIP-G ({title}):** количество содержательных позиций '
                f'различается ({len(a)} и {len(b)}). Энкодеры могут работать с разными токенами.')
    differing = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not differing:
        note = f'**CLIP-L ↔ OpenCLIP-G ({title}):** все {len(a)} Token ID совпадают.'
    else:
        locations = ', '.join(f'позиция {i}: L={x}, G={y}' for i, x, y in differing[:6])
        note = (f'**CLIP-L ↔ OpenCLIP-G ({title}):** отличаются {len(differing)} '
                f'из {len(a)} Token ID ({locations}' + ('…' if len(differing) > 6 else '') + ').')
    if l['lengths'] != g['lengths']:
        note += f' Границы чанков также отличаются: L={l["lengths"][:8]}, G={g["lengths"][:8]}.'
    note += ' Совпадение Token ID **не означает совпадения эмбеддингов** у 12- и 32-блочных энкодеров.' 
    return note


def compare_prompt_ids(positive: dict, negative: dict, *, encoder: str) -> str:
    """Diagnostic token overlap, not a prediction of CFG or prompt conflict."""
    pos = set(i for i in positive['ids'] if i not in (0, positive['id_end']))
    neg = set(i for i in negative['ids'] if i not in (0, negative['id_end']))
    common = pos & neg
    if not common:
        return f'**{encoder}:** совпадающих пользовательских Token ID между positive/negative не обнаружено.'
    show = ', '.join(str(v) for v in sorted(common)[:12])
    return (f'**{encoder}:** {len(common)} общих различных Token ID в positive и negative '
            f'({show}' + ('…' if len(common) > 12 else '') + '). '
            'Включая пунктуацию; это **не** измерение CFG и не признак конфликта промптов.')


def unicode_notes(text: str) -> list[str]:
    non_ascii = [(i, c) for i, c in enumerate(text) if ord(c) > 127]
    hidden = [(i, c) for i, c in enumerate(text)
              if unicodedata.category(c) in ('Cf', 'Cc') and c not in '\n\r\t']
    replacement = '\ufffd' in text
    if not non_ascii and not hidden:
        return []
    result = []
    if non_ascii:
        examples = ', '.join(f'U+{ord(c):04X} ({unicodedata.name(c, "неизвестный символ")})'
                             for _, c in non_ascii[:4])
        result.append(f'**Unicode:** {len(non_ascii)} символов вне ASCII (например, {examples}). '
                      'CLIP использует byte-level BPE; части таких символов иногда выглядят '
                      'как бессмысленные сочетания букв в таблице. Это не обязательно ошибка токенизации.')
    if hidden:
        examples = ', '.join(f'U+{ord(c):04X}' for _, c in hidden[:5])
        result.append(f'**Невидимые символы:** найдено {len(hidden)} ({examples}). '
                      'Если их не добавляли намеренно, проверь скопированный текст.')
    if replacement:
        result.append('**Символ U+FFFD:** обнаружен знак замены повреждённого текста (�). '
                      'Он может указывать на потерю данных при копировании или декодировании.')
    return result
