"""CLIP Lab 2.0: safe server-rendered HTML from real native A1111 tokenizers.

Strictly read-only: no CLIP forward, torch, global options, DOM supplied by users,
network calls, or modifications to the generation arg contract.
The JS enhancer is optional; every view is legible without JavaScript.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math

from studio_core.absolute_timeline import absolute_schedule
from studio_core.interpolation import skip_pair
from studio_core.layer_map import clip_g_layer_map, clip_l_layer_map
from studio_core.layer_map_view import _phases
from studio_core.reference import LayerReference
from studio_core.token_inspector import _ti_slots, MAX_PROMPT_CHARS

MAX_CHUNKS = 8
MAX_CARDS_PER_STREAM = 160


def h(value) -> str:
    """HTML-escape all untrusted strings, including A1111 BPE fragments."""
    return escape(str(value), quote=True)


def _piece(token_id: int, *, tokenizer=None, decoder=None) -> str:
    if tokenizer is not None:
        convert = getattr(tokenizer, 'convert_ids_to_tokens', None)
        if callable(convert):
            value = convert(int(token_id))
            if isinstance(value, (tuple, list)):
                value = value[0] if value else '?'
            return str(value)
    if decoder is not None:
        return str(decoder.get(int(token_id), '?'))
    return f'ID:{int(token_id)}'


@dataclass(frozen=True)
class TokenRow:
    chunk: int
    position: int
    token_id: int
    piece: str
    weight: float
    is_ti: bool = False


@dataclass(frozen=True)
class TokenChunk:
    number: int
    occupied: int
    capacity: int
    rows: tuple[TokenRow, ...]


@dataclass(frozen=True)
class TokenStream:
    label: str
    name: str
    token_count: int
    chunks_total: int
    chunks: tuple[TokenChunk, ...]
    occupied_total: int
    capacity: int


def native_stream(wrapper, prompt: str, *, name: str, label: str,
                  decoder=None, max_chunks=MAX_CHUNKS) -> TokenStream:
    """Read existing A1111 tokenize_line result, with EOS/TI-aware occupancy."""
    if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError('Длина промпта превышает ограничение безопасного просмотра')
    tokenize = getattr(wrapper, 'tokenize_line', None)
    if not callable(tokenize):
        raise ValueError(f'{name}: штатный tokenize_line недоступен')
    chunks, counter = tokenize(prompt)
    if not chunks:
        raise ValueError(f'{name}: энкодер вернул пустой список чанков')
    capacity = int(getattr(wrapper, 'chunk_length', 75))
    if not 1 <= capacity <= 2048 or not 0 <= int(counter) <= capacity * len(chunks):
        raise ValueError(f'{name}: неожиданная длина чанка или счётчик токенов')
    token_count = int(counter)
    result = []
    occupied_total = 0
    tokenizer = getattr(wrapper, 'tokenizer', None) if name == 'CLIP-L' else None
    for index, chunk in enumerate(chunks, 1):
        ids = list(chunk.tokens)
        weights = list(chunk.multipliers)
        if len(ids) != capacity + 2 or len(weights) != capacity + 2:
            raise ValueError(f'{name}: длина чанка {index} не совпадает с A1111')
        if ids[0] != wrapper.id_start or ids[-1] != wrapper.id_end:
            raise ValueError(f'{name}: отсутствует BOS/EOS')
        content = ids[1:-1]
        occupied = 1 + max((i for i, value in enumerate(content)
                            if value != wrapper.id_end), default=-1)
        if index == len(chunks):
            # Same last-chunk-EOS disambiguation as Token Inspector 0.3.10.
            tail = token_count - capacity * (len(chunks) - 1)
            if 0 <= tail <= capacity:
                occupied = max(occupied, tail)
        ti = _ti_slots(chunk)
        if any(not 0 <= offset < occupied for offset in ti):
            raise ValueError(f'{name}: TI за границей чанка')
        occupied_total += occupied
        if index > max_chunks:
            continue
        rows = []
        for pos in range(occupied):
            weight = float(weights[pos + 1])
            if not math.isfinite(weight):
                raise ValueError(f'{name}: некорректный вес токена')
            ident = int(content[pos])
            rows.append(TokenRow(index, pos + 1, ident,
                                 ti.get(pos, _piece(ident, tokenizer=tokenizer, decoder=decoder)),
                                 weight, pos in ti))
        result.append(TokenChunk(index, occupied, capacity, tuple(rows)))
    return TokenStream(label, name, token_count, len(chunks),
                       tuple(result), occupied_total, capacity)


@dataclass(frozen=True)
class LayerPhase:
    start: float
    l_text: str
    g_text: str
    l_blocks: tuple[int, ...]  # zero means input embedding
    g_blocks: tuple[int, ...]


def resolve_phases(*, mode: str, depth_l: int, blocks_g: int | None,
                   anchor_g: str | None, reference: LayerReference | None,
                   absolute_enabled: bool, absolute_text: str,
                   interpolation: str, l: float, g: float, separate: bool,
                   dual: bool, late_l: float, late_g: float, fraction: float,
                   multi_count: int, phase2_l: float, phase2_g: float,
                   third_l: float, third_g: float, fourth_l: float, fourth_g: float,
                   start2: float, start3: float, start4: float) -> tuple[LayerPhase, ...]:
    """Use exactly the same schedule-to-layer mapping as the backend, not a mock."""
    if interpolation not in ('lerp', 'slerp'):
        raise ValueError('Поддерживаются только lerp и slerp')
    if absolute_enabled:
        if mode != 'sdxl' or blocks_g is None or anchor_g is None:
            raise ValueError('Абсолютный режим поддерживает только SDXL')
        plan = absolute_schedule(absolute_text, l_blocks=depth_l,
                                 g_blocks=blocks_g, g_anchor=anchor_g)
        return tuple(LayerPhase(row.start,
                                f'hidden_states[-{row.l_hidden_negative}] (raw)',
                                f'блок {row.g_block}/{blocks_g}; pooled=штатный',
                                (depth_l + 1 - row.l_hidden_negative,),
                                (row.g_block,)) for row in plan.rows)
    if reference is None:
        raise ValueError('Не определён опорный слой CLIP-L')
    settings = _phases(l=l, g=g, separate=separate, dual=dual, late_l=late_l,
                       late_g=late_g, fraction=fraction, multi_count=multi_count,
                       phase2_l=phase2_l, phase2_g=phase2_g, third_l=third_l,
                       third_g=third_g, fourth_l=fourth_l, fourth_g=fourth_g,
                       start2=start2, start3=start3, start4=start4)
    phases = []
    for start, lv, gv in settings:
        l_lo, l_hi, l_frac = skip_pair(lv, reference.index + 1)
        l_pos = tuple(dict.fromkeys((reference.index - l_lo + 1,)
                    + ((reference.index - l_hi + 1,) if l_frac else ())))
        l_name = clip_l_layer_map(reference, depth=depth_l, relative=lv, family=mode)
        if mode == 'sdxl':
            base = blocks_g - (1 if anchor_g == 'penultimate' else 0)
            g_lo, g_hi, g_frac = skip_pair(gv, base)
            g_pos = tuple(dict.fromkeys((base - g_lo + 1,)
                        + ((base - g_hi + 1,) if g_frac else ())))
            g_name = clip_g_layer_map(blocks=blocks_g, model_anchor=anchor_g, relative=gv)
        else:
            g_pos, g_name = (), 'OpenCLIP-G отсутствует (SD1.x)'
        phases.append(LayerPhase(start, l_name, g_name, l_pos, g_pos))
    return tuple(phases)


def _stream_view(stream: TokenStream, key: str, *, opened: bool = False) -> str:
    shown = sum(len(c.rows) for c in stream.chunks)
    header = (f'<details class="tcs-lab-stream"{" open" if opened else ""}><summary>{h(stream.label)} · {h(stream.name)} '
              f'— {stream.chunks_total} чанк(ов), {stream.occupied_total} токенов</summary>'
              '<div class="tcs-lab-stats">'
              f'<span>Счётчик A1111: <b>{stream.token_count}</b></span>'
              f'<span>Ёмкость чанка: <b>{stream.capacity}</b> (+ BOS/EOS)</span>'
              f'<span>Показано чанков: <b>{len(stream.chunks)}/{stream.chunks_total}</b></span></div>')
    card_blocks = []
    slot_blocks = []
    rows_count = 0
    for chunk in stream.chunks:
        chips = []
        slots = []
        for row in chunk.rows:
            if rows_count < MAX_CARDS_PER_STREAM:
                weight_type = 'up' if row.weight > 1.000001 else 'down' if row.weight < .999999 else 'normal'
                klass = ' ti' if row.is_ti else f' {weight_type}'
                keyrow = f'{key}-{row.chunk}-{row.position}'
                attrs = (f'data-tcs-token="{h(keyrow)}" data-tcs-id="{row.token_id}" '
                         f'data-tcs-piece="{h(row.piece)}" data-tcs-weight="{h(row.weight):s}" '
                         f'data-tcs-pos="{row.position}" data-tcs-chunk="{row.chunk}"')
                chips.append(f'<button type="button" class="tcs-lab-token{klass}" {attrs} '
                             f'title="Чанк {row.chunk}, позиция {row.position}, вес {row.weight:g}">'
                             f'<span class="tcs-lab-fragment">{h(row.piece)}</span>'
                             f'<small>#{row.token_id} · {row.chunk}:{row.position}</small></button>')
                rows_count += 1
            slot_kind = ('ti' if row.is_ti else 'up' if row.weight > 1.000001
                         else 'down' if row.weight < .999999 else 'normal')
            slots.append(f'<button type="button" class="tcs-lab-slot {slot_kind}" '
                         f'data-tcs-token="{h(key)}-{row.chunk}-{row.position}" '
                         f'data-tcs-id="{row.token_id}" data-tcs-piece="{h(row.piece)}" '
                         f'data-tcs-weight="{row.weight:g}" data-tcs-pos="{row.position}" '
                         f'data-tcs-chunk="{row.chunk}" '
                         f'title="{h(row.piece)} · ID {row.token_id} · вес {row.weight:g}">'
                         f'<span class="sr-only">Позиция {row.position}: {h(row.piece)}</span></button>')
        for _ in range(chunk.occupied, chunk.capacity):
            slots.append('<span class="tcs-lab-slot pad" title="Паддинг">'
                         '<span class="sr-only">Паддинг</span></span>')
        card_blocks.extend(chips)
        slot_blocks.append(f'<div class="tcs-lab-chunk"><div class="tcs-lab-chunk-title">'
                           f'<b>Чанк {chunk.number}</b><span>{chunk.occupied}/{chunk.capacity} · '
                           f'паддинг {chunk.capacity - chunk.occupied}</span></div>'
                           f'<div class="tcs-lab-slot-grid">{"".join(slots)}</div></div>')
    if stream.occupied_total > rows_count:
        card_blocks.append(f'<p class="tcs-lab-help">Для компактности показаны первые {rows_count} '
                           f'из {stream.occupied_total} токенов. Полный список доступен '
                           'в старом Token Inspector.</p>')
    # Keep full token tables in a closed details block, with safe escaped content.
    table_rows = []
    for chunk in stream.chunks:
        for row in chunk.rows:
            if len(table_rows) >= MAX_CARDS_PER_STREAM:
                break
            table_rows.append(f'<tr><td>{chunk.number}</td><td>{row.position}</td>'
                              f'<td>{row.token_id}</td><td>{h(row.piece)}</td>'
                              f'<td>{row.weight:g}</td></tr>')
    table = ('<details><summary>Техническая таблица Token ID</summary>'
             '<div class="tcs-lab-table-scroll"><table><thead><tr><th>Чанк</th><th>Поз.</th>'
             '<th>ID</th><th>BPE / TI</th><th>Вес</th></tr></thead><tbody>'
             + ''.join(table_rows) + '</tbody></table></div></details>')
    return (header + '<h4>Кликабельные BPE-токены</h4><div class="tcs-lab-chips">'
            + ''.join(card_blocks) + '</div><h4>Позиции чанков</h4>'
            + ''.join(slot_blocks) + table + '</details>')


def _layer_stack(label: str, depth: int, family: str, first: tuple[int, ...]) -> str:
    if not depth:
        return '<p>Энкодер отсутствует.</p>'
    buttons = []
    for block in range(depth, 0, -1):
        klass = ' chosen' if block in first else ''
        buttons.append(f'<div class="tcs-lab-layer{klass}" data-tcs-layer="{family}:{block}">'
                       f'<small>Блок {block}</small></div>')
    if family == 'l':
        buttons.append('<div class="tcs-lab-layer input" data-tcs-layer="l:0">E · входные эмбеддинги</div>')
    return (f'<article class="tcs-lab-layer-column"><h4>{h(label)} · {depth} блоков</h4>'
            '<div class="tcs-lab-layer-stack">' + ''.join(buttons) + '</div></article>')


def _phase_view(phases: tuple[LayerPhase, ...], *, depth_l: int,
                blocks_g: int | None) -> str:
    buttons = []
    for i, ph in enumerate(phases):
        stop = phases[i+1].start if i + 1 < len(phases) else 1.0
        end = max(0.0, stop - ph.start)
        firstclass = ' active' if i == 0 else ''
        attrs = (f'data-tcs-l="{h(",".join(map(str, ph.l_blocks)))}" '
                 f'data-tcs-g="{h(",".join(map(str, ph.g_blocks)))}" '
                 f'data-tcs-ltext="{h(ph.l_text)}" data-tcs-gtext="{h(ph.g_text)}"')
        buttons.append(f'<button type="button" class="tcs-lab-phase{firstclass}" '
                       f'data-tcs-action="phase" {attrs} style="flex:{max(end, .001):.5f}">'
                       f'P{i+1} <small>{ph.start:.0%}–{stop:.0%}</small></button>')
    a = phases[0]
    return ('<div class="tcs-lab-timeline">' + ''.join(buttons) + '</div>'
            '<div class="tcs-lab-layer-current" aria-live="polite">'
            f'<b>CLIP-L:</b> <span data-tcs-output="l">{h(a.l_text)}</span><br>'
            f'<b>OpenCLIP-G:</b> <span data-tcs-output="g">{h(a.g_text)}</span></div>'
            '<div class="tcs-lab-two-stacks">'
            + _layer_stack('CLIP-L', depth_l, 'l', a.l_blocks)
            + (_layer_stack('OpenCLIP-G', blocks_g, 'g', a.g_blocks) if blocks_g else '')
            + '</div>')


def render_lab(*, family: str, depth_l: int, blocks_g: int | None,
               enabled: bool, absolute_enabled: bool, phases: tuple[LayerPhase, ...],
               streams: tuple[TokenStream, ...], prompt: str, negative: str = '',
               variant: str = '', variant_stream: TokenStream | None = None,
               interpolation: str = 'lerp') -> str:
    """Return safe fully self-contained content for a gr.HTML output."""
    if not phases:
        raise ValueError('Для карты необходима хотя бы одна фаза')
    if len(prompt) > MAX_PROMPT_CHARS or len(negative) > MAX_PROMPT_CHARS:
        raise ValueError('Слишком длинный текст для просмотра')
    if absolute_enabled and family != 'sdxl':
        raise ValueError('Абсолютный режим доступен только SDXL')
    state = 'TCS выключен — предпросмотр не применяется' if not enabled else 'TCS включён — запрошенная конфигурация'
    strategy = 'Абсолютный Timeline' if absolute_enabled else 'Относительная глубина'
    pieces = ['<div class="tcs-lab" data-tcs-widget="lab">',
              '<div class="tcs-lab-header"><div><span class="tcs-lab-eyebrow">TripClipSkip Studio</span>'
              '<h3>CLIP Lab 2.0</h3><p>Промпт → BPE → чанки → слои L/G → фазы</p></div>'
              f'<span class="tcs-lab-status">{h(state)}</span></div>',
              f'<div class="tcs-lab-meta"><span>Модель: {h(family.upper())}</span>'
              f'<span>Режим: {h(strategy)}</span><span>CLIP-L: {depth_l} блоков</span>'
              f'<span>OpenCLIP-G: {blocks_g or 0} блоков</span>'
              '<span>pooled G: штатный</span><span>Источник: введённая копия</span></div>',
              '<p class="tcs-lab-note">Только просмотр — не выполнено новое CLIP-кодирование. '
              'Изменение текста здесь не меняет txt2img. Все токены каждого чанка проходят через '
              'последовательные блоки CLIP; выбранные выходы относятся ко всему conditioning, '
              'а не к отдельным словам. Здесь показаны запрошенные настройки, '
              'не измерения на GPU.</p>',
              '<details class="tcs-lab-section" open><summary>① Слои и фазы · активный режим</summary>',
              _phase_view(phases, depth_l=depth_l, blocks_g=blocks_g), '</details>',
              '<details class="tcs-lab-section" open><summary>② Токены, BPE и карта чанков</summary>',
              '<p class="tcs-lab-help">Данные от штатных токенизаторов A1111. '
              'Нажми на BPE-фрагмент или ячейку чанка. Автоматическая привязка '
              'исходных тегов к точным символам пока не реализована: без offset-aware '
              'токенизатора такое сопоставление было бы недостоверным.</p>']
    if not streams:
        pieces.append('<p>Введи промпт, затем нажми «Обновить CLIP Lab».</p>')
    for i, stream in enumerate(streams):
        pieces.append(_stream_view(stream, f's{i}', opened=(i == 0)))
    for label in sorted(set(s.label for s in streams)):
        left = next((s for s in streams if s.label == label and s.name == 'CLIP-L'), None)
        right = next((s for s in streams if s.label == label and s.name == 'OpenCLIP-G'), None)
        if left is not None and right is not None:
            a = [row.token_id for c in left.chunks for row in c.rows]
            b = [row.token_id for c in right.chunks for row in c.rows]
            same = sum(x == y for x, y in zip(a, b))
            pieces.append(f'<p class="tcs-lab-help">{h(label)} · сравнение токенизаторов: '
                          f'{same} совпадающих Token ID на показанных позициях; '
                          f'L={len(a)}, G={len(b)}. '
                          'Совпадающие Token ID не означают одинаковые CLIP-эмбеддинги.</p>')
    pieces.extend(['<div class="tcs-lab-detail" data-tcs-output="token" aria-live="polite">'
                   'Нажми на токен или позицию, чтобы увидеть ID, вес и выбранные слои.</div>',
                   '<p class="tcs-lab-help">BOS/EOS: служебные позиции; EOT-паддинг '
                   'не является пользовательским тегом. Цвета показывают только '
                   'веса токенизации, а не силу влияния токена на картинку.</p>',
                   '<details><summary>Как читать BPE, BOS/EOS, TI и pooled?</summary>'
                   '<p><b>BPE</b> — фрагмент словаря, а не обязательно целое слово или тег. '
                   '<b>BOS</b> и <b>EOS</b> — служебные позиции чанка. '
                   '<b>BREAK</b> начинает новый чанк; запятая после него не нужна. '
                   '<b>TI</b> может занимать несколько векторных позиций. '
                   '<b>Вес</b> — коэффициент обработки текста, не процент качества. '
                   '<b>Pooled G</b> — отдельный вектор SDXL; здесь он штатный. '
                   'Изменение выбранного CLIP-слоя не маршрутизирует отдельные слова '
                   'мимо других блоков Transformer.</p></details>',
                   '</details>'])
    if variant and variant_stream is not None and streams and prompt.strip():
        first = streams[0]
        a_ids = [r.token_id for c in first.chunks for r in c.rows]
        b_ids = [r.token_id for c in variant_stream.chunks for r in c.rows]
        mismatches = sum(1 for x, y in zip(a_ids, b_ids) if x != y) + abs(len(a_ids) - len(b_ids))
        pieces.extend(['<details class="tcs-lab-section" open><summary>③ Песочница A/B · повторная токенизация</summary>',
                       '<p>Вариант B обработан тем же штатным CLIP-L токенизатором. '
                       'Никакой ручной склейки существующих Token ID не происходит.</p>',
                       '<div class="tcs-lab-stats">',
                       f'<span>A: {first.chunks_total} чанк(ов), {first.occupied_total} позиций</span>',
                       f'<span>B: {variant_stream.chunks_total} чанк(ов), {variant_stream.occupied_total} позиций</span>',
                       f'<span>Отличающихся ID в показанных чанках: {mismatches}</span>',
                       '</div><p class="tcs-lab-help">Сравнение по позиции, не semantic diff. '
                       'Отличия в длине и BREAK могут сдвигать все последующие позиции.</p></details>'])
    pieces.extend(['<p class="tcs-lab-footer">Для точной диагностики эмбеддингов '
                   'нужен будущий Conditioning Inspector. Layer Map не доказывает '
                   'native-equivalence или исполнение на GPU; для этого используй Phase Trace.', '</p></div>'])
    return ''.join(pieces)
