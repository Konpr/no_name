"""Read-only Token / Chunk Inspector for A1111 native text-encoder wrappers.

Uses the actual `tokenize_line` method of the *loaded* CLIP-L and OpenCLIP-G
wrappers, including A1111 emphasis, BREAK, comma backtracking and TI markers.
Does not encode text, access the UNet, install hooks, change options or write
conditioning caches. Input is manually supplied; prompt editing, wildcards,
REGION/TONEG rewrites and runtime prompt scheduling are NOT resolved here.
"""
from __future__ import annotations

import math
from typing import Any

from studio_core.compose.tokenizer_map import (
    TokenSnapshotError, snapshot_from_native_chunks, summarize_native_token_map,
)
from studio_core.compose.token_trace import trace_native_segments, render_span_trace
from studio_core.compose.token_groups import select_verified_group, render_group_selection
from studio_core.compose.smart_break import preview_break_layout, render_break_preview
from studio_core.compose.readable_preview import readable_card

from studio_core.token_laboratory import (
    has_break, strip_break_for_comparison, overview_lines,
    compare_encoders, compare_prompt_ids, unicode_notes, ru_chunks,
)

MAX_PROMPT_CHARS = 12000
MAX_RENDER_CHUNKS = 12
MAX_ROWS_PER_STREAM = 110


def _text(value: Any) -> str:
    """Protect Markdown table cells, inline code and control characters."""
    raw = str(value).replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')
    return raw.replace('|', '\\|').replace('`', 'ˋ')


def _piece(token_id: int, *, tokenizer=None, decoder=None) -> str:
    if tokenizer is not None:
        fn = getattr(tokenizer, 'convert_ids_to_tokens', None)
        if callable(fn):
            item = fn(int(token_id))
            if isinstance(item, (tuple, list)):
                item = item[0] if item else '?'
            return _text(item)
    if decoder is not None:
        # The OpenCLIP tokenizer used by native A1111 keeps `decoder: id->bpe`.
        return _text(decoder.get(int(token_id), 'неизвестно'))
    return 'ID (декодирование недоступно)'


def _ti_slots(chunk) -> dict[int, str]:
    """Map zero-based content offsets to textual-inversion vector labels."""
    result = {}
    for fix in getattr(chunk, 'fixes', ()):
        offset = int(fix.offset)
        embedding = fix.embedding
        vectors = int(getattr(embedding, 'vectors', 0))
        if vectors < 1 or vectors > 4096:
            raise ValueError('Некорректная длина Textual Inversion')
        for n in range(vectors):
            result[offset + n] = f'TI:{_text(getattr(embedding, "name", "?"))} [{n + 1}/{vectors}]'
    return result


def _inspect_one(name: str, wrapper, prompt: str, *, tokenizer=None,
                 decoder=None, max_chunks=MAX_RENDER_CHUNKS,
                 max_rows=MAX_ROWS_PER_STREAM, collector=None,
                 show_details=True, branch="positive",
                 span_parser=None, emphasis_disabled=None, group_query="",
                 readable_sink=None) -> str:
    if not callable(getattr(wrapper, 'tokenize_line', None)):
        raise ValueError(f'{name}: у загруженного энкодера нет штатного tokenize_line()')
    chunks, token_count = wrapper.tokenize_line(prompt)
    if not chunks:
        raise ValueError(f'{name}: A1111 вернул пустой набор чанков')
    chunk_length = int(getattr(wrapper, 'chunk_length', 75))
    if chunk_length < 1 or chunk_length > 2048:
        raise ValueError(f'{name}: неожиданная длина чанка {chunk_length}')
    id_start = wrapper.id_start
    id_end = wrapper.id_end
    capacity = chunk_length + 2
    summary = []
    content_counts = []
    weights_per_chunk = []
    ti_positions = []
    content_id_list = []
    rows = []
    displayed = 0
    for index, chunk in enumerate(chunks, start=1):
        ids = list(chunk.tokens)
        multipliers = list(chunk.multipliers)
        if len(ids) != capacity or len(multipliers) != capacity:
            raise ValueError(f'{name}: чанк {index} не имеет {capacity} позиций')
        if ids[0] != id_start or ids[-1] != id_end:
            raise ValueError(f'{name}: чанк {index} без ожидаемых BOS/EOS')
        # A1111 pads with EOT IDs, but a user can also supply an actual EOT
        # token inside their prompt. Therefore the FIRST EOT is NOT necessarily
        # padding: tokens can legitimately follow it. Keep all positions up to
        # the last non-EOT, including any earlier EOT tokens.
        content_ids = ids[1:-1]
        last_non_eot = max((pos for pos, token_id in enumerate(content_ids)
                            if token_id != id_end), default=-1)
        occupied = last_non_eot + 1
        if index == len(chunks):
            # The native token_count accounts for earlier chunks in units of
            # chunk_length (including BREAK padding), plus the exact unpadded
            # length of the last chunk. This also disambiguates a literal EOT
            # that is itself the final user token, indistinguishable from pad
            # tokens by looking only at the IDs. Do not use it for prior chunks.
            last_len = token_count - chunk_length * (len(chunks) - 1)
            if 0 <= last_len <= chunk_length:
                occupied = max(occupied, last_len)
        ti = _ti_slots(chunk)
        if any(offset < 0 or offset >= occupied for offset in ti):
            raise ValueError(f'{name}: Textual Inversion выходит за позиции чанка {index}')
        content_counts.append(occupied)
        ti_positions.append(set(ti))
        content_id_list.extend(int(t) for t in content_ids[:occupied])
        relevant_weights = [float(x) for x in multipliers[1:occupied+1]]
        if any(not math.isfinite(w) for w in relevant_weights):
            raise ValueError(f'{name}: нечисловой вес токена')
        weights_per_chunk.append(relevant_weights)
        summary.append(f'{index}: {occupied}/{chunk_length}' + (f', TI={len(getattr(chunk, "fixes", []))}' if getattr(chunk, 'fixes', ()) else ''))
        if index > max_chunks:
            continue
        if not show_details:
            continue
        for pos in range(occupied):
            if displayed >= max_rows:
                break
            token_id = int(content_ids[pos])
            token_piece = ti.get(pos)
            if token_piece is None:
                token_piece = _piece(token_id, tokenizer=tokenizer, decoder=decoder)
            weight = float(multipliers[pos + 1])
            if not math.isfinite(weight):
                raise ValueError(f'{name}: нечисловой вес токена')
            rows.append(f'| {index} | {pos + 1} | `{token_id}` | `{token_piece}` | {weight:.4g} |')
            displayed += 1
    total_occupied = sum(content_counts)
    if collector is not None:
        collector.update({
            'name': name, 'chunks': len(chunks), 'occupied': sum(content_counts),
            'lengths': content_counts, 'token_count': token_count,
            'ids': content_id_list, 'id_end': int(id_end), 'capacity': capacity,
        })
    result = [f'#### {name}',
              f'**Чанков:** {len(chunks)} · **Позиции на чанк:** {capacity} ({chunk_length} пользовательских + BOS/EOS) · '
              f'**Занято содержимым:** {total_occupied} · **Счётчик A1111:** {token_count}',
              '**Заполнение по чанкам:** ' + '; '.join(summary[:max_chunks]) + ('; …' if len(summary) > max_chunks else ''),
              '']
    result.extend(overview_lines(
        name=name, lengths=content_counts, capacity=chunk_length,
        ti_offsets=ti_positions, token_count=token_count,
        weights=weights_per_chunk, prompt=prompt,
    ))
    # WP-02A: derive an immutable structural map from ALREADY tokenized native
    # chunks. No additional tokenizer/CLIP forward and no guessed text offsets.
    # A broken/nonstandard wrapper must not produce an apparently exact map;
    # keep the older read-only inspector operational with a clear diagnostic.
    try:
        native_map = snapshot_from_native_chunks(
            chunks=chunks, token_count=token_count, wrapper=wrapper,
            source_id=branch + ('-g' if name == 'OpenCLIP-G' else '-l'),
            branch=branch, encoder='g' if name == 'OpenCLIP-G' else 'l',
            prompt=prompt,
        )
    except TokenSnapshotError as exc:
        if readable_sink is not None:
            readable_sink.append(f'### {name}\n\nНативная карта не подтверждена: ' + _text(exc))
        result.append('\n**WP-02A:** Невозможно подтвердить нативную карту: '
                      + _text(exc) + '. Исходные токены не изменены.')
    else:
        result.append('\n' + summarize_native_token_map(native_map))
        # WP-02B derives parser segment offsets separately; the already captured
        # native chunks are the oracle. No extra tokenize_line or CLIP forward.
        # If A1111's parser is unavailable, do not invent offsets.
        if emphasis_disabled is None or span_parser is None:
            try:
                from modules import prompt_parser as a1111_parser, shared as a1111_shared
                if emphasis_disabled is None:
                    emphasis_disabled = getattr(a1111_shared.opts, 'emphasis', None) == 'None'
                if span_parser is None:
                    span_parser = a1111_parser.parse_prompt_attention
            except (ImportError, AttributeError):
                pass
        span_trace = trace_native_segments(
            prompt=prompt, snapshot=native_map, wrapper=wrapper, native_chunks=chunks,
            parse_attention=span_parser, emphasis_disabled=bool(emphasis_disabled),
        )
        result.append('\n' + render_span_trace(span_trace))
        # WP-02C analyzes verified WHOLE source segments and the occupied
        # native chunks. User group text is never fed into generation.
        selection = select_verified_group(prompt=prompt, query=group_query,
                                          trace=span_trace)
        result.append('\n' + render_group_selection(selection,
                            query_supplied=bool(group_query)))
        break_plan = preview_break_layout(native_map=native_map, trace=span_trace)
        result.append('\n' + render_break_preview(break_plan))
        if readable_sink is not None:
            readable_sink.append(readable_card(
                name=name, branch=branch, selection=selection, query=group_query,
                plan=break_plan, occupied_per_chunk=tuple(content_counts)))
    if show_details:
        result.extend(['', '| Чанк | Позиция | Token ID | BPE / Textual Inversion | Вес |',
                       '|---:|---:|---:|:---|---:|'])
        if rows:
            result.extend(rows)
        else:
            result.append('| — | — | — | *(пустой промпт)* | — |')
        if total_occupied > displayed:
            result.append(f'\n*Для удобства показаны первые {displayed} из {total_occupied} занятых позиций. '
                          'Число чанков и подсчёт A1111 приведены для полного введённого текста.*')
        if len(chunks) > max_chunks:
            result.append(f'*Отображение ограничено первыми {max_chunks} чанками из {len(chunks)}.*')
    return '\n'.join(result)


def render_token_inspector(*, mode: str, clip_l, clip_g=None,
                           prompt: str, negative: str = '',
                           openclip_decoder=None,
                           max_prompt_chars=MAX_PROMPT_CHARS,
                           show_details=True, compare_without_break=False,
                           group_query: str = "", split_output: bool = False) -> str | tuple[str, str]:
    """Format A1111 native tokenizer output; never call encode/forward.

    `clip_l`/`clip_g` are *A1111 wrappers*, not raw HF/open_clip models.
    `openclip_decoder` is the decoder dict from modules.sd_hijack_open_clip.tokenizer.
    """
    if mode not in ('sd1', 'sdxl'):
        raise ValueError('Инспектор поддерживает только SD1.x и SDXL')
    if mode == 'sdxl' and clip_g is None:
        raise ValueError('Для SDXL не найден штатный OpenCLIP-G')
    if not isinstance(prompt, str) or not isinstance(negative, str):
        raise ValueError('Промпты должны быть строками')
    if not isinstance(group_query, str) or len(group_query) > 256:
        raise ValueError("Группа для проверки должна быть строкой не длиннее 256 символов")
    if len(prompt) > max_prompt_chars or len(negative) > max_prompt_chars:
        raise ValueError(f'Текст слишком длинный для просмотра (не более {max_prompt_chars} символов в каждом поле)')

    sources = [('Положительный промпт', prompt)]
    if negative.strip():
        sources.append(('Негативный промпт', negative))
    result = ['### Инспектор токенов и чанков A1111',
              f'**Модель:** {"SDXL" if mode == "sdxl" else "SD1.x"} · '
              '**Режим:** только просмотр · **Источник:** `tokenize_line()` загруженных энкодеров A1111',
              '', '**Важное ограничение:** введи копию текста вручную. Это токенизация '
              '*введённой строки*, не результат подстановок `wildcards`, '
              '`prompt editing`, `REGION/TONEG` и других преобразований при генерации. '
              'Количество BPE-токенов не равно числу Danbooru-тегов. '
              'Модель CLIP не вызывается; conditioning и настройки не меняются.',
              '', '### Краткий путеводитель',
              '**Token ID** — номер элемента словаря BPE, а не числовая оценка слова. '
              '**Чанк** — отдельный проход через текстовый Transformer на 77 позиций '
              '(75 + BOS/EOS). **BREAK** — начать следующий чанк принудительно; '
              '**паддинг** — заполнение оставшихся позиций. **Вес** — коэффициент '
              'A1111 до последующей обработки conditioning. **TI** — обученные '
              'векторы Textual Inversion вместо обычных BPE-токенов.',
              '**Важно:** одинаковые ID в CLIP-L и CLIP-G не означают одинаковые '
              'скрытые представления; разделение по чанкам меняет доступный '
              'контекст self-attention, но не гарантирует улучшения изображения.']
    collected = {}
    readable_sections = []
    for label, text in sources:
        result.append(f'\n### {label}\n')
        l_stats = {}
        result.append(_inspect_one('CLIP-L', clip_l, text,
                                   tokenizer=getattr(clip_l, 'tokenizer', None),
                                   collector=l_stats, show_details=show_details,
                                   branch='negative' if label.startswith('Негативный') else 'positive',
                                   group_query=group_query if label == 'Положительный промпт' else '',
                                   readable_sink=readable_sections if split_output else None))
        collected[(label, 'CLIP-L')] = l_stats
        if mode == 'sdxl':
            result.append('\n')
            g_stats = {}
            result.append(_inspect_one('OpenCLIP-G', clip_g, text,
                                       decoder=openclip_decoder,
                                       collector=g_stats, show_details=show_details,
                                       branch='negative' if label.startswith('Негативный') else 'positive',
                                       group_query=group_query if label == 'Положительный промпт' else '',
                                       readable_sink=readable_sections if split_output else None))
            collected[(label, 'OpenCLIP-G')] = g_stats
            result.append('\n' + compare_encoders(l_stats, g_stats, title=label))
        u_notes = unicode_notes(text)
        if u_notes:
            result.append('\n' + '\n\n'.join(u_notes))
        if compare_without_break and has_break(text):
            changed = strip_break_for_comparison(text)
            result.append('\n#### Эксперимент: что будет без BREAK?')
            result.append('Сравнение рассчитано повторным вызовом штатного `tokenize_line()` '
                          'для **изменённой копии** текста; исходный промпт НЕ изменён. '
                          'Символы и знаки препинания, кроме удалённого слова `BREAK`, сохраняются.')
            for name, wrapper in [('CLIP-L', clip_l)] + ([('OpenCLIP-G', clip_g)] if mode == 'sdxl' else []):
                counterfactual = {}
                _inspect_one(name, wrapper, changed,
                             tokenizer=getattr(wrapper, 'tokenizer', None) if name == 'CLIP-L' else None,
                             decoder=openclip_decoder if name == 'OpenCLIP-G' else None,
                             collector=counterfactual, show_details=False)
                original = collected[(label, name)]
                result.append(f'**{name}:** с `BREAK` — {ru_chunks(original["chunks"])} '
                              f'({original["chunks"]*original["capacity"]} входных позиций, '
                              f'счётчик {original["token_count"]}); '
                              f'без `BREAK` — {ru_chunks(counterfactual["chunks"])} '
                              f'({counterfactual["chunks"]*counterfactual["capacity"]} позиций, '
                              f'счётчик {counterfactual["token_count"]}). '
                              'Это сравнение нагрузки/разбиения, **не прогноз качества**.')
    if negative.strip():
        result.append('\n### Совпадения Token ID между positive и negative')
        result.append(compare_prompt_ids(collected[('Положительный промпт', 'CLIP-L')],
                                         collected[('Негативный промпт', 'CLIP-L')],
                                         encoder='CLIP-L'))
        if mode == 'sdxl':
            result.append(compare_prompt_ids(collected[('Положительный промпт', 'OpenCLIP-G')],
                                             collected[('Негативный промпт', 'OpenCLIP-G')],
                                             encoder='OpenCLIP-G'))
    result.append('\n**Примечания:** `BREAK` может принудительно начать следующий чанк и '
                  'изменить счётчик A1111. Нули в Token ID с меткой `TI` — слоты '
                  'векторов Textual Inversion, а не обычные BPE-токены. '
                  'Паддинг и служебные BOS/EOS не перечислены построчно; '
                  'явный EOS внутри промпта отображается как пользовательский токен. '
                  'Если EOT оказался последним пользовательским токеном промежуточного чанка, '
                  'его нельзя гарантированно отличить от паддинга лишь по Token ID. '
                  'Для негативного промпта используй второе поле; '
                  'эмбеддинги/веса здесь только анализируются, не вычисляются.')
    summary = ['### Сводка по введённым текстам']
    for label, _ in sources:
        for encoder in (['CLIP-L', 'OpenCLIP-G'] if mode == 'sdxl' else ['CLIP-L']):
            data = collected[(label, encoder)]
            summary.append(
                f'**{label} · {encoder}:** {ru_chunks(data["chunks"])} · '
                f'{data["occupied"]} содержательных позиций · '
                f'{data["chunks"] * data["capacity"]} входных позиций CLIP · '
                f'счётчик A1111 {data["token_count"]}.'
            )
    technical = '\n\n'.join(result[:4] + summary + result[4:])
    if not split_output:
        return technical
    readable = ('## Результат проверки — понятным языком\n\n'
                '**Только просмотр:** промпт введён вручную; генерация и conditioning '
                '**не изменены**.\n\n'
                + ('\n\n---\n\n'.join(readable_sections)
                   if readable_sections else 'Нативное сопоставление недоступно.')
                + '\n\n*Точные Token ID, веса, технические таблицы WP-02A/B/C и '
                'подробные оговорки находятся ниже, в раскрываемом блоке.*')
    return readable, technical
