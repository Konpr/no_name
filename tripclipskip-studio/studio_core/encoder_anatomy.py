"""Read-only learning aid for actual native CLIP-L/OpenCLIP-G wrapper layout.

Accept only validated metadata already obtained by A1111 model inspection.
Never executes either text encoder or touches tensor/attention hooks.
"""
from __future__ import annotations


def render_encoder_anatomy(*, family: str, depth_l: int, blocks_g: int | None,
                           g_anchor: str | None, native_reference: str):
    if family not in ('sd1', 'sdxl'):
        raise ValueError('Поддерживается SD1.x и SDXL')
    if not 1 <= depth_l <= 128:
        raise ValueError('Неизвестная глубина CLIP-L')
    if family == 'sdxl' and (blocks_g is None or blocks_g < 2 or g_anchor not in ('penultimate', 'last')):
        raise ValueError('Неизвестная структура CLIP-G')
    lines = [
        '### Как реально работает текстовый энкодер',
        '',
        '**Промпт → BPE/позиции (77) → текстовый Transformer → скрытые состояния → conditioning UNet.**',
        '',
        f'**Обнаружена модель:** {"SDXL" if family == "sdxl" else "SD1.x"}; '
        f'**CLIP-L:** {depth_l} блоков.',
        '',
        '#### CLIP-L: индексы `hidden_states`',
        '',
        '| Индекс HF | Что внутри |',
        '|---|---|',
        f'| `hidden_states[-1]` | сырой выход последнего ({depth_l}-го) блока |',
    ]
    if depth_l >= 2:
        lines.append(f'| `hidden_states[-2]` | сырой выход блока {depth_l - 1} |')
    lines.extend([
        f'| `hidden_states[-{depth_l+1}]` = `[0]` | **входные эмбеддинги** до первого Transformer-блока |',
        '',
        '`last_hidden_state` может включать финальную LayerNorm и **не обязан** '
        'равняться сырому `hidden_states[-1]`. '
        'Абсолютный экспериментальный режим использует именно *raw hidden_states*, '
        'не добавляя новую LayerNorm.',
        '',
        f'**Опора CLIP-L штатного режима:** `{str(native_reference).replace("`", "")[:140]}`.',
    ])
    if family == 'sdxl':
        last_g = blocks_g - (g_anchor == 'penultimate')
        lines.extend([
            '',
            '#### OpenCLIP-G: отдельные блоки и pooled',
            '',
            f'**Transformer-блоков:** {blocks_g}; '
            f'**штатная опора:** `{g_anchor}` → блок **{last_g}/{blocks_g}**.',
            f'Абсолютный режим может выбирать блоки **1…{last_g}**. '
            'Диапазон выше — **ограничение нашего экспериментального редактора**, '
            'а не полный список исполняемых блоков: SDXL `FrozenOpenCLIPEmbedder2` '
            'может выполнять последний блок для pooled, даже когда token-wise '
            'использует `penultimate`.',
            '',
            '**Token-wise G** передаёт токеновое представление в cross-attention. '
            '**Pooled G** — отдельный глобальный сигнал SDXL; в TripClipSkip Studio '
            'он сохраняется с нативного пути и **не переключается** вместе с G-блоками.',
            '',
            'Обычно в SDXL токеновые ветви **CLIP-L (768)** и **OpenCLIP-G (1280)** '
            'объединяются до **2048 каналов**; это типичная архитектура SDXL, '
            'а не измерение выходов именно этого checkpoint.',
        ])
    lines.extend([
        '',
        '#### Что означают эксперименты',
        '',
        '- **Relative Skip** выбирает слой относительно опоры A1111/model/custom.',
        '- **Absolute Timeline** выбирает конкретный *сырой* HF L-слой и выполненный блок G; '
        'внутри использует заранее подготовленные conditioning и проверенный фазовый планировщик.',
        '- **Token Inspector** показывает BPE/чанки, но не измеряет внимание UNet и не доказывает, '
        'что тег отвечает только за конкретный объект.',
        '- Выбор более ранних слоёв или смена pooled **не гарантирует** улучшения деталей; '
        'сравнивай изображения и параметры на одинаковом seed.',
        '',
        '*Только просмотр: нет вызовов CLIP, загрузки весов, изменения shared.opts '
        'или установки хуков. GPU native-equivalence ещё требует проверки.*',
    ])
    return '\n'.join(lines)
