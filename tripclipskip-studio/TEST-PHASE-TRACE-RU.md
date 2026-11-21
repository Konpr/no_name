> **Версия 0.3.2:** исходные проверки 0.3.1 остаются актуальными. Дополнительно проверь `Independent OpenCLIP-G = off` при намеренно разных значениях Phase2 L/G: итоговое `TCS Phases L/G` должно показывать для второй фазы одинаковые L/G. При `Multi-Phase count = 3` лог должен содержать только три счётчика фаз. Подробнее: `AUDIT-0.3.2-RU.md`.

# Как проверить TripClipSkip Studio 0.3.1 на GPU

**Основная цель:** понять, почему в прошлом запуске `per-phase calls=[6,3,9,0]`, и убедиться, что PNG Info показывает реальные, а не только запрошенные фазы.

## Параметры первого теста

- `txt2img`, SDXL / Illustrious checkpoint, один фиксированный seed, один prompt.
- Euler, 20 Steps, CFG 7, Batch size/count 1, 768×768; без Hires, REGION, Refiner, ControlNet.
- TripClipSkip Studio: включено; CLIP-L 1.0, CLIP-G 1.0, `lerp`, reference `a1111`.
- Timeline: **Multi-Phase count = 4**; **Dual-Phase checkbox можно не включать**.
- Phase 2 (свои новые слайдеры): L 1.5 / G 1.0, start 0.25.
- Phase 3: L 2.0 / G 1.5, start 0.50.
- Phase 4: L 2.5 / G 2.0, start 0.75.
- `Phase progress source = denoiser` (сохраняет прежнее поведение).

Нажми **Generate** и найди в терминале:

```
[TripClipSkip Studio] STEP timeline counters: ... per-phase calls=[...]
[TripClipSkip Studio] Phase Trace: source=denoiser, observed_calls=..., last=.../..., transitions=P1@...; P2@...; ...
```

В **PNG Info** проверь:

```
TCS Phase count: 4                    # запрошено
TCS Phase effective: enabled: 4 requested
TCS Phase progress source: denoiser
TCS Observed phase calls: 5,5,5,5    # пример, не гарантированный результат
TCS Observed transitions: P1@...;P2@...;P3@...;P4@...
TCS Phase trace warning: none
```

Если Phase 4 имеет ноль, скопируй `Phase Trace` вместе с `TCS Phase boundaries` из PNG Info. Затем попробуй **тот же** тест с `Phase progress source = sampler`, но **не называй результат точным эквивалентом предыдущего — сменился способ выбора фазы**.

## Ещё три проверки

1. **Старый Dual-Phase:** `Multi-Phase count=0`, включить Two-phase, настроить старую Phase 2 1→3 с switch 0.5. Должно по-прежнему переключаться.
2. **REGION:** вернуться к `count=4`, использовать свой рабочий REGION prompt. Проверить `TCS Phase effective: disabled: REGION (phase 1 only)`; региональное изображение должно строиться. Multi-Phase с REGION в этой версии намеренно не выполняется.
3. **Hires Fix:** при multi=4 и Hires=on проверить `Hires calls=...` и отсутствие использования поздних фаз во втором проходе.

## Что отправить для расследования

- Одну строку `STEP timeline counters` и строку `Phase Trace` (плюс warnings, если есть).
- PNG Info с `Phase boundaries`, `Phase effective`, `Observed phase calls`.
- Название самплера и Steps, было ли Interrupt/Refiner/Hires.

**Важно:** наличие 4 записанных параметров в PNG Info само по себе не означает, что четыре фазы использовались; для этого нужны observed counts/transitions.
