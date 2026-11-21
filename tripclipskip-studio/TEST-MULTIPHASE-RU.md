# TripClipSkip Studio 0.3.0-preview — тест Multi-Phase для A1111

Статус: **CPU/mock-тесты пройдены. Графическая генерация на GPU с Multi-Phase не проверена.**
Поддерживаемые модели: SD1.x и SDXL; другие архитектуры не включены.

## Установка

1. Сохранить прежний ZIP `tripclipskip-studio-0.2.1-audit.zip` для отката.
2. Полностью выключить старое расширение TripClipSkip (если установлено).
3. Удалить прежнюю папку `extensions/tripclipskip-studio/` (пресеты при необходимости сохранить отдельно).
4. Распаковать `tripclipskip-studio-0.3.0-multiphase-preview.zip` в `extensions/`; убедиться, что путь `extensions/tripclipskip-studio/scripts/tripclipskip.py` существует.
5. Полностью перезапустить WebUI. Свои файлы `modules` **не менять**.

## 4-фазный пример

В `txt2img` открой **TripClipSkip Studio → Timeline**. Включи расширение в верхней части панели.

- Открой секцию `Multi-Phase STEP · 3–4 phases (preview)`.
- `Multi-Phase count` = `4` (0 — исходный Dual-Phase).
- **Phase 1:** настрой в `CLIP Control` CLIP-L 1.0 / CLIP-G 1.0.
- **Phase 2:** значения в `Enable two-phase STEP switch` (эти элементы появляются при выборе Multi-Phase); CLIP-L 1.5 / CLIP-G 1.0; `Phase 2 start = 0.25`.
- **Phase 3:** CLIP-L 2.0 / CLIP-G 1.5; `Phase 3 start = 0.50`.
- **Phase 4:** CLIP-L 2.5 / CLIP-G 2.0; `Phase 4 start = 0.75`.
- Чтобы G имел независимые значения, оставь `Independent OpenCLIP-G` включённым на вкладке CLIP Control.

Выход: 0–25% — фаза 1; 25–50% — фаза 2; 50–75% — фаза 3; 75–100% — фаза 4 от фактического числа *вызовов denoiser*, а не от экранного номера шага.

## Проверка

1. Фиксированный seed, Euler, 20 steps, без Hires, LoRA/ControlNet, REGION. Сначала сравни baseline (`Multi-Phase count=0`, `Enable two-phase` выкл.) и 4-phase.
2. В терминале в конце должна появиться строка `STEP timeline counters ... per-phase calls=[...,...,...,...]`. Для четырёх фаз ожидаются ненулевые счётчики каждой фазы. Если нет — пришли терминальный лог.
3. В PNG Info проверь `TCS Phase count: 4`, `TCS Phase boundaries` и `TCS Phases L/G`.
4. Отдельно проверь `Multi-Phase count=3`: должны использоваться только фазы 1–3.
5. При включённом `REGION{...}` Multi-Phase автоматически **не переключает** фазы, оставляя обычный fractional CLIP и REGION. В PNG Info появляется предупреждение `TCS compatibility`.
6. При Hires Fix второй проход использует только значения фазы 1. При смене Hires checkpoint/Refiner STEP-режим не поддерживается.
7. Layer Explorer принудительно отключает как Dual, так и Multi-Phase *только для своих ячеек*.
8. Прерви одну генерацию и повтори baseline, чтобы проверить снятие хуков.

## Известные ограничения

- Multi-Phase 3–4 заранее пересчитывает conditioning нужное число раз; это может быть заметно медленнее при подготовке промпта. На **каждом шаге** повторного CLIP-кодирования нет.
- На кастомной сборке `modules` даже без REGION возможны дополнительные side effects при повторном `setup_conds`; нужна реальная проверка с LoRA, TI, TONEG и prompt scheduling.
- REGION + Multi-Phase отключён защитой. Независимая временная шкала Hires и плавные кривые пока не реализованы.
- Для условий фаз с несовпадающей структурой расширение прекращает подготовку и пишет ошибку вместо подстановки несовместимых тензоров.
- Нет подтверждения на GPU, что каждый sampler сообщает корректные `step` и `total_steps`.

При проблеме верни 0.2.1 из сохранённого ZIP. Никакие внешние `modules` обновлять не требуется.
