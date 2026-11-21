# TripClipSkip Studio — текущая сборка

**Актуальный runtime: WP-03B1c-a-r1 (21 ноября 2026, CPU preview).** Поставка документации `WP-03B1c-a-r1-docsync` синхронизирует README, план, handoff и machine state, **не меняя runtime**. Проверено ранее: **758 passed** с исходниками A1111 dev + custom (2 прежних `SyntaxWarning`); **736 passed / 22 skipped** без внешних исходников. **GPU-приёмка B1c-a-r1 и B1b-r2 — PENDING.** [Аудит r1](development/WP03B1C_A_R1_INDEPENDENT_REAUDIT_RU.md) · [Состояние](development/CURRENT_STATE.json) · [Документационная синхронизация](development/WP03B1C_A_R1_DOCSYNC_RU.md).

## Последние изменения (read-only Native Proof)

- **WP-03B1c-a-r1 (актуально):** исправлена ошибочная наследуемая связь `token_chunk_ref` / `token_forward_ref` между разными экземплярами CLIP-L и OpenCLIP-G при вложенных вызовах. Контекст привязывается к точному wrapper; native conditioning и pooled-G не меняются. [Детали и два adversarial теста](development/WP03B1C_A_R1_INDEPENDENT_REAUDIT_RU.md).
- **WP-03B1c-a (база):** наблюдение реальных A1111 `process_tokens()` чанков и TI metadata; job-local HMAC входных token ID без сохранения текста/сырых ID; связка chunk → native encoder invocation → layers; pooled object identity первого чанка; границы `get_conds_with_caching()` и *лексические признаки* `ASSEMBLE`, `BLEND`, `MORPH`, `POOL`, `CHUNK`, `BIND`, `REGION`. **Лексические признаки не доказывают выполнение оператора или происхождение итогового token/pooled после композиции.** [Технический отчёт](development/WP03B1C_READ_ONLY_TOKEN_CHUNK_POOLED_RU.md).
- **WP-03B1b-r2 (наследуется):** более полный read-only захват Hires через `calculate_hr_conds()`, точную cache-slot identity, раздельные/reused denoiser instances и ограниченный sampling; `source_guard` запрещает использовать `NONFINITE` как источник для injection и **не** объявляет конечные наблюдения доверенными. GPU именно r2 пока не проверялся. [Отчёт](development/WP03B1B_R2_GPU_FINDINGS_HARDENING_RU.md).

**Уровень доказательств:** SHA-256 OFF/ON и три одинаковых PNG относятся к **исторической B1b-r1**, не к текущему runtime. Отдельной доказанной привязки каждого PNG к JSON нет. Для B1c-a-r1 нет GPU-проверки, а окончательный post-operator lineage и Layer Composer injection пока не реализованы. Новых Gradio-контролов нет (по-прежнему 28 существующих параметров).

**Исторический повторный аудит WP-03B1b-r1:** исправлена неверная метка `construction_pass=base` при раннем расчёте Hires.fix. Для определения base/hires учитывается идентичность A1111 cache-slot; неизвестные варианты получают `unknown`. Историческая r1-регрессия: 728 passed с обоими исходными деревьями, 712 passed/16 skipped без них. Новая r2: 742 passed с обоими и 724 passed/18 skipped без них. Генерационный путь без изменений; на момент аудита GPU-проверки не было. Позднее получены SHA-256 и PNG для r1, но r2 ещё не проверялся на GPU. [Отчёт](development/WP03B1B_R1_REAUDIT_RU.md).

## Что реально работает

- SD1.x / SDXL (поддерживаемые A1111-обёртки): относительная и дробная глубина CLIP-L/G (`lerp`, `slerp`), фазы и Absolute Timeline; отдельные режимы имеют ограничения по модели.
- Layer Explorer с проверкой применения каждой ячейки; в **r2** пользователь проверил SDXL сетку L=1/G=1 и L=1.5/G=1, и чистый положительный prompt в PNG Info.
- Native Proof (`TCS_PROOF_MODE=shapes|sha256`, `TCS_PROOF_LAYERS=1`) — наблюдение блоков, форм, SHA и численных аномалий без записи сырых тензоров. На SDXL CUDA проверялась **предшествующая B1b-r1** (12/12 CLIP-L, 32/32 OpenCLIP-G, native pooled-G, сопоставленные SHA); **B1b-r2 и B1c-a-r1 ещё GPU pending**.
- 28 существующих Gradio параметров UI; вкладки CLIP/фазы/диагностика/карты/токены и CLIP Lab. Новые разделы в дорожной карте **пока не реализованы**.

## Что добавлено в WP-03B1a (только диагностика)

В опциональном Native Proof (`TCS_PROOF_MODE=shapes|sha256`, `TCS_PROOF_LAYERS=1`)
добавлен структурный descriptor обнаруженных L/G, наличие сторонних hooks до установки
временных, declared layer config, наблюдаемый attached pooled и связь двух видов
энкодерных событий одним `invocation_id`. Фактический positive/negative и BPE/chunks **не
доказаны**, meta-digest **не является** fingerprint весов. Никаких новых Gradio-контролов,
обновлений процесса генерации или pooled mixing. [Подробности](development/WP03B1A_READ_ONLY_DIAGNOSTICS_RU.md).

## Новое в WP-03B1b: Scoped Branch Provenance

Только в opt-in Native Proof (`TCS_PROOF_LAYERS=1`) привязка CLIP invocation к **контексту** построения positive/negative через фактический вызов A1111 `setup_conds → get_conds_with_caching` (исторический B1b-r1 путь; B1b-r2 дополняет его Hires `calculate_hr_conds`). Дополнительные encoder calls и pooled, собранные кастомным parser, не считаются proven final branch. В JSON — `construction_branch`, `construction_scope_id`; старый `branch=unknown` для `encoder_l/g` остаётся. Смотри [технический отчёт](development/WP03B1B_SCOPED_BRANCH_PROVENANCE_RU.md). GPU именно для B1b-r2+B1c-a-r1 не выполнялся.

## Что изменено в r3-r2

1. **G `last`:** нормализация дробных конечных точек выполняется функционально без повторных вызовов `ln_final` и сторонних forward-hooks; принимаются узко известные нативные классы с проверкой точного якоря. Изменённые или неизвестные реализации отклоняются.
2. **L `last`:** нормализация применяется к обоим endpoint-слоям, когда опора — финально нормализованное состояние. Нативный `L=1.0` остаётся прямым вызовом обёртки.
3. Планы, документы и архивные отчёты упорядочены. **Не добавлены новые Gradio контролы, новый pooled-вектор или новая генерационная математика для penultimate.**

## Что ещё НЕ доказано

- Полная эквивалентность конечных PNG, поддержка всех LoRA/TI/чужих обёрток, SD1.x GPU, SDXL Refiner и других видов OpenCLIP.
- Новый путь дробного `L last` / `G last` на реальном GPU: **PENDING**. На неподдерживаемом классе/нормализаторе TCS отказывает вместо молчаливого изменения conditioning.
- Полный trusted final positive/negative и token/chunk provenance для всех encoder invocations, в особенности после `ASSEMBLE/POOL/REGION`; полноценный Pooled Composer и WP-03B2 generation injection.

## Навигация

- [План реализации](PLAN-NEXT-RU.md) · [Дорожная карта](ROADMAP.md)
- [Актуальный handoff разработчику](development/AI_HANDOFF_READ_FIRST_RU.md) · [Машиночитаемый статус](development/CURRENT_STATE.json)
- [Аудит и критерии тестирования r3-r2](development/WP03B0_2_R3_R2_READ_FIRST_RU.md)
- [Архивная документация](docs/history/ARCHIVE_INDEX_RU.md) · [История версий](CHANGELOG.md)

**Безопасная рекомендация:** сохранять B1b-r1 GPU-данные как исторический контроль, текущую **WP-03B1c-a-r1** считать только CPU-проверенной и не использовать её read-only provenance как разрешение для injection. Перед WP-03B2 выполнить объединённую GPU-приёмку B1b-r2 + B1c-a-r1; отдельную проверку `last`-нормализации r3-r2 не считать закрытой.
