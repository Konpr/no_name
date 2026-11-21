# TripClipSkip Studio — актуальный план реализации

**Срез 21.11.2026. Актуальный runtime: WP-03B1c-a-r1 (CPU preview: 758 passed с dev+custom; 736 passed/22 skipped без них). Документационная синхронизация не меняет runtime. Предыдущая B1b-r1 имеет исторические GPU SHA/PNG; B1b-r2 и B1c-a-r1 ожидают совместной GPU-приёмки. Исторические планы не являются текущими задачами.** [Текущий README](README-RU.md) · [Machine state](development/CURRENT_STATE.json).

## P0 — завершить r3-r2 Normalization & Hook Safety

- [x] Исправить повторные вызовы `G last` нормализатора и реакцию на сторонние хуки, не вызывая `norm.forward` вне штатного encoder forward.
- [x] Проверять узко распознанные `torch LayerNorm/Identity`, OpenCLIP `LayerNorm/LayerNormFp32`, и точную идентичность native anchor; неизвестные классы отклонять.
- [x] Привести оба endpoint'а `L last` к одной политике нормализации и сохранить `L=1` native passthrough.
- [x] CPU unit/adversarial tests, полные regressions, чистая распаковка ZIP, SHA256 манифест.
- [ ] **Live SDXL GPU acceptance r3-r2:** один checkpoint/seed, TCS OFF, ON L=1/G=1; отдельно L=1.5/G=1, L=1/G=1.5; если модель использует `last`, отдельный контролируемый тест. Проверить готовые PNG, prepared/denoiser SHA и pooled; сохранять baseline r2.
- [ ] Совместимость с TI / TE-LoRA / сторонними hooks, `FP16/BF16`, длинными prompt, BREAK, многократными jobs, interrupt; **не заявлять до получения логов**.

## P1 — данные, происхождение и диагностика энкодеров

- [x] **WP-03B1a (частично):** opt-in read-only structural capability descriptor и native output shape/pooled attachment; link `encoder_l/g` ↔ `encoder_layers` через один `invocation_id`. Нет prompt/token ID raw, нового forward, новых UI-параметров. CPU mocks, не live GPU; **это не полноценный trusted provenance или encoder diagnostics UI**. [Отчёт](development/WP03B1A_READ_ONLY_DIAGNOSTICS_RU.md).

- [x] **WP-03B1b-r1 (частично):** read-only scoped **construction branch** на реальном пути `setup_conds → get_conds_with_caching`, exact callable + `SdConditioning` flags, связь через `construction_scope_id`. Фактическая итоговая ветка encoder/pooled и BPE/token/chunks **по-прежнему unknown**. CPU tests с custom A1111 AST и без него; **позднее проведён GPU smoke B1b-r1**, данные сохранены как исторические и не подтверждают B1b-r2. [Отчёт](development/WP03B1B_SCOPED_BRANCH_PROVENANCE_RU.md); [повторный аудит r1](development/WP03B1B_R1_REAUDIT_RU.md).

- [x] **WP-03B1b-r2 (только read-only CPU preview):** исправлена привязка позднего `calculate_hr_conds()` к Hires cache-slot и ошибочная диагностика `p.c/p.uc` как Hires; поддержаны разные/reused denoiser instances и per-pass sampling; metadata `source_guard` помечает все observed states как injection-ineligible. Предыдущие GPU SHA результаты r1 приложены только как evidence. [Отчёт](development/WP03B1B_R2_GPU_FINDINGS_HARDENING_RU.md).
- [ ] **Live GPU acceptance WP-03B1b-r2:** один Hires job + обычный job без рестарта, проверить `construction_pass=hires`, `prepared` из `hr_c/hr_uc`, denoiser bindings, limits, cleanup, и native OFF/ON parity.

- [x] **WP-03B1c-a (частичный, CPU preview):** opt-in per-A1111 `process_tokens` chunk metadata, job-local HMAC token ID input, TI fixes offsets/vectors, `token_chunk_ref`→`encoder invocation`→layer, pooled object identity первого чанка, return-contract `get_conds_with_caching`, lexical custom markers. **Не полная provenance после ASSEMBLE/POOL/REGION; refiner и TE-LoRA unverified.** [Отчёт](development/WP03B1C_READ_ONLY_TOKEN_CHUNK_POOLED_RU.md).
- [x] **WP-03B1c-a-r1 (CPU re-audit):** предотвращено перекрёстное наследование chunk/forward refs между CLIP-L/G при вложенных вызовах. 2 adversarial tests; 758 passed dev+custom, 736 passed / 22 skipped без sources. Полная GPU-приёмка ещё не выполнена. [Аудит](development/WP03B1C_A_R1_INDEPENDENT_REAUDIT_RU.md).
- [ ] **WP-03B1c (следующий read-only этап):** исследовать фактический source-to-result lineage для кастомных операторов; не повышать lexical hints до execution evidence, если нет прямого наблюдения.
- [ ] **B1b-r2 + B1c-a-r1 unified GPU gate:** Hires и subsequent ordinary без рестарта; OFF/ON native parity; long prompt и double chunk; pooled first chunk, TI и пользовательские операторы; контроль numeric NONFINITE; не начинать B2 injection до acceptance.

- [ ] **Capability fingerprint (полная задача; B1a даёт только структурную часть)** (без промптов и raw tensors): семейство L/G, версии/классы, `layer`, `layer_idx`, `legacy`, dtype, `ln_final`, зарегистрированные хуки, формат `[B,T,C]` или `[T,B,C]`, source of pooled. Сигнализировать неуверенность, не угадывать.
- [ ] **Native provenance WP-03B1**: связывать реальные encoder invocation, positive/negative, BPE/chunk/token IDs и LoRA/TI state; не выводить branch по порядку вызова.
- [ ] **Pooled/token identity matrix**: замерять token G penultimate, last, native pooled отдельно и проверять tensor/shape/digest при одинаковом seed.
- [ ] **Encoder Diagnostics UI**: безопасная read-only вкладка, explanation для отказов, статус normalizer/hooks/capabilities, копирование технического отчёта.
- [ ] **Layer Explorer 2.0**: визуальная карта выполненных слоёв и их `raw/normalized`, режимы `hidden/penultimate/last`, численные предупреждения, A/B изображения. Не подписывать ячейки без runtime confirmation.

## P2 — расширение реальных пользовательских возможностей

- [ ] Перепроектировать навигацию без десятков вкладок: 4 группы «Генерация», «Эксперименты», «Анализ CLIP», «Диагностика»; сохранить существующие 28 Gradio параметров и PNG Info round-trip миграцию.
- [ ] **Pooled-G Lab read-only**: показывать отличие native token conditioning vs native pooled; без манипуляций pooled на первом этапе.
- [ ] **Layer Composer integration WP-03B2** после trusted provenance + native OFF/ON identity + GPU A/B; opt-in, fail closed; не смешивать 768 L и 1280 G напрямую.

## P3 — исследовательские функции (не включать заранее)

- [ ] **Experimental Pooled Composer** с формально определёнными LayerNorm, EOT pool и text projection; детальная native parity с pooled и микрокондиционированием SDXL. Отдельный режим и GPU gate.
- [ ] Другие семейства OpenCLIP-L, SDXL Refiner, форки A1111, полнофункциональная LoRA/TI поддержка — только с отдельными модельными capability contracts.
- [ ] Больше UI ползунков и режимов интерполяции только после доказанной семантики и качественной A/B оценки.

## Запреты / регрессионные инварианты

1. **Не изменять классы библиотеки OpenCLIP или A1111 глобально**; использовать ограниченные scoped hooks, сохранять ровно один native wrapper forward и удалять временные хуки.
2. **Не путать `penultimate` и `pooled`**. `pooled-G` по умолчанию остаётся штатным; не трогать его при fractional token G.
3. При неизвестном layout/norm/backend — безопасный отказ, не тихое смешивание.
4. Не обещать качество изображений по SHA, RMS, cosine, shapes и CPU мокам.
5. Не увеличивать 28 Gradio params без миграции PNG Info/presets/Explorer, проверки UI и тестов.

[Подробные риски](development/WP03_LAYER_COMPOSER_RISKS_RU.md) · [Исторический план накануне ревизии](docs/history/snapshots_2026-11-20/PLAN-NEXT-RU.md).
