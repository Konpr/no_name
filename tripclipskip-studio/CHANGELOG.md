## 2026-11-21 — WP-03B1c-a-r1-docsync (только документация)

- Синхронизированы текущие ревизия, CPU/GPU статус и ссылки в `README-RU.md`, `README.md`, `PLAN-NEXT-RU.md`, `ROADMAP.md`, `development/AI_HANDOFF_READ_FIRST_RU.md`, `development/CURRENT_STATE.json`.
- **Текущий runtime остаётся WP-03B1c-a-r1.** Исходные 758 passed (dev+custom, два `SyntaxWarning`) / 736 passed + 22 skipped (без внешних sources) не переименованы в новый GPU acceptance; B1b-r2 и B1c-a-r1 — GPU pending.
- Старые ZIP-манифесты остаются историческими; добавлен новый для этой документационной сборки. **Не менялись** source code расширения, алгоритм генерации, native pooled-G, UI и исходники A1111. [Краткий аудит](development/WP03B1C_A_R1_DOCSYNC_RU.md).

## 2026-11-20 — WP-03B1c-a-r1 (independent CPU re-audit)

- Исправлено неверное наследование `token_chunk_ref` и `token_forward_ref` между CLIP-L и OpenCLIP-G при вложенных cross-wrapper вызовах. Контекст ограничен идентичностью wrapper; никакой смены алгоритма генерации.
- Добавлено 2 adversarial CPU-теста; обе ветки A1111: 758 passed (2 прежних SyntaxWarning); без sources: 736 passed, 22 skipped.
- GPU для r1 и B1b-r2 всё ещё не пройден. [Аудит](development/WP03B1C_A_R1_INDEPENDENT_REAUDIT_RU.md).

## 2026-11-20 — WP-03B1c-a (CPU preview)

- Opt-in read-only A1111 process_tokens and text-forward observations: token chunk links, job-local keyed HMAC token input, TI fix metadata; pooled first-chunk object identity; construction result tensor boundary, custom parser lexical hints only.
- Fail-open scoped per-instance wrappers, no global monkey-patching or additional forward/tokenization; no changes to generated conditioning, pooled-G, UI controls, A1111 sources.
- Full tests with both user A1111 source trees: 756 passed, 2 pre-existing SyntaxWarnings. B1b-r2+B1c-a GPU acceptance still pending.

## 2026-11-20 — WP-03B1b-r2 (CPU preview)

- Read-only Hires `calculate_hr_conds()` scopes with verified cache identity; Hires prepared from `hr_c/hr_uc` rather than stale `c/uc`.
- Bounded multi-denoiser / reused-denoiser forward capture, first N per instance and pass.
- Numeric observed-layer source_guard; r1 GPU evidence archived, r2 GPU pending.
- No change to denoising algorithms, UI parameters, A1111 sources or pooled-G.

## 2026-11-20 — WP-03B1b-r1 повторный аудит A1111 dev + custom (CPU preview)

- Исправлено ложное `construction_pass=base` для Hires.fix conditioning, вычисляемого до `is_hr_pass=True`. Достоверные cache-slots определяются через identity; неизвестные/alias слоты обозначаются `unknown`.
- Добавлено evidence для pass; 6 независимых проверок с обоими source ZIP и реальными AST-функциями `get_conds_with_caching`.
- 728 passed / 2 исторических SyntaxWarning с обоими исходными деревьями; 712 passed / 16 skipped без них. GPU r1 не тестировался.
- `Script.process()`, CLIP/U-Net/pooled, 28 UI параметров и исходники A1111 не изменены. [Отчёт](development/WP03B1B_R1_REAUDIT_RU.md).

## 2026-11-20 — WP-03B1b Scoped Branch Provenance (CPU preview)

- Только для opt-in Native Proof (`TCS_PROOF_LAYERS=1`) добавлена безопасная scoped-классификация A1111 positive/negative **construction context** через `setup_conds → get_conds_with_caching`, точные функции и `SdConditioning.is_negative_prompt`.
- В `encoder_l/g` и `encoder_layers` — `construction_branch`, `construction_scope_id`, `construction_pass`; отдельные `construction_scopes` показывают число native encoder invocations. Основной `branch` намеренно остаётся `unknown`, `trusted_provenance=false`.
- Никакой подмены conditioning, pooled-G, TE/U-Net, 28 UI controls или новой генерационной математики.
- 710 passed / 12 skipped без custom A1111; 722 passed / 2 SyntaxWarning с присланным `modules (4).zip`; **GPU не проверен**.
- [Отчёт и ограничения](development/WP03B1B_SCOPED_BRANCH_PROVENANCE_RU.md).

## 2026-11-20 — WP-03B1a read-only encoder capability diagnostics (CPU preview)

- Opt-in structural fingerprint (metadata SHA256, не идентификатор модели или весов).
- Один invocation ID связывает `encoder_layers` и `encoder_l/g` в Native Proof; branch остаётся UNKNOWN.
- Только формы входа/выхода и attached pooled, no raw tokens/prompts/tensors, no extra forwards.
- 12 новых CPU adversarial tests, 703 passed / 11 skipped без custom modules.
- Без GPU acceptance и без изменения UI/denoising/Layer Composer/pooled injection.

# TripClipSkip Studio — история изменений

## 2026-11-20 · WP-03B0.2-r3-r2 · Normalization & Docs

- Для G `last` больше нет повторного вызова `ln_final.forward` при проверке якоря и дробных endpoint; поддерживаются ограниченные известные классы OpenCLIP с обязательной native byte parity.
- L `last` смешивает состояния с согласованной финальной нормализацией, без дополнительного вызова `final_layer_norm.forward`.
- Adversarial тесты на OpenCLIP-подобный LayerNorm, сторонние хуки, неизвестный нормализатор, 1 native forward и cleanup.
- README, PLAN-NEXT-RU, ROADMAP, handoff и machine-current-state обновлены; исторические root-аудиты и GPU-отчёты перенесены в docs/history, ссылки перепроверены.
- CPU preview; live SDXL GPU проверки именно этой ревизии нет.

---

## 2026-11-19 — WP-03B0.1-r1 (independent CPU regression recheck)

- Исправлен `tests/test_recheck_0315r1.py`: отсутствующий `TCS_CUSTOM_MODULES` теперь приводит к явному pytest skip; неправильный путь с установленной переменной — к понятной ошибке.
- Два воспроизводимых прогона: 627 passed / 11 skipped без внешних исходников и 638 passed / 2 warnings с архивом `modules (4).zip`.
- Внутренние документы и ZIP-манифест обновлены; code observer/CLIP/generation/UI полностью сохранён.
- **Нет live A1111/GPU identity.** [Ревизия](development/WP03B0_1_R1_RECHECK_AND_HANDOFF_RU.md) · [live checklist](development/WP03B0_1_R1_LIVE_A1111_CHECKLIST_RU.md).

---

## 2026-11-19 — WP-03B0.1 Native Layer Observer CPU Preview

- Новый `studio_core/compose/native_layer_observer.py`: read-only scoped forward hooks HF L/OpenCLIP G (executed blocks and optional final LayerNorm).
- `live_proof.py`: явный `TCS_PROOF_LAYERS=1`, cap encoder invocations, per-invocation UUID, metadata-only layer events and same-call native output incl. pooled.
- Реализован CPU test suite, а не реальная A1111 native identity. Текущие 28 UI params, генерационный код и legacy features сохранены.
- Нет trusted positive/negative, token provenance, Layer Composer injection, GPU acceptance. [Инструкция](development/WP03B0_READ_FIRST_RU.md).

---

## 2026-11-19 · WP-03A-r2 · Audit-driven CPU contract hardening

- `LayerSample`: asserted `token_layout_digest`, `capture_invocation_id`; строгий `require_capture_metadata` для смешивания, legacy-compatible default.
- `diagnose_layer_signal`: отдельно разрешённый `cross_stage_observational`, `high_rms_ratio`, `high_signal_fraction`, предупреждение о zero reference. Без quality/native claims.
- `snapshot_layer_sample`, `validate_observed_layer_index` — opt-in CPU helpers.
- 53 новых целевых проверок, полный регресс и manifest (см. новый handoff). **Нет изменений scripts/runtime/generation/UI.**
- [Читать первым](development/WP03A_R2_READ_FIRST_RU.md); [закрытие пунктов аудита](development/WP03A_R2_AUDIT_CLOSURE_RU.md).

---

## 2026-11-18 · WP-03A-r1 Docs/Handoff Refresh (документационная поставка)

- **Runtime code unchanged:** ни CLIP/torch math, ни `Script.process`, ни UI, ни PNG Info/пресеты, ни 28 A1111 positional control args не менялись.
- Добавлен единый [handoff для следующего ИИ/разработчика](development/AI_HANDOFF_READ_FIRST_RU.md), машиночитаемый [current state](docs/history/historical_state/WP03_CURRENT_STATE_2026-11-18.json), спецификация [WP-03B0 native capture](development/WP03B0_NATIVE_CAPTURE_SPEC_RU.md), [регрессионная матрица](development/WP03B0_REGRESSION_MATRIX_RU.md), [реестр рисков](development/WP03_LAYER_COMPOSER_RISKS_RU.md), [состав поставки](development/WP03_BUILD_CONTENTS_RU.md).
- Актуальный статус WP-03 вынесен наверх README/ROADMAP/PLAN-NEXT без удаления исторических спецификаций.
- В поставке отдельный ZIP-манифест с проверкой SHA-256; исторические манифесты сохранены, **не являются проверкой текущего архива**.
- **Статус:** WP-03A-r1 CPU verified (568 previous tests); WP-03B0/B1/B2 и GPU acceptance ещё не выполнены.

---

## WP-03A · Layer Composer Math (CPU Preview, 2026-11-18)

- Added `compose/layers.py`: exact alpha endpoints, FP32 intermediate, guarded RMS matching, per-channel preview and strict model/tokenizer/TI-LoRA provenance.
- Added CPU regressions and independent SHA-256 verification of distributable ZIP. No changes to A1111 hooks, CLIP forward, generation UI (28 fields), PNG or token inspector.
- WP-03B capture owner and GPU native-equivalence are **not implemented/NOT-RUN**. See `development/WP03A_START_HERE_RU.md`.

---

## WP-02C UX r1 — readable Token Group and Smart BREAK (CPU Preview, 2026-11-18)
- Added read-only concise explanation of verified group matches and BREAK occupancy before detailed tables; native Gradio Accordion holds technical WP-02A/B/C sections.
- Existing diagnostic inputs remain 4 / 5, both callbacks now return 2 display outputs. Script.ui generation ABI remains 28; CLIP and PNG unchanged.
- See `development/WP02C_UX_START_HERE_RU.md` and `development/WP02C_UX_REGRESSION_RU.md`.


---

## 0.3.16 — Conditioning Inspector 2.0 (2026-11-18)

- Исправлена систематическая смещённость выборки первых 1024 элементов: теперь воспроизводимое покрытие токеновых позиций и каналов L/G, включая конец длинных промптов, без полного `flatten` не-contiguous срезов.
- Добавлена read-only позиционная карта для до 8 чанков: по 4 четверти с относительным RMS разности P2–P4 против P1 отдельно для L и G. Строго **не** интерпретировать как attention/tag influence.
- Для одномерного SDXL ADM-вектора длиной до 4096 сравнение между уже подготовленными фазами полное; это **не** Native Difference Meter между двумя запусками.
- Сохранены все 28 Script.ui параметров, режим «Только формы», ограничение на объём наблюдений и отказоустойчивость. Прочее ядро CLIP/фаз/PNG Info/пресетов не менялось.
- CPU/mock тесты с кастомным A1111, сравнение архивов предыдущих версий; требуется GPU/UI-проверка новых численных карт.

## 0.3.15-r1 — Preset migration recheck (2026-11-18)

- Исправлена повторная миграция legacy-пресетов: старые новые записи теперь копируются в стабильное `data/` даже если этот файл уже создан; стабильная копия побеждает конфликты имён.
- Добавлен CPU-интеграционный тест с извлечённой оригинальной функцией `modules.infotext_utils.connect_paste()`, проверяющий генерацию Gradio-обновлений всех 28 контролов из `TCS State`.
- Не менялись CLIP-L/G, Multi-Phase, Absolute Timeline, Conditioning Inspector, `shared.opts` или пользовательские `modules`.
- Проверка реального PNG Info → Send to txt2img/img2img и перезапуска A1111 всё ещё требует браузерного пользовательского теста.

## 0.3.14 — Conditioning Inspector (2026-11-17)

- Новый **opt-in Conditioning Inspector**: во время существующего `p.setup_conds()` считывает фактически подготовленные структуры `p.c`/`p.uc`, включая кастомный `MulticondLearnedConditioning.batch → schedules → cond`; не запускает второй CLIP-forward.
- Режим «Только формы» считывает shape/dtype/device без GPU→CPU переноса данных. Опциональные «Нормы и сравнение фаз» измеряют ограниченный детерминированный префикс (до 1024 элементов) и попарные sample `max_abs_diff`, `cosine`; для SDXL `[... ,2048]` показывает также срезы каналов CLIP-L (768) и G (1280).
- Отдельная русская вкладка с кнопкой «Показать измерения последней генерации». В PNG Info — только короткая запись о статусе; prompt, Token IDs и сырые эмбеддинги не сохраняются.
- Диагностика включается вручную, по умолчанию выключена. Она может синхронизировать CUDA в sampled-режиме и никогда не заявляет полную native equivalence. Отказ диагностики не прерывает generation.
- XYZ Plot и Layer Explorer выключают новый инспектор в сравнительных ячейках (без вмешательства в параметры следующего расширения).
- Скриптовый контракт расширен **с 27 до 28** аргументов: первые 27 сохранены в прежнем порядке, последний — `conditioning_inspect`. Старые пресеты загружаются с `Выключено`.
- Старые методы CLIP-энкодирования, PhaseSchedule, Absolute Layer Timeline и алгоритмы перехвата denoiser не менялись.

## 0.3.13 — CLIP Lab 2.0 integrated read-only UI (2026-11-17)

- Интегрирована настоящая новая вкладка Gradio `CLIP Lab 2.0 · интерактивно` с адаптивным HTML/CSS/JS, который загружается из каталога расширения; исходный офлайн-макет не подменяет реальную токенизацию.
- Вкладка использует штатные `tokenize_line()` CLIP-L/G для введённой копии positive/negative; доступны кликабельные BPE ID, веса, TI, карта 75 слотов, паддинг и подробная таблица по запросу.
- Единый read-only предпросмотр реально активного relative/absolute режима: фазы и слои L/G подсвечиваются, при TCS off показывается явный статус. Без исполнения CLIP forward и без редактирования `shared.opts`.
- A/B-песочница повторно токенизирует другой текст через штатный CLIP-L и сравнивает Token ID по позициям, а не фальшивое «слияние» отдельных токенов. Source offsets пока не заявляются.
- Технические значения и текст промпта HTML-экранируются, управление фазами выполняется только локально в DOM. Скрипты генерации, хуки и 27 позиционных UI-параметров A1111 **не изменены**.
- 213 локальных тестов с custom `modules (4).zip`; Chromium browser-check (desktop/mobile) для изолированного рендера. Полный browser/GPU-тест пользователя A1111 и Native Equivalence остаются открыты.

## 0.3.12-r1 · Аудит формулировок CLIP-G (17.11.2026)

- Исправлено неверное объяснение G32: при `penultimate` стандартный `FrozenOpenCLIPEmbedder2` может исполнять последний блок для pooled. G32 пока запрещён в экспериментальном редакторе как token-wise слой **по политике безопасности**, не из-за отсутствия вычисления.
- Изменены только тексты интерфейса, справки и сообщения валидации. Расчёт слоёв, CLIP-хуки, `setup_conds`, PNG Info и фазовый планировщик не изменены.
- GPU-проверка абсолютного режима всё ещё необходима.

## 0.3.12 — Experimental Absolute SDXL Layer Timeline + Encoder Anatomy (2026-11-17)

- Opt-in SDXL absolute raw `hidden_states[-N]` and physical OpenCLIP-G block numbers in 2–4 STEP phases, validated before installation of hooks.
- Read-only preview of parsed phases, conservative G `penultimate` availability guard, native pooled G preserved.
- Separate Russian Encoder Anatomy tab documenting token-wise vs pooled, raw/final LayerNorm, real L/G block counts.
- A1111 script_args now 27 (first 25 unchanged); legacy presets remain disabled by default; XYZ/Layer Explorer override absolute timeline for relative comparisons.
- Preview only; CPU/mock tests and supplied custom-modules source contract, GPU validation of this feature pending.

## 0.3.11-token-laboratory-preview — 2026-11-17

- Расширен read-only Token Inspector до русскоязычной **Лаборатории токенов и чанков**: позиционная карта BOS/75 слотов/EOS для каждого чанка (■ обычный токен, + усиленный, - ослабленный, ◆ TI, · паддинг), занятость и вместимость по чанкам, статистика padding и веса.
- Новые подсказки о досрочном закрытии чанка из-за `BREAK` или `comma_padding_backtrack`, в том числе для чанков с заполнением менее 20%; `BREAK` на отдельной строке без запятой — штатный вариант.
- Сравнение BPE Token ID между CLIP-L и OpenCLIP-G с предупреждением: **совпадающие ID не означают совпадающие эмбеддинги**. Для positive/negative показано пересечение ID (не метрика CFG). Странные BPE-символы при Unicode объясняются byte-level токенизацией; обнаруживаются невидимые символы и U+FFFD.
- Два необязательных UI-параметра *только инспектора*: «Обзор и таблицы / Только обзор» и «Сравнить с вариантом без BREAK». Во втором случае `tokenize_line()` повторно вызывается на изменённой **копии** текста: показываются число чанков и входных позиций; оригинальные поля txt2img/img2img не меняются.
- Ограничены размер вывода, число карт и таблиц. Статистика учитывает полную строку; явный EOT в конце промежуточного чанка остаётся потенциально неоднозначным, потому что A1111 не возвращает длину каждого промежуточного чанка.
- Добавлено 16 тестов (включая исполнение `tokenize_line` из кастомных `modules (4).zip`); **169 passed** с custom source. По сравнению с 0.3.10 генерационные 25 `script_args`, Phase Trace, CLIP-L/G, REGION/Hires и Math Scheduler не изменены.
- Пользовательские скриншоты подтверждают, что исходная вкладка 0.3.10 показывала два чанка, positive/negative, веса и CLIP-L/G; новая 0.3.11 требует собственного браузерного теста. Native-equivalence остаётся открытой.

## 0.3.10-token-inspector-eot-hotfix-preview — 2026-11-17

- Повторный аудит 0.3.9 обнаружил крайний случай в read-only Token Inspector: первый `id_end` внутри содержимого считался началом паддинга. Если пользовательский EOS-токен появлялся **внутри** промпта, реальные токены после него скрывались в выводе.
- Исправлено определение заполненных позиций: сохраняются все токены до последнего не-EOS, а для последнего чанка дополнительно учитывается точный `token_count` из оригинального `tokenize_line()`. Явный EOS в конце последнего чанка также отображается.
- Добавлены 6 регрессионных тестов, включая исполнение оригинального метода `TextConditionalModel.tokenize_line()` из кастомных `modules (4).zip` в изоляции (без запуска GPU/A1111).
- Основной генерационный код, CLIP-L/G, Multi-Phase, 25 аргументов script_args, REGION и Hires не изменены. Проверка на GPU и в браузере остаётся необходимой.

## 0.3.9-token-inspector-preview — 2026-11-17

- Добавлена русская read-only вкладка **«Токены и чанки»** для ручной копии positive/negative prompt. Использует настоящие загруженные обёртки A1111 CLIP-L и OpenCLIP-G (`tokenize_line()`, BPE-декодеры, multipliers, TI fixes). Показывает Token ID, BPE, веса, TI, число чанков, 75/77 позиций и token_count.
- Ограничен объём UI-результата и входного текста, добавлены проверки нештатных форматов. Не кодирует CLIP и не обходит REGION/TONEG/Prompt Editing; важные ограничения отмечены в UI и руководстве.
- Генерационные 25 контролов/порядок A1111, алгоритм фаз, CLIP-математика, PNG Info и пресеты **не менялись**.
- 147 локальных CPU/mock/контрактных тестов (с пользовательскими `modules (4).zip`), включая 9 новых тестов Token Inspector. GPU/UI-тест 0.3.9 ещё требуется.

## 0.3.8-russian-ui-preview — 2026-11-17

- Интерфейс расширения переведён на русский язык: управление CLIP, Timeline / Multi-Phase, пресеты, диагностика модели, Layer Explorer и новая CLIP Layer Map. Вкладка Layer Map по умолчанию формирует русские подписи и пояснения.
- Внутренние значения `a1111/model/custom`, `lerp/slerp`, `denoiser/sampler`, названия PNG Info, параметры пресетов и 25 аргументов A1111 **не изменены**. `render_layer_map(locale='en')` сохранён для совместимости с прежними программными тестами.
- Параметры CLIP-L/G, подготовка conditioning, хуки, алгоритм фаз, Phase Trace и поведение REGION/Hires не изменялись. Локальные CPU/mock тесты не заменяют запуск новой сборки в браузере A1111.
- Добавлены регрессионные тесты локализации.

## 0.3.7-layer-map-readonly-preview — 2026-11-17

- Added **Layer Map · read-only** tab to txt2img/img2img with Refresh button, requested 1–4 phase layer labels, selected block diagrams, per-branch fractional interpolation, and native pooled annotation.
- Uses existing runtime `layer_map.py`, `reference.py` and `multiphase.py` calculations; no generation arguments, CLIP hooks, conditioning or `shared.opts` are changed.
- Added isolated CPU tests for native/global/custom references, 4-phase G linkage, SD1, unavailable layers, and no global-option mutation.
- Native-equivalence GPU acceptance and real checkpoint/UI testing **remain open**. 0.3.6 Phase Trace code unchanged.

## 0.3.6 Phase Trace GPU Recheck Preview (2026-11-17)

- Corrected **false early-stop warning** in `sampler` progress mode after user GPU proof: Euler SDXL used all 20 denoiser calls while A1111 callback `sampling_step` ran `0,0,1,...,18` (one-forward lag). A narrowly scoped trace guard suppresses that warning only when 20/20 forwards, exactly one counter step of lag, stable totals and no fallback/nonmonotonic counters are observed.
- Genuine short runs, other counter mismatches and missing phase warnings are still reported. `TCS Phase last progress` retains the *actual* raw sampler value (18/20); no counter rewriting, no changed phase selection or conditioning.
- Added deterministic log-replay regressions: observed denoiser `[5,4,6,5]` and sampler `[6,4,6,4]` for thresholds `0/.25/.45/.75`, plus malformed/early-stop cases. Updated status and GPU evidence notes.
- The actual GPU test was on **0.3.5** (user's SDXL/Euler, base 20 steps, no Hires). This 0.3.6 patch has not run on the user's GPU yet. `modules` untouched.

## 0.3.5 Passive Sigma Trace / XYZ Guard Preview (2026-11-17)

- **Fix confirmed XYZ Plot bug:** `Trip Clip Skip late value` and `Trip Clip Skip switch fraction` now force the grid cell into legacy Dual-Phase (`multi_count=0`), preventing Multi-Phase from silently ignoring the requested axis. CLIP-G remains separately controlled if `Independent OpenCLIP-G` is enabled.
- **Opt-in Passive Forward Trace:** checkbox in Timeline, OFF by default. Records denoiser `step/total_steps`, A1111 `state.sampling_step/sampling_steps`, sigma (scalar or bounded range), call index and base/Hires pass label in the terminal. No phase switching or extra CLIP encoding is introduced; works also with the Timeline disabled.
- Terminal trace limited to 160 per-forward lines; PNG Info stores aggregate counts and the final diagnostic call. **These counts are per-job, NOT per-image** with batches/iterations.
- Sigma inspection on CUDA may synchronize the device; intended for short debugging runs, not performance benchmarking. Diagnostic errors disable the recorder without aborting sampling.
- UI and preset script argument count changes from 24 to 25 (new optional checkbox appended at the end; legacy positional fields retain their indices). Added mock regression tests for safe sigma reading, lifecycle, A1111 progress, Phase Trace, XYZ grid and error isolation.
- Core fractional-CLIP mathematics, scheduling math and REGION/Hires policies unchanged. **GPU/A1111 runtime validation of 0.3.5 is pending**, including the earlier `[6,3,9,0]` case.

## 0.3.4 Phase Thresholds Preview (2026-11-15)

- Fixed preset validation so inactive custom native Clip Skip in SDXL is preserved without rounding; active SDXL and SD1 native references still require integers.
- Added planned phase thresholds and actual forward-call count to PNG Info for easier comparison with observed phase transitions.
- Warn explicitly when a phase's requested start threshold was not reached; retained the old progress source and scheduler semantics.
- Preserved legacy Dual-Phase at 0% and 100% without misleading missing-phase warnings.
- Added regression tests and a GPU verification checklist. No changes to CLIP tensor math, REGION, Hires or Layer Explorer.

# Changelog

## 0.3.15 — PNG Info round-trip / долговечные пресеты (2026-11-18)

- Восстановление всех 28 настроек через versioned `TCS State` в PNG Info и `Script.infotext_fields` A1111. Для старых PNG — честное частичное восстановление.
- Пресеты в WebUI data_path (с миграцией из extensions), кнопки «Запомнить» / «Восстановить последние».
- Явная подпись scope статистики: Phase Trace текущей итерации / Passive Trace всего job.
- Генерация, CLIP-L/G, Absolute Timeline, conditioning hooks не изменены. Новый браузерный/GPU тест PNG Info и пресетов ещё нужен.



## 0.3.3-reference-hotfix-preview — 2026-11-15

- **P0 reference fix:** fractional `TCS CLIP-L relative depth=2.5` is independent from native **integer-only** `CLIP_stop_at_last_layers=2`.  Exact old error `Native A1111 Clip Skip must be an integer` is emitted only by `source=custom` and an invalid fractional `manual_skip`, not by valid fractional relative depth.
- SDXL `source=custom` without **Apply custom native Clip Skip to SDXL CLIP-L** now correctly ignores the unused integer reference field, as A1111 ignores it when not enabled.  Still rejects fractional native Skip if that reference is actually applied.
- Removed silent `int(2.5) -> 2` truncation when saving a custom-reference preset.  Invalid values now give actionable guidance instead of writing a misleading configuration.
- UI now labels the native reference as **integer only**, explains that fractional values belong in TCS CLIP-L, and distinguishes the per-job reference from global WebUI options.
- **Read-only Layer Map metadata preview:** `TCS CLIP-L layers` reports exact `hidden_states` indexes/weights; `TCS CLIP-G blocks` reports actual transformer block numbers and `pooled=native`. Does not modify model tensors. Full visual UI and absolute-layer schedule remain planned.
- Regression tests covering global Skip 2 with TCS L 2.5; SDXL apply-toggle on/off; invalid custom used/unused; actual `Script.process()` and PNG Info; saved preset validation; layer-map math. **98 passed** with user's custom `modules (4).zip` source contracts; no GPU integration tests for 0.3.3.
- No changes to sampling math, `lerp/slerp`, REGION policy, Layer Explorer, Hires policy, or phase scheduling. The earlier GPU trace `[6,3,9,0]` remains an OPEN issue.


## 0.3.2-recheck-preview — 2026-11-15

- Исправлена ошибка связанного CLIP-G в Multi-Phase: при `Independent OpenCLIP-G = off` Phase 2 CLIP-G теперь всегда соответствует Phase 2 CLIP-L, как и для фаз 1, 3 и 4; скрытый независимый слайдер G игнорируется.
- Вывод `STEP timeline counters` теперь не показывает фиктивную четвертую позицию, когда выбраны три фазы, и помечает отсутствие активной трассировки при блокировке (например, REGION).
- Исправлен текстовый комментарий UI: границы задаются относительно **выбранного** источника прогресса (`denoiser` или экспериментальный `sampler`).
- Новые регрессионные тесты, включая вызов настоящего `Script.process()` со stub-объектом A1111 и чтение собранного conditioning.
- Остальной CLIP-L/G, `lerp` / `slerp`, Hires, REGION, старый Dual-Phase, Explorer не изменены. Исходники пользовательских `modules` не изменялись.
- Проверка в ZIP: **80 passed** с предоставленным исходным каталогом `modules (4).zip`; без подключённых исходников `79 passed, 1 skipped`. **Это не 80 GPU-тестов**.
- Обнаруженная ранее `Phase 4 = 0` на пользовательском GPU ещё не воспроизведена под новой трассировкой; первопричина не установлена, поэтому мы её не объявляем исправленной.


## 0.3.1-phase-trace-preview — 2026-11-15

- На основании GPU-лога `per-phase calls=[6,3,9,0]` добавлены фактические Phase Trace и диагностика недостигнутых фаз; источник проблемы пока не установлен.
- Отдельные CLIP-L/G для Phase 2 внутри Multi-Phase, старый Dual-Phase независим и остаётся прежним.
- Включаемый вариант отсчёта по sampler steps (`denoiser` остаётся режимом по умолчанию).
- `PNG Info`: observed phase calls, transitions, last progress, warnings, effective status, источник прогресса.
- Улучшен UI Phase2/Dual visibility, прежние UI-индексы не сдвинуты; пресеты версионированы и мигрируют при загрузке.
- REGION/Hires safety policy не менялась; никаких изменений пользовательских `modules`.
- **74 локальных теста** с исходниками `modules (4).zip`; A1111 GPU-генерация именно этой сборки ещё не проверена.
- Подробности: `AUDIT-0.3.1-RU.md`, `TEST-PHASE-TRACE-RU.md`.


## 0.3.0-multiphase-preview — 2026-11-14

- Added a 3/4-phase STEP scheduler (separate CLIP-L and CLIP-G values; custom start fractions). Legacy Dual-Phase stays available through `Multi-Phase count=0` and unchanged saved settings.
- Added collapsible native Gradio UI rows for phases 2–4; all inputs are still passed positionally through A1111 Script hooks.
- Compute and validate all phase conditioning **once per native setup_conds invocation**, then switch only tensors during denoising. No per-step CLIP encoding.
- REGION + phase switching remains restricted to protect custom region attention hooks. Hires still uses phase 1.
- Fixed a reference validation issue: ignored A1111 global clip skip does not falsely reject SDXL CLIP-L when sdxl_clip_l_skip is off.
- Fixed Layer Explorer to disable the new Multi-Phase count inside grid jobs; does not modify the user's top-level controls.
- Added structural validation for alternate positive/negative conditioning, including SDXL pooled tensors; fail-closed error and PNG metadata.
- Added CPU/mock tests for boundaries, config, old presets, actual hook substitution, REGION/TONEG/Hires, and Explorer. **GPU integration unverified.**
- See TEST-MULTIPHASE-RU.md and PLAN-NEXT-RU.md.

## 0.2.1-audit — 2026-11-14

- Aligned Layer Explorer gallery metadata arrays with grid + all sample images.
- Preserve completed cells when Explorer is interrupted.
- Set native A1111 job_count before Explorer runs.
- Restore TCS hooks in processing.cleanup_processing via guarded p.close() even on generation errors.
- Added 3 regression tests and repeated source-contract tests against uploaded modules.
- See AUDIT-0.2.1-RU.md for limitations and reproducible test cases.


## 0.2.0-preview — 2026-11-14
- Добавлен выбираемый скрипт `TCS Layer Explorer (fixed seed)` для txt2img: сравнение CLIP-L × CLIP-G на одинаковом seed в штатном `process_images` A1111.
- Максимум 16 ячеек; CSV/диапазоны; подписанная сетка плюс отдельные кадры; ограничение batch=1; защита от refiner checkpoint swap.
- В Explorer Dual-Phase временно выключается только для этой серии изображений. Исходная панель UI пользователя не перезаписывается.
- Дополнена диагностика Dual-Phase: счётчики вызовов base/phase2/Hires в терминале.
- Расширены CPU/mock-тесты. Реальная GPU-генерация версии 0.2.0 в этой среде не запускалась.
- REGlON/TONEG-модули пользователя не изменялись. Защита `REGION + Dual-Phase` 0.1.2 сохранена.

## 0.1.2-custom-modules — 2026-11-14

- Audit supplied `modules (4).zip` and `modules(1).zip` (the smaller is byte-identical subset).
- Protect REGION custom setup from a second call during Dual-Phase; keep fractional CLIP active.
- Prevent TONEG negative-prompt duplication when preparing two phase conditionings.
- Preserve first-phase plus Hires conditioning after computing late-phase conditioning.
- Fix missing `SimpleNamespace` import that broke encoder diagnostics.
- Add custom processing detection in diagnostics, source-contract and mock tests.
- Add Russian compatibility report and outstanding GPU test plan.

# 0.1.0 (2026-11-14) — A1111 local preview

- Replaced repeated CLIP-L encoding with single-pass hidden-states extraction.
- Replaced CLIP-G independent second pass with native forward-hook capture.
- Kept the SDXL CLIP-G pooled tensor from the native encoder.
- Robust FP16-safe spherical interpolation; explicit shape/range validation.
- Addressed positional `process_before_every_sampling` argument mismatch.
- Added Hires cache invalidation and disabled phase switching during Hires pass.
- Based phase switching on actual `denoiser.total_steps` when available.
- Added JSON presets, Gradio UI tabs, architecture validation, teardown handling.
- Deliberately excluded unverified Illustrious TE swap, SD2 and unsupported forks.

# 0.1.1-audit (2026-11-14) — проверка A1111 Clip Skip / SDXL

- Добавлены опорные режимы CLIP-L (`a1111`, `model`, `custom`) и явное управление параметрами A1111 `CLIP_stop_at_last_layers` / `sdxl_clip_l_skip` без изменения глобальных настроек.
- Исправлено отсчитывание дробного CLIP-L от актуального native A1111 reference в SD1 и SDXL.
- CLIP-G по умолчанию независим от CLIP-L; улучшены подписи и диагностика.
- `scripts.xyz` перенесён в `studio_core.xyz` для устранения потенциальной коллизии импорта.
- Добавлена очистка при выгрузке scripts, отдельные тесты native/reference режимов, 24 теста прошли.
- План следующих этапов вынесен в `PLAN-NEXT-RU.md`, аудит — в `AUDIT-0.1.1-RU.md`.
- Статус остаётся preview: GPU/реальный A1111 не проверены.


## WP-03A-r1 · численная диагностика (CPU Preview)
- FP64 математика без промежуточного FP32 cast для FP64.
- Новый `signal_diagnostics.py`, предупреждения об ослаблении/изменении сигнала.
- 47 новых тестов, никакого live CLIP integration. См. `development/WP03A_R1_START_HERE_RU.md`.
