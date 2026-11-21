# TripClipSkip Studio — исследование расширенных манипуляций CLIP/conditioning

**Дата:** 2026-11-14  
**Исходная сборка:** 0.2.1-audit; в этом исследовании её исходники не изменены.  
**Платформа:** AUTOMATIC1111, включая пользовательскую модификацию `modules (4).zip` с REGION/TONEG.  
**Уровень подтверждения:** архитектура и возможности проверены по исходникам A1111, SGM/SDXL и локальному ZIP; перечисленные новые функции **ещё не реализованы и не протестированы на GPU**.

## 1. Что реализовано сейчас, и что пока нельзя обещать

- Сейчас CLIP-L выбирает два соседних hidden state относительно опоры `a1111/model/custom` и смешивает `lerp/slerp` с параметром `depth >= 1`.
- В SDXL CLIP-G независим, соседние слои берутся через временные forward hooks на residual blocks; pooled-вектор остаётся неизменённым, полученным штатным путём.
- Dual-Phase переключает подготовленное conditioning на поздней стадии denoising. При активном REGION вторую фазу текущая сборка отключает.
- Layer Explorer умеет фиксированный seed и сетку L×G, но ещё нуждается в проверке на GPU пользователя.
- Нельзя трактовать CLIP-L как «только персонажи», а CLIP-G как «только стиль»: их влияние перекрывается и зависит от checkpoint.

## 2. Первая группа: работа с глубиной слоёв — ближе всего к текущему коду

### M1. Bidirectional fractional depth (P1, после GPU-регрессий)

Сейчас `d=1` — слой-опора с индексом `r`, а `d=2` — слой `r-1`. Предлагаем для `0 <= d < 1` интерполировать между `r` и **следующим более поздним** `r+1`:

`H(d) = d*H[r] + (1-d)*H[r+1]` для `0 <= d <= 1`.

`d=0.5` — 50% опорного и 50% следующего слоя. `d=0` — следующий слой целиком **при его существовании**. Для опоры на последнем слое значения ниже 1 запрещены. Нельзя автоматически считать `0` выключением CLIP. Нормализацию «last» и «hidden» необходимо воспроизводить по правилам конкретного энкодера.

### M2. Arbitrary two-layer blend (P1)

Выбор произвольных двух слоёв одного энкодера, например `CLIP-L layer 7 + layer 11`, с отдельным `mix=0..1`. Поддержка `lerp`, защищённого `slerp` и сравнения с native. Показывать фактические индексы и стадии LayerNorm. Не смешивать напрямую 768-мерный L с 1280-мерным G.

### M3. Multi-Layer Fusion / Layer Bank (P2)

Получить 3–5 hidden states за один проход CLIP, затем `H = Σ(w_i*H_i)`, где `Σw_i=1`, `w_i>=0` в безопасном режиме. Примеры: тонкая смесь ранних/средних/поздних слоёв, presets для конкретных checkpoint. Можно изучить RMS-настройку перед смешиванием, но это **дополнительная экспериментальная трансформация**, а не гарантированное улучшение. Сохранять исходные масштабы отдельным режимом.

### M4. Layer Noise / Feature Steering (P3 / лаборатория)

Добавление контролируемого нормированного шума или вектора различий `H(prompt B)-H(prompt A)` в совместимые токеновые embeddings. Нужны ограничения по величине/норме и механизм отката. Риск артефактов и разрушения соответствия тегам; не включать в «обычную» панель.

## 3. Вторая группа: управление словами/промптами

### M5. Per-Token Layer Control (P2/P3)

Для выбранных токенов выходной conditioning брать из слоя A, для остальных — из слоя B. Маска применяется **после** полного прохождения CLIP; контекст самих токенов от других слов уже сформирован в энкодере. Это НЕ эквивалент независимой обработке каждого слова. Важно использовать реальные token IDs и chunk mapping (75-токенные сегменты A1111), не угадывать границы по пробелам. Обособить BOS/EOS/padding, учесть обе токенизации L/G и Textual Inversion. Примеры ручных групп: тег персонажа, выражение лица, фон, одежда.

### M6. Independent prompts for CLIP-L and OpenCLIP-G (P2)

Два текстовых поля, например L=`1girl, character_name, expression`, G=`lighting, painting style, scene`. Каждый энкодер получает собственную строку, а токеновые представления конкатенируются по каналам по стандартной SDXL-схеме. Это **не гарантирует** семантическое разделение «персонаж/стиль». Обязательные тесты: негативные промпты, расписания `[a:b:0.5]`, TI, LoRA, длинные тексты, порядок chunk/паддинга. Pooled должен соответствовать CLIP-G prompt.

### M7. Token/branch strength shaping (P2, экспериментальный)

Отдельный вес токена/группы для L и G, включая расписание по времени. A1111 уже поддерживает prompt emphasis, поэтому сначала измерить отличие и не дублировать без нужды штатную возможность. При нормировке менять масштаб умеренно, измерять распределения. Отдельно проверить positive/negative ветви, связанные с CFG.

### M8. Prompt-to-Prompt semantic edit / attention reweighting (P3)

Подход научных работ: контроль cross-attention отдельных слов, замена/фиксация карт внимания, управление интенсивностью выделенных токенов. Может помогать менять атрибуты объекта, сохраняя композицию, но потребует модификации attention внутри UNet. **Не совмещать автоматически** с текущими REGION/Forge Attention hooks; отдельный backend с явным обнаружением конфликта.

## 4. Третья группа: время генерации

### M9. Smooth Layer Timeline (P2)

Вместо фиксированного Dual-Phase заранее вычислять два или несколько совместимых conditioning и интерполировать их по `t` (доля denoiser-вызовов): `C(t)=(1-a(t))*C_early+a(t)*C_late`. Кривые: step, linear, smoothstep, keyframes. Нужно учитывать A1111 scheduled prompt structures, AND, positive/negative, SDXL pooled, batch и Hires, а не смешивать слепо произвольные объекты. Кэшировать только под корректными ключами (checkpoint, prompt, LoRA, опора, TI, размер SDXL и др.). Для REGION сначала только проверенный uniform-across-regions путь.

### M10. Independent L/G timelines (P2/P3)

Отдельные кривые глубины CLIP-L и CLIP-G: пример `L: 1→2.5`, `G: 2→1`. По желанию отдельные positive и negative кривые. Большой эффект на reproducibility; все расписания сохраняются в PNG info/JSON.

### M11. Hires/Refiner-aware schedules (P2/P3)

Раздельные расписания базового и Hires-прохода с явными временем, размерностью и pooled/spatial условиями. Refiner с другим checkpoint/энкодерами — отдельный совместимый backend, не автоматически переносить прежние тензоры.

## 5. Четвёртая группа: специфика SDXL

### M12. Pooled CLIP-G Lab (P3)

В SDXL 2D pooled conditioning формируется штатно из финального состояния и проекции OpenCLIP-G. Безопасный базовый режим — оставлять его native. Экспериментальные варианты: `pooled(prompt A) ↔ pooled(prompt B)` и расписание смеси двух **валидно закодированных** pooled vectors. Пытаться получить «pooled от произвольного промежуточного слоя» без специального преобразования может исказить распределение входа UNet; не называть эквивалентом обычного pooled.

### M13. Encoder-source pairing / Illustrious Lab (P3)

Перед заменой весов энкодеров сравнить реальные тензоры Illustrious и цели. Переключать CLIP-L/G только если архитектура/токенизация/размерности соответствуют. Важно: Danbooru-поведение не переносится гарантированно, если UNet обучался с другими представлениями. Потребуются LoRA/TI проверки и строгий откат.

## 6. Пятая группа: пространство изображения и диагностика

### M14. REGION-specific conditioning (P3; завязано на modules пользователя)

У каждой области свой набор параметров L/G и timeline. Требует собственного безопасного сборщика **региональных** conditioning ветвей без повторного `p.setup_conds()` и без сброса/переустановки REGION hooks. Пока REGION+Dual-Phase остаётся ограниченным. Проверять backend `latent`, `forge`, `monkey` раздельно на GPU.

### M15. CLIP Layer Map + Token Inspector (P1/P2)

UI показывает действительные `hidden_states[-k]`, доли смешивания, native anchor, используемые токены/чанки, подсветку TI и раздельные полосы L/G. Это визуализация, а не модификация conditioning. Сравнение реальных LayerNorm режимов и «last vs penultimate» обязательно.

### M16. Layer Explorer 2.0 (P2)

Сетки `dL × dG`, `layerA × layerB`, `strength × timestep`, fixed-seed несколько seed. Экспорт CSV/JSON с временем, VRAM, сопоставлением baseline; результаты без автосуждения «качественнее» без метрик/ручной оценки. Просмотр A/B и diff изображений. Сохранять все параметры генерации и модельные хэши.

## 7. Практическая архитектура

1. **LayerBank**: capture CLIP-L hidden states/CLIP-G residual blocks в одном проходе, с проверкой native-equality; избегать повторной TI токенизации.
2. **LayerComposer**: `select`, `lerp`, `slerp`, `weighted_sum`, `token_mask`, безопасная валидация dtype/shape/norm.
3. **ConditioningSchema**: рекурсивная проверка структуры SDXL, A1111 prompt-parser, pooled/vector, negative branch, batch и regional mapping. Не смешивать несовместимые расписания.
4. **ScheduleEngine**: одна кривая на группу параметров и проход (base/Hires); заранее вычисленные embeddings и вычисление коэффициента на шаге.
5. **Adapters**: в первой версии A1111 native; отдельный адаптер custom modules/REGION. Forge и SwarmUI — отдельные задачи.
6. **Read-only Diagnostics**: система фактически выбранных слоёв, нормализации, числа вызовов denoiser, времени кодирования и причины отключения функции.
7. **Experiment Runner**: A/B, фиксация seed/sampler/checkpoint/prompt/LoRA и сохранение исходных параметров.

## 8. Порядок реализации и обязательные стоп-условия

**P0**. Сначала GPU-регрессии 0.2.1: disabled/native identity, CLIP-G layers/pooled, Explorer metadata, Hires, REGION, error/interrupt cleanup. Без этого не усложнять sampling hooks.

**P1**. Layer Map, bidirectional 0…1 при доступном более позднем слое, Arbitrary two-layer blend. Проверка математических границ и режимов LayerNorm, тесты disabled/native equality.

**P2**. Multi-Layer Fusion, Layer Explorer 2.0, затем independent L/G prompts и smooth timeline (без REGION-specific branch). Проверить TI/LoRA/prompt editing/negative/batch.

**P3**. Per-token controls, pooled lab, region-specific layer banks, attention manipulation, foreign text encoders — только отдельные opt-in исследовательские режимы и расширенная GPU матрица.

**Stop conditions**: несовпадающее native output при identity-настройке, NaN/Inf, mismatch shape/chunk/pooled, потеря REGION-ветвей, оставшиеся чужие хуки, непредсказуемый рост памяти. При обнаружении — fail closed/rollback и видимое предупреждение, никаких silent fallbacks.

## 9. Первичные источники

- A1111 CLIP-L и prompt emphasis: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_clip.py и https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_emphasis.py
- A1111 OpenCLIP-G: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_open_clip.py
- A1111 prompt scheduling: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/prompt_parser.py
- A1111 callbacks: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/script_callbacks.py
- Stable Diffusion XL / SGM original encoder: https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/encoders/modules.py
- Diffusers SDXL conditioning documentation (для семантики, а не реализации A1111): https://huggingface.co/docs/diffusers/main/api/pipelines/stable_diffusion/stable_diffusion_xl
- Prompt-to-Prompt (paper + implementation): https://arxiv.org/abs/2208.01626 и https://github.com/google/prompt-to-prompt
- Attend-and-Excite: https://arxiv.org/abs/2301.13826
