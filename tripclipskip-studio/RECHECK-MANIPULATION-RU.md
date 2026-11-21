# TripClipSkip Studio — перепроверка исследования манипуляций CLIP

**Дата:** 14 ноября 2026 г.  
**База:** ZIP `tripclipskip-studio-0.2.1-audit.zip`, исследование `MANIPULATION-LAB-RESEARCH-RU.md`, кастомные `modules (4).zip`.  
**Статус:** статический аудит и локальные тесты; **GPU-валидации новых идей не было**. Никаких изменений runtime-кода расширения в ходе этой перепроверки не внесено.

## 1. Что подтверждено

- Фактическое ядро 0.2.1 допускает только дробную глубину `1 <= depth <= maximum` (`studio_core/interpolation.py`, `skip_pair`). Значения 0, 0.5, 0.75 **не реализованы**. Проверено прямым вызовом функции.
- Выбор опорного слоя L действительно зависит от `a1111 / model / custom` (`studio_core/reference.py`). Для SDXL штатный A1111 `sdxl_clip_l_skip=True` читает `hidden_states[-CLIP_stop_at_last_layers]`; при `False` берётся модельная `last` либо `hidden + layer_idx`. Для L при skip=1 источники `a1111` и `model` поэтому могут не совпадать.
- CLIP-G в текущем ZIP захватывается forward-хуками на residual blocks и штатный pooled сохраняется. Изменение depth G **не меняет pooled**; не следует описывать результат как «полное изменение G». Во встроенном SDXL/SGM pooled извлекается из финальной ветви G, а не из произвольного захваченного слоя.
- Смешивание произвольных двух и нескольких слоёв *одного энкодера* вычислительно осуществимо при совпадающих shape/dtype и корректной обработке нормализации. Это **не гарантия качества изображения**.
- Пер-слойные настройки для отдельных токенов, независимые промпты L/G и плавные временные кривые требуют дополнительной системы согласования chunk/token/conditioning; в текущей 0.2.1 этих функций нет.
- REGION + Dual-Phase в 0.2.1 остаётся намеренно выключенным. REGION + дробный skip пользователь предварительно подтвердил на GPU; это не подтверждает совместимость будущих REGION-specific манипуляций.

## 2. Важные уточнения / недочёты прошлой формулировки

### A. `0.5` нельзя просто разрешить в UI

Формула `H(d) = d H[r] + (1-d) H[r+1]` для `0 <= d <= 1` математически верна для **двух совместимых тензоров**, но `r+1` может отсутствовать. Дополнительная тонкость: SDXL CLIP-L `last_hidden_state` после финального LayerNorm не равен автоматически сырому `hidden_states[-1]`; при переходе от native-`last` к более раннему скрытому состоянию нельзя бессознательно смешивать нормализованный и ненормализованный векторы. Нужно явно фиксировать политику (`native`/`raw`/`normalized`) и тестировать её.

### B. Индекс `hidden_states[0]` — не обычный выход transformer block

Для HF CLIP `hidden_states[0]` соответствует входному embedding-представлению до transformer layers, а `hidden_states[1]` — состоянию после первого блока. `Layer Map` обязан это явно обозначать. Диапазон skip не должен автоматически называть `hidden_states[0]` «нулевым CLIP-слоем» без пояснений.

### C. Потенциальная пограничная ошибка текущего `reference_layer`

При `source='a1111'` код проверяет `1 <= CLIP_stop_at_last_layers <= depth` **даже в случае SDXL, когда `sdxl_clip_l_skip=False` и глобальный skip штатно игнорируется**. Например, искусственный `skip=20`, `sdxl_clip_l_skip=False`, `depth=12` порождает `ValueError`, хотя штатный SDXL в таком режиме не использовал бы skip. Нужно проверять диапазон только при фактическом применении skip. В обычном UI значение за пределами глубины может быть недоступно, но сама логика должна быть корректной.

### D. Независимые L/G-промпты — сложнее простой конкатенации

Токенизация и 75-токенные chunks двух текстов могут дать разное число последовательностей/длин. Нужны выравнивание, маски BOS/EOS/padding, policy для missing-chunks, negative/prompt-editing и корректный pooled-G; нельзя просто взять два несовместимых тензора и вызвать `torch.cat`.

### E. Больше attention-хуков — выше риск конфликта с кастомной сборкой

`Prompt-to-Prompt`/`Attend-and-Excite` манипулируют UNet attention. В модифицированных `modules` уже есть REGION/Forge attention. Исследовательские attention-функции допустимы только как отдельный opt-in backend с проверкой владельца хуков и изоляцией от REGION.

### F. Метод восстановления патчей не решает все сторонние конфликты

`MethodPatch.remove()` осторожно не перезаписывает более позднюю стороннюю замену метода, но это **не гарантирует**, что новая обёртка не ссылается на ранее установленную TCS-функцию. Для совместимости с несколькими расширениями нужна проверка порядка установки и удаление патчей на реальной A1111-сборке.

## 3. Итоги тестов и границы уверенности

- Запуск `pytest -q` в каталоге 0.2.1: **49 passed, 1 skipped, 0 failed**.
- Код `scripts/tripclipskip.py`, `studio_core/interpolation.py` и `studio_core/reference.py` в рабочей папке **побайтово совпал** с `tripclipskip-studio-0.2.1-audit.zip`.
- Проверка `skip_pair(0.5, 12)` дала ожидаемый `ValueError`.
- Отдельный пограничный тест `reference_layer('sdxl', ..., opts.skip=20, opts.sdxl_clip_l_skip=False, depth=12)` воспроизвёл `ValueError` (см. пункт C).
- GPU-тестов плавного timeline, произвольных слоёв и per-token смешивания **нет**, потому что этих функций нет в 0.2.1.

## 4. Уточнённый приоритет разработки

**P0, прежде чем расширять CLIP:** исправить проверку неиспользуемого SDXL-native skip, добавить нормализационную диагностику `raw/final_norm`, подтвердить на GPU native identity, совместимость Hires, Explorer и REGION; зафиксировать commit A1111.

**P1:** read-only Layer Map, указывающий 0 = input embeddings; затем bidirectional `0..1` только при наличии позднего слоя, потом arbitrary two-layer blend с явным режимом нормализации. Покрыть тестами граничные слои и сохранение pooled G.

**P2:** Multi-Layer Fusion, Layer Explorer 2.0, независимые L/G prompts (после согласования chunk sizes), плавный Timeline на структурированном conditioning без повторного side-effects `setup_conds`.

**P3:** per-token / pooled / region-specific манипуляции, attention editing и Illustrious Encoder Lab только экспериментально и с проверкой конкретных SDXL checkpoints.

## 5. Основные источники для проверки

- https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_clip.py
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_open_clip.py
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/processing.py
- https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/encoders/modules.py

**Решение:** техническое направление в исследовании сохраняется, но новые функции нельзя включать в стабильный режим без указанных проверок. Текущий ZIP 0.2.1 этой перепроверкой не изменён.
