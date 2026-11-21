# TripClipSkip Studio 0.1.2 — аудит совместимости с пользовательской сборкой `modules`

Дата: 2026-11-14. Проверены именно предоставленные архивы:
- `modules (4).zip`: 195 записей, полный пользовательский набор `modules/`.
- `modules(1).zip`: 29 записей, выборочные файлы из того же набора.

## Результат сравнения

Каждый из **29 файлов** второго архива совпадает побайтово с одноимённым файлом в первом. Это не две альтернативные реализации. `modules(1).zip` **не является самостоятельной сборкой**: в нём отсутствуют некоторые важные файлы, которые используются модифицированным `processing.py`, например пользовательский `prompt_parser.py`, `forge_attention.py`, `sd_samplers_cfg_denoiser.py` и пакет `lib_region`. Установка этих 29 файлов поверх чистого A1111 без соответствующих дополнительных файлов может вызвать ошибки импорта или несоответствие API. Пожалуйста, не смешивайте неполный архив с другой версией A1111.

## Ключевые факты, подтверждённые содержимым ZIP

1. В `modules/processing.py::StableDiffusionProcessing.setup_conds()` реализованы извлечение `TONEG{...}`, REGION, включая `region_data`/`region_branch_counts`, а также установка/снятие attention-хуков `_monkey_patcher` / `_forge_shim`.
2. `setup_conds()` **меняет `self.negative_prompts`**, добавляя текст TONEG; повторный вызов без восстановления исходных отрицательных промптов дублирует его.
3. `REGION{...}` может задействовать собственный attention backend (`forge`, `monkey`, `latent`) и многоветвевой conditioning `AND`. Второй `setup_conds()` сбрасывает и воссоздаёт региональные хуки.
4. `modules/sd_samplers_cfg_denoiser.py` содержит специальные механизмы REGIONS, `region_branch_counts`, `state.region_skip_uncond`, а также CFG-параметры по изображениям. Точное соответствие `cond/uncond` и ветвей REGION критично.
5. `modules/scripts.py` передаёт всегда-активным скриптам значения `ui()` в порядке компонентов; `process_before_every_sampling()` принимает `**kwargs` и передаёт их в расширение.
6. `sd_hijack_clip.py` и `sd_hijack_open_clip.py` поддерживают штатную обёртку CLIP-L/G, которой пользуется наше расширение.
7. `modules/processing.py` формирует независимые Hires-conditioning, а `process_before_every_sampling()` вызывается и для txt2img, и для img2img, и для Hires.

## Совместимость сборки 0.1.2

| Режим | Статус по коду | Что происходит |
|---|---|---|
| Расширение выключено | Штатный путь | Ничего не патчится |
| SD1.x / SDXL, обычный дробный Clip Skip | Предварительно совместим | CLIP-патчи не меняют CFG/REGION-хуки и глобальные `shared.opts` |
| `TONEG{...}` + дробный Clip Skip | Предварительно совместим | Отдельных модификаций TONEG не требуется |
| `TONEG{...}` + Dual-Phase | Исправлено; CPU/mock | Второй проход получает исходные negative prompts; дополнение TONEG не дублируется |
| `REGION{...}` + дробный Clip Skip | Предварительно совместим, нужен GPU-тест | Первый штатный `setup_conds()` сохраняет установленные REGION хуки; изменения только в CLIP |
| `REGION{...}` + Dual-Phase | **Ограничено из соображений безопасности** | Для этого задания Dual-Phase не включается: остаются обычный дробный Clip Skip и REGION. Предупреждение в логах и метаданных |
| CFG по изображению / собственные REGION mask/batch | Не затронуто кодом расширения | Не меняем `region_roles`, `CFG`, `state` и `forward()` внутреннего attention. Нужна интеграционная проверка |
| Hires Fix (фаза 1) | CPU/mock проверено | Нативное однократное построение conditioning для HR, без phase-2 из предыдущего прохода |
| SDXL pooled | Сохраняется native CLIP-G | В preview версии pooled отдельно не интерполируется |
| LoRA / ControlNet / refiner / разные самплеры | Полная совместимость не доказана | Требуются GPU end-to-end тесты; refiner c Dual-Phase блокируется существующей защитой |

### Что было изменено в 0.1.2

- **Переставлен порядок построения Dual-Phase conditioning**: сначала вычисляется оригинальная фаза 1, затем (если можно) фаза 2, а исходные `p.c`/`p.uc` и `p.hr_c`/`p.hr_uc` восстанавливаются.
- Сохраняется начальный список `p.negative_prompts`, чтобы второй расчёт **не дублировал TONEG**; после поздней фазы возвращается результат первой.
- После первого native `setup_conds()` проверяется фактическое `p.region_data` и при активном REGION повторный расчёт не делается. Безопаснее проверять готовые данные, чем угадывать по синтаксису промпта: кастомные макросы могут создать REGION позднее.
- При активном REGION появится сообщение `REGION detected: Dual-Phase safely disabled` и параметр `TCS compatibility` в PNG Info.
- Исправлен отсутствующий импорт `SimpleNamespace`, ранее нарушавший диагностическую кнопку.
- В диагностике отмечается обнаружение пользовательской REGION/TONEG реализации обработки conditioning.
- Добавлены опциональные **тесты контрактов исходных файлов** (через `TCS_CUSTOM_MODULES`) и mock-тесты TONEG, REGION и Hires.

## Как выполнить проверку на вашей видеокарте

1. Сделайте резервную копию `extensions/` и конфигурации `config.json`, а также убедитесь, что `modules/` соответствует работающей сборке. **Не заменяйте файлы modules ZIP-архивом наугад.**
2. Отключите оригинальный TripClipSkip и удалите старую папку `tripclipskip-studio`, прежде чем распаковать 0.1.2 в `extensions/`.
3. Убедитесь, что модель SD1.5/SDXL генерирует без расширения (фиксированные seed, размеры, sampler и batch).
4. С включённым расширением проверьте L=1/G=1 в режиме reference=`a1111`: результат должен совпасть с контрольным при детерминированной конфигурации.
5. Проверьте L=1.5; G=1.0 и затем G=1.5 на SDXL, отрицательный prompt, длинный prompt, LoRA, `--medvram`, если используете.
6. Проверьте `TONEG{bad hands}` + Dual-Phase; убедитесь, что `bad hands` встречается в effective negative conditioning однократно, и что нет ошибок в консоли.
7. Проверьте `REGION{...}` с выключенным Dual-Phase, затем с включённым. Во втором случае должен появиться лог о выключении **только** phase switch; REGION и fractional CLIP должны продолжить работу.
8. Повторите Hires Fix и img2img; проверьте, что после остановки генерации новая генерация без TCS идёт нормально.
9. Если какие-либо тесты не проходят — сохраните console traceback, версию A1111, hash коммита и конфигурацию генерации; без этих данных заявлять полную совместимость нельзя.

**Статус:** исходники и пользовательская модификация просмотрены, 29 CPU/mock/source-contract тестов пройдено в данной среде. **Полноценный GPU-запуск пользовательской сборки здесь не выполнялся.** 

## Интернет-источники, сверенные дополнительно

- Официальный A1111 `modules/sd_hijack_clip.py`: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_clip.py
- Официальный A1111 `modules/sd_hijack_open_clip.py`: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_open_clip.py
- Официальный A1111 `modules/processing.py`: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/processing.py
- Официальный A1111 `modules/scripts.py`: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/scripts.py
