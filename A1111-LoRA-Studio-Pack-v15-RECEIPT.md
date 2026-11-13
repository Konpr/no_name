# Квитанция v15 — 2026-11-13

Архив: `A1111-LoRA-Studio-Pack-v15.zip`

- Размер: **537494 байта**.
- SHA256: **b2cb3e6e6d1786249edf6e848d31d5f37e8efb2d43c7fe6776d3b40bbc0a3923**.
- ZIP CRC и совпадение 146 записей с каталогом пакета: PASS.
- 114 manifest assets, 73 install operations, 77 Python payload files.
- Все replace payload побайтово совпадают с рабочим проектом на D:.
- Processing text-patch проверен на исходной базе и на идемпотентность.
- SHA256 старых ZIP v8–v14 не изменены; содержимое каталогов v10–v14
  совпадает с их архивами. `lyco_helpers.py` не изменён.

Проверки на реально установленной одноразовой базе:

- 142 Torch CPU runtime PASS.
- 64 pytest parser/API/config PASS, 13 subtests PASS.
- 22 Node tests PASS, 62 UI assertions PASS в DOM shim.
- 73/73 SAME после установки; rollback 59 изменённых/добавленных путей;
  все 73 целевых пути проверены побайтово/на отсутствие.
- Installer/header tests: 26 PASS, 1 SKIP (symlink недоступен текущему пользователю).
- Ruff изменённых Python и Node syntax: PASS.

Тестовый host: `D:/AI-tools/lora-audit/tmp/lora-v15-installed-kha0e4qa`.
База — локальный `stable-diffusion-webui-dev.zip` плюс `modules.zip`, не живая
установка. 16 UNKNOWN разрешены только в этой новой тестовой копии. Логи находятся
в `evidence/` пакета; промежуточные неуспешные UI-прогоны описаны в VERIFICATION.md.
Скрипт smoke дошёл до итогового PASS после всех проверок rollback; при продолжении
диалога его завершённая exec-сессия уже отсутствовала, результат сверялся по
сохранённому полному журналу. Финальные installer tests и seal verifier завершились
с кодом 0 в текущем продолжении.

Исправлены два воспроизведённых Options race-сценария, согласованность локальных
LoRA asset caches и диагностические подписи. Полный аудит и остаточный план:
`A1111-LoRA-Studio-Pack-v15/AUDIT_V15_RU.md`.

Это не GPU/Gradio-приёмка и не универсальный resolver. Реальные hermitmoth/α,
Drive/FUSE, новые backend-функции, runtime coverage и partial-encoder TI warning
остаются открытыми. Зависимости не устанавливались; существующий Torch CPU
использовался с D:. `/content/A1111` не изменялся.
