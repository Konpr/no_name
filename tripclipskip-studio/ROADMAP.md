# TripClipSkip Studio — дорожная карта

**Актуально 21.11.2026 · runtime WP-03B1c-a-r1, CPU preview; B1b-r2+B1c-a-r1 GPU pending. Документационное обновление без изменений runtime.** · [Детальный план](PLAN-NEXT-RU.md) · [Статус](development/CURRENT_STATE.json).

| Очередь | Функция | Статус |
|---|---|---|
| WP-03B0.1/B0.2 | Native layer proof, SDXL 12/12 L + 32/32 G, pooled-G | Live SDXL evidence, но не универсальная native equivalence |
| Runtime r1 | Denoiser patch lifecycle, mixed `cond/uncond` | CPU и SDXL GPU smoke |
| Explorer/PNG r2 | Пер-ячеечное подтверждение, prompt hygiene, cache refresh | CPU и пользовательский SDXL smoke/PNG Info |
| CLIP r3-r1 | One-pass L, value-proved G layout, symmetric axis guards | CPU-tested, GPU pending |
| **Normalization r3-r2** | `L last` consistency; `G last` norm/hooks safety | **CPU preview, GPU pending** |
| WP-03B1a / B1b-r2 | Structural diagnostics, construction positive/negative и Hires scopes | Частично read-only, CPU-проверено; r2 GPU pending |
| **WP-03B1c-a-r1** | A1111 token chunks/TI, HMAC inputs, first-chunk pooled identity, изоляция L/G | **Read-only CPU preview: 758 passed**, GPU pending; final operator lineage НЕ доказан |
| WP-03B1 (полная) | Trusted final source-to-result lineage после `ASSEMBLE/POOL/REGION` | Не реализовано; следующий read-only этап |
| Encoder Diagnostics UI | Архитектура L/G, нормализаторы, предупреждения и unsupported причины | План |
| Layer Explorer 2.0 | Выходы и нормализация выбранных слоёв, карты, проверенный A/B | План |
| Pooled-G Lab | Read-only токены vs native pooled | План |
| WP-03B2 / Pooled Composer | Injection, experimental pooled math, независимый pooled | Отложено до native GPU acceptance |

Для пользовательского опыта планируется компактная навигация по 4 разделам; **новые вкладки/слайдеры сейчас не добавлены**. Исторические GPU и CPU сведения в [архиве](docs/history/ARCHIVE_INDEX_RU.md).
