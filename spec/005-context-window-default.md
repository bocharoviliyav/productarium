# SPEC-005: Дефолт контекстного окна 200k и overflow-ретрай

Статус: planned · Волна 1 · Приоритет: высокий

## Проблема

При неудачных пробах сервера контекст молча клампится к 8192 (`api/utils/llm_tokens.py:get_model_context_window`, fallback `final_ctx = ctx_found or 8192`) — бюджеты docgen ужимаются в 8k, качество генерации падает. Ожидаемый дефолт — от 200 000 токенов.

## Решение

1. **Каталог известных моделей**: статическая таблица «подстрока имени модели → окно» (qwen3*≈262k, deepseek*≈128k, llama-3*≈128k, gemini*≈1M, gpt-4o/gpt-5*≈128k+, claude*≈200k и т.п.), проверяется ПОСЛЕ env/админ-override и ДО живых проб.
2. **Fallback 200_000** (вместо 8192), когда модель неизвестна и пробы ничего не дали; WARNING-лог сохраняется (именованный, с подсказкой переопределения).
3. **Overflow-ретрай**: при ошибке context-overflow от сервера (маркеры уже есть: `_CONTEXT_OVERFLOW_MARKERS` в `api/docgen/codebase.py`) — халвинг действующего бюджета (floor 8192) и повтор вызова; подключить в общий путь бюджетирования docgen (не только в codebase).
4. **Env**: `PRODUCTARIUM_MODEL_CONTEXT_WINDOW`; легаси `RLM_MODEL_CONTEXT_WINDOW` читается с deprecation-warning (удаляется в SPEC-006/след. релизе).
5. Админ-панель Models: показать фактически разрешённое окно для модели каждой задачи (resolved), а не только поле `max_prompt_tokens`.

## Критерии приёмки

- Юнит-тесты: каталог (по имени), fallback 200k, порядок приоритетов (env > admin > каталог > пробы > 200k), халвинг при overflow (до floor).
- Лог `Resolved model context window ... 200000` для неизвестной модели без отвечающих проб.
- Прогон docgen на маленькой модели не ломается: overflow → халвинг → успех (мок-сервер в тесте).

## Затрагиваемые файлы

`api/utils/llm_tokens.py`, `api/docgen/codebase.py` (overflow-обработка → общий хелпер), `api/docgen/_common.py`, `api/routers/admin.py`, `src/app/admin/page.tsx` (Models), `.env.example`, `tests/unit/test_llm_tokens.py`.
