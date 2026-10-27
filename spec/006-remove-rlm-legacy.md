# SPEC-006: Чистка наследия RLM

Статус: planned · Волна 1 · Приоритет: низкий

## Проблема

В коде/доках остались упоминания RLM (прежний стек; сейчас deep-agents + langchain). Точки: env `RLM_MODEL_CONTEXT_WINDOW` (`api/utils/llm_tokens.py`, `.env.example:183`, README*.md), комментарии «The RLM map loop stays sequential» (`api/config/timeout.py:613`, `.env.example:130`), бессмысленный комментарий `zero-width space..RLM` (`api/memory/pgvector_backend.py:73`), docstrings/тексты в `api/expert/*`, `api/docgen/codebase.py:2007,3091`, `api/routers/expert.py:151`, тесты (`tests/unit/test_llm_tokens.py` и др.), README.md/README.ru.md.

## Решение

1. Env → `PRODUCTARIUM_MODEL_CONTEXT_WINDOW` (легаси-алиас с deprecation-warning — согласовано с SPEC-005).
2. Комментарии/docstrings переформулировать под текущий стек (deep-agents/langchain) без потери смысла; кривой unicode-комментарий в pgvector_backend исправить по факту диапазона.
3. README.md / README.ru.md / `.env.example` / `api/README.md` — вычистить упоминания, привести к единой терминологии.
4. Мёртвые куски кода, оставшиеся от RLM-пути (если найдутся при обходе), удалить; поведение не менять.

## Критерии приёмки

- `grep -ri 'rlm'` по репо возвращает только легаси-алиас env с пометкой deprecated (или пусто, если решено удалить сразу).
- Полный прогон pytest зелёный; `bun run lint/build` не затронуты.

## Затрагиваемые файлы

`api/utils/llm_tokens.py`, `api/config/timeout.py`, `api/memory/pgvector_backend.py`, `api/expert/prompt.py`, `api/expert/types.py`, `api/routers/expert.py`, `api/docgen/codebase.py`, `README.md`, `README.ru.md`, `api/README.md`, `.env.example`, `tests/**`.
