# SPEC-025: Краш чата эксперта — ValueError «content_and_artifact … <class 'list'>» при вызове MCP-инструмента

Статус: planned · Волна 13 · Приоритет: высокий

## Проблема

Ошибка в чате с экспертом: «Since response_format='content_and_artifact' a two-tuple of the message content and raw tool output is expected. Instead, generated response is of type: <class 'list'>» — граф агента падает целиком.

Механика (подтверждено по коду):
- `langchain-mcp-adapters` создаёт инструменты как `StructuredTool(..., response_format="content_and_artifact")` — контракт langchain_core требует, чтобы coroutine возвращала кортеж `(content, artifact)`; при `isError=True` их error-handler возвращает голый `list` блоков → внутренняя валидация кидает ValueError.
- Наш `api/mcp/manager.py::_wrap_bounded_tool` (~230-266) делает `tool.model_copy(update={"coroutine": _bounded})` — копия СОХРАНЯЕТ `response_format="content_and_artifact"`, а новый coroutine `_bounded` возвращает результат `tool.ainvoke(...)` (контент — str/list, не кортеж) → валидация копии валится на ЛЮБОМ вызове; исключение из внутреннего ainvoke (в т.ч. ValueError isError-пути) тоже не ловится.
- Экспертный чат/Deep Research получают инструменты через `get_tools_for_product` → `_wrap_bounded_tool` без защитного рерwrap'а (в отличие от DB deep-units, где `docgen/_common.plain_agent_tools` уже решает ровно эту проблему — см. docstring docgen/_common.py:760).

## Решение

- В `_wrap_bounded_tool`: копия с `response_format="content"` (контракт кортежа больше не применяется); `_bounded` обернуть в `try/except Exception` → короткая текстовая ошибка «ERROR: …» (таймаут-ветка остаётся), чтобы любой сбой внешнего сервера деградировал в текст, а не убивал граф.
- Поведение нормального пути не менять (контент проходит через `_cap_tool_result` как раньше).
- `plain_agent_tools` в docgen оставить как есть (защитный слой; для менеджерных инструментов станет pass-through).

## Критерии приёмки

- Регресс-тесты: (а) fake-инструмент как у адаптеров (response_format=content_and_artifact + handle_tool_error, возвращающий list) — обёрнутый инструмент возвращает строку/контент без ValueError; (б) нормальный путь с кортежем — контент доходит; (в) таймаут. Полный сьют зелёный.

## Затрагиваемые файлы

`api/mcp/manager.py`, `tests/` (новые тесты wrapper'а).
