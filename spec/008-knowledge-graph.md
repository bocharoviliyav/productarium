# SPEC-008: Графовая база знаний (JSONB) и публичный MCP-доступ

Статус: planned · Волна 5 · Приоритет: средний

## Проблема/идея

Память сейчас — чанки pgvector с изоляцией по `product_id` (`KnowledgeChunkORM`), термины неоднозначны между продуктами (пример: RMS = роботы в одном продукте, ритейл — в другом); нет структурированного доступа к знаниям для внешних агентов. Решение из grilling: граф знаний в JSONB + граф-дизамбигуация поверх ОДНОГО векторного namespace на продукт; публичный доступ через inbound MCP.

## Решение

1. **Модель данных**: таблица `knowledge_graphs` (`product_id` PK→FK products, `graph` JSONB, `version`, `updated_at`).
   - Узлы: `term` (нормализованный термин, описание, aliases), `page` (artifact_type + artifact_id + page_id), у каждого узла стабильный key для upsert;
   - Рёбра: `term→page` (вес/контекст упоминания), `term—term` (synonym/part_of/related), `page→page` (`depends_on` — переиспользуется SPEC-010);
   - Размер графа ограничен бюджетом (кап на узлы/рёбра, LRU-обрезка слабых рёбер).
2. **Построение/обновление**: в конвейере индексации памяти после docgen — батчевая LLM-экстракция терминов из страниц (бюджет по `api/config/timeout.py`, `DOCGEN_LLM_CONCURRENCY`), merge по ключу узла (upsert, удаление умерших терминов/рёбер при регенерации страницы). Инкрементально: только изменённые страницы (синергия с SPEC-010).
3. **Retrieval**: запрос → резолв терминов запроса по графу продукта (lexical + embeddings aliases) → векторный поиск в namespace продукта с graph-boost чанков связанных страниц. Дизамбигуация: термин и его страницы существуют только внутри продукта ⇒ RMS-кейс решается scope'ом; в `api/memory/resolver.py` добавить graph-aware путь, старый путь — fallback.
4. **Публичный доступ (inbound MCP, `api/mcp/inbound.py`)**:
   - tool `knowledge_search(query, product_id)` → релевантные чанки + цитаты (existing citation-формат);
   - tool `knowledge_graph(product_id, focus_term?)` → подграф (термин + соседние узлы) или граф целиком с капом;
   - `product_id` обязателен; изоляция как у чанков (фильтр по product_id); авторизация — существующие механизмы inbound MCP (API-токены/сессия).
5. **Эксплуатация**: в админ-панели Memory — статистика графов (узлы/рёбра по продуктам) и принудительный reindex.

## Не-цели

- Кросс-продуктовые рёбра, визуальный редактор графа, отдельные векторные подпространства на узлы (решение grilling — один namespace на продукт).

## Критерии приёмки

- RMS-сценарий: два продукта с термином RMS разных смыслов → поиск в каждом продукте возвращает только своё значение (герметичный тест).
- Внешний MCP-клиент получает оба tools со scope-изоляцией; попытка без product_id → ошибка валидации.
- Тесты экстракции (мок-LLM) и merge/upsert графа; регресс по скорости поиска отсутствует.

## Затрагиваемые файлы

`api/models.py` (новая ORM), `api/memory/pgvector_backend.py`, `api/memory/resolver.py`, `api/docgen/_common.py` (пост-индексация), `api/mcp/inbound.py`, `api/routers/knowledge.py`, `api/routers/admin.py` (Memory), `src/app/admin/page.tsx`, миграция `create_all`/bootstrap, `tests/unit/test_knowledge_graph*.py`, `AGENTS.md` (Knowledge Graph).
