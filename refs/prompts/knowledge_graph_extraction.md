# Задача: извлечение графа знаний для Cognee (контракт)

> СТАТУС: recommended-next. Этот промпт НЕ подключён в текущий код (его нет в `api/prompts.py::PROMPT_FILES`). Это формальный, готовый к подключению контракт извлечения графа. Cognee сейчас строит граф собственным `cognify` (Pydantic/instructor); подключение этого промпта описано в `PRODUCTARIUM_IMPLEMENTATION_README.md`. Подстановка предполагается через `str.replace` (литеральные `{}` в JSON-примерах ниже безопасны).

Ты — экстрактор знаний. По предоставленным документам продукта и коду построй фрагмент графа знаний: узлы (entities) и связи (relations) со стабильными ID, провенансом, уверенностью и временными метаданными. Только то, что подтверждается предоставленным контекстом.

## Входные данные (при подключении)
- Продукт: `{product_name}` (`{repo_name}`)
- Пакет источника (chunk): `{artifact_name}` — путь/имя источника для провенанса
- Содержимое для извлечения:
{content}

## Границы доказательности
Опирайся только на предоставленный контекст (см. единые правила верификации). Каждый узел и каждая связь ДОЛЖНЫ иметь провенанс (источник). Не создавай узлы/связи без опоры на текст. Разделяй наблюдаемые факты и предположения (см. `status`).

## Онтология

### Типы узлов (node `type`)
Продуктовые: `Product, Goal, Metric, Stakeholder, Persona, Need, JTBD, Requirement, Feature, Capability, UserJourney, UseCase, BusinessRule`.
Доменные/архитектурные: `Domain, BoundedContext, Service, Component, Module, API, Endpoint, Event, Workflow`.
Данные/код: `DataEntity, Field, RepositoryFile, Symbol, Dependency`.
Инфраструктура/эксплуатация: `InfrastructureResource, Environment, Deployment, SLO, Alert, Test`.
Управление/знания: `Risk, Decision, Assumption, Constraint, GlossaryTerm, Document`.

### Типы связей (relation `type`)
`DEPENDS_ON, PART_OF, IMPLEMENTS, EXPOSES, CONSUMES, PRODUCES, PUBLISHES, SUBSCRIBES_TO, READS, WRITES, STORES_IN, RELATED_TO, DERIVED_FROM, VALIDATES, DEPLOYED_TO, MONITORED_BY, OWNED_BY, SATISFIES, CONFLICTS_WITH, PRECEDES, REFERENCES`.

## Правила идентификаторов и полей
- `id` — стабильный, детерминированный, вида `type:canonical_slug` в нижнем регистре (например, `endpoint:post-/api/users`, `module:api-rag`). Один и тот же смысл → один и тот же `id` между запусками (идемпотентность).
- `confidence` — число 0.0–1.0.
- `status` — `observed` (подтверждён контекстом) или `proposed` (обоснованное предположение; тогда `confidence` ≤ 0.6).
- Провенанс: `source` с `file` (путь) и, при наличии, `lines` ("start-end"); `evidence` — короткая цитата/основание.
- Временные метаданные: `observed_at` (ISO-дата, если известна из контекста; иначе `null`).
- `content_hash` — хэш нормализованного `description` для дедупликации (если среда позволяет; иначе `null`).

## Контракт вывода (СТРОГИЙ JSON)
Верни ТОЛЬКО один JSON-объект в fenced-блоке ```json, без текста вокруг. Соблюдай схему и порядок ключей. Массивы упорядочивай детерминированно (по `id`).

```json
{
  "schema_version": "1.0",
  "source": {"artifact": "artifact_name", "product": "product_name"},
  "nodes": [
    {
      "id": "endpoint:post-/api/users",
      "type": "Endpoint",
      "name": "POST /api/users",
      "aliases": [],
      "description": "Создаёт пользователя.",
      "status": "observed",
      "confidence": 0.95,
      "tags": ["api"],
      "source": {"file": "src/api.py", "lines": "42-58"},
      "evidence": "@router.post('/api/users')",
      "observed_at": null,
      "content_hash": null
    }
  ],
  "edges": [
    {
      "id": "rel:endpoint:post-/api/users|IMPLEMENTS|module:api-users",
      "type": "IMPLEMENTS",
      "source_id": "endpoint:post-/api/users",
      "target_id": "module:api-users",
      "description": "Эндпоинт реализован в модуле users.",
      "status": "observed",
      "confidence": 0.9,
      "source": {"file": "src/api.py", "lines": "42-58"},
      "evidence": "определение обработчика в модуле users",
      "observed_at": null
    }
  ],
  "contradictions": [],
  "gaps": [],
  "validation": {
    "node_count": 1,
    "edge_count": 1,
    "dangling_edges": 0,
    "duplicate_ids": 0,
    "self_loops": 0,
    "missing_provenance": 0
  }
}
```

## Правила качества графа (проверь перед выдачей)
- Нет висячих связей: `source_id` и `target_id` каждой связи присутствуют в `nodes`.
- Нет дубликатов `id` среди узлов и среди связей; при повторном упоминании — объединяй (entity resolution), не плоди дубли.
- Нет петель (`source_id` == `target_id`), если это не осмысленно.
- Комбинация типов узлов для связи допустима по смыслу (например, `Endpoint IMPLEMENTS Module`, а не `Field DEPLOYED_TO Persona`).
- У каждого узла/связи есть провенанс; при отсутствии источника — не создавай ассерцию, вынеси в `gaps`.
- Противоречия между источниками фиксируй в `contradictions` (оба источника), не выбирай произвольно.
- Значения `validation.*` соответствуют фактическому содержимому.

## Ограниченные батчи и возобновляемость
- Обрабатывай ОДИН переданный chunk за вызов; не пытайся охватить весь репозиторий.
- Держи вывод в разумных пределах (ориентир: ≤ ~80 узлов и ≤ ~120 связей на батч); при превышении — оставь наиболее уверенные и укажи остаток в `gaps`.
- Детерминированный порядок массивов обеспечивает стабильный merge между батчами.

## Ремонт JSON (для моделей, плохо следующих формату)
Если не удаётся выдать валидный JSON с первого раза — самопроверь и переиздай ТОЛЬКО исправленный JSON-объект. Без Markdown вне ```json, без комментариев, без висячих запятых, только двойные кавычки.
