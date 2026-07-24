# Задача: структура (оглавление) wiki-документации

Ты — инженер по документации. Построй структуру wiki-документации проекта и верни её строго в формате JSON по схеме ниже. Адаптируй акценты под тип проекта, но сохрани фиксированный набор разделов и их идентификаторы.

## Входные данные
- Репозиторий: `{repo_url}` (`{repo_name}`)
- Тип репозитория: `{repo_type}`
- Количество файлов: `{file_count}`
- Основные директории: `{main_directories}`
- Язык основного кода: `{primary_language}`

## Границы доказательности
Опирайся только на предоставленные данные репозитория. Не выдумывай разделы вне фиксированной схемы. Адаптация допускается только в текстовых `title`/`description` и в акцентах, но не в наборе `id`.

## Фиксированный набор разделов (менять нельзя)
1. `overview` — Общая информация
2. `architecture` — Системная архитектура
3. `functional` — Функциональное описание
4. `technical` — Технические детали
5. `cicd` — CI/CD
6. `lld` — LLD (Low Level Design)
7. `datamodel` — Модель данных

## Контракт вывода (СТРОГО)
Верни ТОЛЬКО один JSON-объект в fenced-блоке ```json. Без текста до и после. Схема:

```json
{
  "id": "wiki_structure",
  "title": "Документация {repo_name}",
  "description": "Полная документация проекта {repo_name}",
  "sections": [
    {
      "id": "overview",
      "title": "Общая информация",
      "pages": ["project-overview", "tech-stack", "features", "requirements", "structure"]
    },
    {
      "id": "architecture",
      "title": "Системная архитектура",
      "pages": ["c4-context", "c4-container", "c4-component", "components", "data-flows"]
    },
    {
      "id": "functional",
      "title": "Функциональное описание",
      "pages": ["overview", "use-cases", "functions", "user-stories", "business-logic"]
    },
    {
      "id": "technical",
      "title": "Технические детали",
      "pages": ["api-endpoints", "configuration", "security", "error-handling", "caching"]
    },
    {
      "id": "cicd",
      "title": "CI/CD",
      "pages": ["pipeline-overview", "build", "test", "deployment", "infrastructure"]
    },
    {
      "id": "lld",
      "title": "LLD (Low Level Design)",
      "pages": ["component-card", "service-context", "service-api", "implementation", "integrations"]
    },
    {
      "id": "datamodel",
      "title": "Модель данных",
      "pages": ["tables-overview", "tables", "er-diagram", "indexes", "relations"]
    }
  ]
}
```

## Правила валидности JSON
- Ровно 7 объектов в `sections` с указанными `id` в этом порядке.
- Двойные кавычки для всех ключей и строк; без комментариев и висячих запятых.
- `pages` — непустые массивы строк-слагов (kebab-case).
- Никакого текста вне JSON-блока.

## Адаптация под тип проекта (только в тексте, не в id)
- Backend/API — усиль `technical` и `lld` (интеграции, API).
- Frontend — добавь акцент на UI-компоненты в `description`/`pages` в пределах схемы.
- Библиотека — сделай упор на использование в `overview`/`functional`.
- Есть БД — `datamodel` критичен.

## Проверки качества (перед выдачей)
- JSON валиден и парсится; 7 разделов с корректными `id` и порядком.
- Нет текста вне JSON; нет комментариев и висячих запятых.
