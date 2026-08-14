# Каталог промптов Productarium (`refs/prompts/`)

Все тела промптов вынесены сюда и загружаются в рантайме через `api/prompts.py::load_prompt_file`. Редактируй `.md` напрямую — код менять не нужно. Многие промпты hot-reload'ятся через админ-панель (`PUT /api/admin/prompts/{filename}`) для файлов, зарегистрированных в `PROMPT_FILES`.

## Контракт подстановки переменных (КРИТИЧНО)

Есть ДВА способа подстановки. От этого зависит, можно ли использовать в теле литеральные фигурные скобки `{` `}` (например, в JSON/Mermaid/`erDiagram`).

| Способ | Кто использует | Литеральные `{}` | Правило |
|--------|----------------|------------------|---------|
| `str.replace("{var}", value)` | все wiki-секции, structure, compact, expert_agent_*, openapi/asyncapi/testcase/documentation, product_summary, mermaid_repair | **разрешены** | заменяются только точные токены `{var}` из списка ниже; прочие скобки остаются как есть |
| `str.format(**vars)` | deep_research_* и simple_chat | **запрещены** | в теле допустимы ТОЛЬКО токены-плейсхолдеры; любая другая `{`/`}` сломает рендер |
| jinja-переменная | rag_system_prompt (вставляется как `{{system_prompt}}` в `RAG_TEMPLATE`) | избегать `{{`,`{%`,`{#` | держать тело без jinja-разметки |

Плейсхолдеры чувствительны к регистру и должны сохраняться дословно. Полный список ниже.

## Реестр промптов

### Wiki-секции (последовательная генерация, `api/docgen/wiki.py`; подстановка `str.replace`)
Каждая секция строится на ранее сгенерированных через `{previous_content}` (кроме overview). К каждой секции в рантайме добавляется `_verification_guard.md`.

| Файл | Назначение | Плейсхолдеры | Вывод |
|------|-----------|--------------|-------|
| `overview.md` | Общая информация | `{repo_url} {repo_name} {repo_type} {primary_language} {file_count} {main_directories}` | Markdown, 6 разделов + футер провенанса |
| `architecture.md` | Системная архитектура (C4) | `{repo_url} {repo_name} {project_structure} {main_files} {previous_content}` | Markdown + Mermaid |
| `functional.md` | Функциональное описание | `{repo_url} {repo_name} {app_type} {main_modules} {api_endpoints} {previous_content}` | Markdown |
| `technical.md` | Технические детали | `{repo_url} {repo_name} {tech_stack} {config_files} {previous_content}` | Markdown |
| `cicd.md` | CI/CD | `{repo_url} {repo_name} {cicd_files} {docker_files} {config_files} {previous_content}` | Markdown + Mermaid |
| `lld.md` | Low Level Design (шаблон) | `{repo_url} {repo_name} {components} {api_endpoints} {modules} {previous_content}` | Markdown по шаблону |
| `datamodel.md` | Модель данных (шаблон) | `{repo_url} {repo_name} {databases} {entities} {db_config} {previous_content}` | Markdown + ER Mermaid |

### Оркестрация wiki
| Файл | Назначение | Плейсхолдеры | Вывод |
|------|-----------|--------------|-------|
| `structure.md` | Оглавление wiki | `{repo_url} {repo_name} {repo_type} {file_count} {main_directories} {primary_language}` | **строгий JSON** (7 разделов, фиксированные `id`) |
| `compact_generation.md` | Вся документация одним ответом | `{repo_url} {repo_name} {tech_stack} {project_structure}` | Markdown, 7 разделов |

`structure.md`: контракт вывода — единственный JSON-объект с 7 секциями и фиксированными `id` (`overview, architecture, functional, technical, cicd, lld, datamodel`). Парсится downstream — не менять набор/порядок `id`.

### Артефактные документы (`api/docgen/spec.py`; подстановка `str.replace`)
| Файл | Назначение | Плейсхолдеры | Вывод |
|------|-----------|--------------|-------|
| `openapi_doc.md` | Документация REST API из OpenAPI | `{repo_name} {artifact_name} {previous_content} {content}` | Markdown |
| `asyncapi_doc.md` | Документация async API из AsyncAPI | `{repo_name} {artifact_name} {previous_content} {content}` | Markdown + Mermaid |
| `testcase_doc.md` | Документация тест-кейсов | `{repo_name} {artifact_name} {previous_content} {content}` | Markdown |
| `documentation_doc.md` | Причёсывание произвольной документации | `{artifact_name} {content}` | Markdown |

### Чат / исследование / RAG
| Файл | Назначение | Подстановка | Плейсхолдеры |
|------|-----------|-------------|--------------|
| `simple_chat_system_prompt.md` | System prompt чата по репозиторию | `str.format` | `{repo_type} {repo_url} {repo_name} {language_name}` |
| `deep_research_first_iteration.md` | Deep Research, итерация 1 | `str.format` | `{repo_type} {repo_url} {repo_name} {language_name}` |
| `deep_research_intermediate_iteration.md` | Deep Research, промежуточная | `str.format` | `+ {research_iteration}` |
| `deep_research_final_iteration.md` | Deep Research, финал | `str.format` | `{repo_type} {repo_url} {repo_name} {language_name}` |
| `rag_system_prompt.md` | System prompt RAG | jinja-переменная | нет плейсхолдеров |

### Экспертный агент (`api/expert/`; подстановка `str.replace`)
Блоки `<product_knowledge>`, `<conversation_history>`, `<query>` добавляются кодом — в теле их не подставляй.
| Файл | Назначение | Плейсхолдеры |
|------|-----------|--------------|
| `expert_agent_system.md` | System prompt эксперта (ответ инлайн) | `{product_name} {language_name}` |
| `expert_agent_doc.md` | Эксперт в режиме генерации документа | `{product_name} {language_name}` |

### Сервисные
| Файл | Назначение | Подстановка | Плейсхолдеры |
|------|-----------|-------------|--------------|
| `product_summary.md` | Краткое саммари продукта (1 абзац) | `str.replace` | `{product_name} {content}` |
| `mermaid_repair.md` | Починка Mermaid-диаграммы | `str.replace` | `{broken_diagram} {error}` |
| `_verification_guard.md` | Единые правила верификации/провенанса | добавляется конкатенацией | нет плейсхолдеров |

### Recommended-next (НЕ подключён в код)
| Файл | Назначение | Статус |
|------|-----------|--------|
| `knowledge_graph_extraction.md` | Формальный контракт извлечения графа знаний для Cognee | Шаблон, готов к подключению. Не загружается рантаймом (нет в `PROMPT_FILES`). См. `PRODUCTARIUM_IMPLEMENTATION_README.md`. |

## Зависимости стадий (pipeline)
1. `structure.md` → оглавление (JSON).
2. Wiki-секции по порядку: `overview → architecture → functional → technical → cicd → lld → datamodel`; каждая получает `{previous_content}`.
   Либо `compact_generation.md` — всё за один проход (для малых репозиториев/малого контекста).
3. Артефакты (`openapi/asyncapi/testcase/documentation`) — независимые, могут идти параллельно (map).
4. `product_summary.md` — reduce по всем артефактам и страницам знаний.
5. Индексация в Cognee (внутренний cognify) → recall для `expert_agent_*` и `rag_system_prompt`.
6. `mermaid_repair.md` — постобработка любых Mermaid-блоков, не прошедших валидацию.

## Профили контекста
Каждый содержательный промпт задаёт правила для Large и Small контекста: в Small сохраняются все заголовки-разделы, но объём ужимается, таблицы сокращаются до ключевых строк, диаграммы упрощаются. Приоритет — покрытие разделов и корректность ссылок над объёмом.

## Рекомендации по sampling
- Извлечение, документация, JSON-структуры, граф знаний: детерминированно — temperature 0–0.2, top_p ~1.0. Для `structure.md` и `knowledge_graph_extraction.md` — строго 0–0.1.
- Чат/deep research: 0.2–0.4.
- `mermaid_repair.md`: 0 (нужен предсказуемый синтаксис).
- Идеация/креатив: не используется в этом пайплайне.

## Проверка при редактировании
После правки любого промпта запусти валидатор:
```
python tests/validate_prompts.py
```
Он проверяет: наличие/непустоту, сохранность плейсхолдеров, рендерабельность `str.format`-промптов, баланс fenced-блоков, отсутствие битых символов, валидность JSON-примера в `structure.md`.
