# Каталог промптов Productarium (`refs/prompts/`)

Все тела промптов вынесены сюда и загружаются в рантайме через `api/prompts.py::load_prompt_file`. Редактируй `.md` напрямую — код менять не нужно. Файлы, зарегистрированные в `PROMPT_FILES`, hot-reload'ятся через админ-панель (`PUT /api/admin/prompts/{filename}`).

Инвентарный инвариант: ключи `PROMPT_FILES` == все `*.md` в этой директории минус `README.md` (проверяется тестом `tests/unit/test_prompts.py`).

## Контракт подстановки переменных (КРИТИЧНО)

Основной способ подстановки — `str.replace("{var}", value)`: литеральные фигурные скобки `{` `}` (JSON/Mermaid/`erDiagram`) **разрешены**, заменяются только точные токены `{var}`. Плейсхолдеры чувствительны к регистру и должны сохраняться дословно.

## Реестр промптов

### Docgen-пайплайн кодовой базы (`api/docgen/codebase.py`)
| Файл | Назначение | Плейсхолдеры | Вывод |
|------|-----------|--------------|-------|
| `docgen_sections.md` | Контракты всех 7 секций вики (`<section id="...">`) | — (внутри контрактов) | парсится в `SECTION_PROMPTS` |
| `docgen_subpages.md` | Контракты подстраниц (`<subpage id="...">`: functional_item / technical_item / datamodel_item) | — (внутри контрактов) | парсится в `SUBPAGE_CONTRACTS` |
| `docgen_decomposer.md` | Планировщик декомпозиции секций на юниты-подстраницы | `{repo_brief} {sections_list} {section_hints}` | **строгий JSON** (`functional`/`technical`/`datamodel` + slug/title/focus/kind) |
| `docgen_router.md` | Роутер: подсказки files/focus по секциям | `{repo_brief} {sections_list}` | **строгий JSON** |
| `docgen_orchestrator.md` | Системный промпт deepagents-оркестратора | `{repo_name} {sections_list} {reused_sections}` | dispatch-отчёт |
| `docgen_agent_system.md` | Системный промпт юнит-сабагента (инструменты, grounding, язык) | `{language_name}` | Markdown одной страницы |
| `docgen_agent_section.md` | Задача юнит-сабагента (родитель или подстраница) | `{repo_url} {repo_name} {section_id} {section_title} {repo_brief} {sections_list} {section_hints} {siblings_list} {section_instruction}` | Markdown |
| `docgen_judge.md` | LLM-judge верификации секций | `{source_evidence} {draft_section}` | **строгий JSON** (consistent/issues) |

Секции с подстраницами: `functional` (возможности, ≤10), `technical` (эндпоинты/джобы/интеграции/справочники, ≤12), `datamodel` (слои данных, ≤6). Дети генерируются раньше родителя; юниты обмениваются контекстом через notes-workspace (`notes_read`/`notes_write`).

### Спецификации (`api/docgen/spec.py`)
| Файл | Назначение | Плейсхолдеры |
|------|-----------|--------------|
| `spec_agent_system.md` | Системный промпт агента-обогатителя | `{language_name}` и др. |
| `spec_enrich_task.md` | Задача обогащения скелета | `{skeleton}` и др. |

### Экспертный агент (`api/expert/`)
Блоки `<product_knowledge>`, `<conversation_history>`, `<query>` добавляются кодом — в теле их не подставляй.
| Файл | Назначение | Плейсхолдеры |
|------|-----------|--------------|
| `expert_agent_system.md` | System prompt эксперта (ответ инлайн) | `{product_name} {language_name}` |

### Deep Research (`api/expert/deep_research.py`)
| Файл | Назначение | Плейсхолдеры |
|------|-----------|--------------|
| `deep_research_planner.md` | Планировщик итераций | `{query} {product_name} {language_name}` |
| `deep_research_researcher.md` | Итерация исследования | `{plan} {product_name} {language_name}` |
| `deep_research_synthesizer.md` | Синтез финального ответа | `{query} {product_name} {language_name}` |

### База данных (`api/docgen/database.py`)
| Файл | Назначение | Плейсхолдеры |
|------|-----------|--------------|
| `database_doc.md` | Документирование БД по MCP-интроспекции | `{database_name} {dsn_masked} {schema_dump} {skeleton} {language_name}` |

### Сервисные
| Файл | Назначение | Подстановка | Плейсхолдеры |
|------|-----------|-------------|--------------|
| `product_summary.md` | Краткое саммари продукта (1 абзац) | `str.replace` | `{product_name} {content}` |
| `openapi_doc.md` / `asyncapi_doc.md` | Документация API из спек | `str.replace` | `{repo_name} {artifact_name} {previous_content} {content}` |
| `mermaid_repair.md` | Починка Mermaid-диаграммы | `str.replace` | `{broken_diagram} {error}` |
| `_verification_guard.md` | Единые правила верификации/провенанса | добавляется конкатенацией | нет плейсхолдеров |

## Зависимости стадий (codebase docgen)
1. Роутер (`docgen_router.md`) → подсказки по секциям (JSON).
2. Декомпозер (`docgen_decomposer.md`) → план подстраниц (JSON) → notes-workspace (`repo_brief.md`, `decomposition.json`).
3. Юнит-сабагенты (`docgen_agent_system.md` + `docgen_agent_section.md` + контракт секции/подстраницы): дети раньше родителя; каждый пишет `summary_<unit>.md` в notes-workspace.
4. Верификация каждого юнита (`verification.py` + `docgen_judge.md` + `mermaid_repair.md`).
5. `product_summary.md` — саммари продукта по артефактам и узлам знаний.

## Проверка при редактировании
После правки любого промпта:
```
poetry -C api run sh -c 'cd "$(git rev-parse --show-toplevel)" && python -m pytest -q tests/unit/test_prompts.py'
```
Тесты проверяют: инвентарь (`PROMPT_FILES` == файлы на диске), наличие всех блоков `<section>`/`<subpage>`, обязательные плейсхолдеры, отсутствие обёртки языка в контрактах.
