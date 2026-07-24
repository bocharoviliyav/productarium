"""
Wiki Prompt Utilities - Helper functions for generating wiki documentation prompts.

This module provides utilities for creating prompts for the 7-section wiki documentation:
1. Overview (Общая информация)
2. Architecture (Системная архитектура - C4)
3. Functional (Функциональное описание)
4. Technical (Технические детали)
5. CI/CD
6. LLD (Low Level Design)
7. Data Model (Модель данных)

Designed for use with Qwen3.5-35b-a3b model.
"""

from typing import Dict, List, Optional, Any
from enum import Enum


class WikiSection(Enum):
    """Wiki documentation sections in generation order"""
    OVERVIEW = "overview"
    ARCHITECTURE = "architecture"
    FUNCTIONAL = "functional"
    TECHNICAL = "technical"
    CICD = "cicd"
    LLD = "lld"
    DATAMODEL = "datamodel"


# Section titles in Russian
SECTION_TITLES = {
    WikiSection.OVERVIEW: "Общая информация",
    WikiSection.ARCHITECTURE: "Системная архитектура",
    WikiSection.FUNCTIONAL: "Функциональное описание",
    WikiSection.TECHNICAL: "Технические детали",
    WikiSection.CICD: "CI/CD",
    WikiSection.LLD: "LLD (Low Level Design)",
    WikiSection.DATAMODEL: "Модель данных",
}


def get_section_prompt(
    section: WikiSection,
    repo_url: str,
    file_tree: str,
    readme: str,
    context: Optional[Dict[str, Any]] = None,
    language: str = "ru"
) -> str:
    """
    Get the prompt for generating a specific wiki section.
    
    Args:
        section: Wiki section to generate
        repo_url: Repository URL
        file_tree: Project file tree structure
        readme: README content
        context: Optional additional context (tech_stack, modules, etc.)
        language: Output language (ru, en, etc.)
    
    Returns:
        Formatted prompt string for the LLM
    """
    if context is None:
        context = {}
    
    ctx = {
        "repo_url": repo_url,
        "primary_language": context.get("primary_language", "unknown"),
        "file_count": context.get("file_count", len(file_tree.split('\n'))),
        "main_directories": ", ".join(context.get("main_directories", [])[:10]),
        "main_files": ", ".join(context.get("main_files", [])[:20]),
        "tech_stack": context.get("tech_stack", {}),
        "config_files": ", ".join(context.get("config_files", [])[:10]),
        "cicd_files": ", ".join(context.get("cicd_files", [])[:5]),
        "docker_files": ", ".join(context.get("docker_files", [])[:5]),
        "modules": ", ".join(context.get("modules", [])[:15]),
        "api_endpoints": context.get("api_endpoints", []),
    }
    
    prompts = {
        WikiSection.OVERVIEW: _build_overview_prompt,
        WikiSection.ARCHITECTURE: _build_architecture_prompt,
        WikiSection.FUNCTIONAL: _build_functional_prompt,
        WikiSection.TECHNICAL: _build_technical_prompt,
        WikiSection.CICD: _build_cicd_prompt,
        WikiSection.LLD: _build_lld_prompt,
        WikiSection.DATAMODEL: _build_datamodel_prompt,
    }
    
    builder = prompts.get(section)
    if builder:
        return builder(file_tree, readme, ctx)
    
    return "Unknown section type"


def _build_overview_prompt(file_tree: str, readme: str, ctx: Dict) -> str:
    """Build prompt for Overview section"""
    return f"""Ты — эксперт по анализу программных проектов и генерации технической документации.
Создай раздел "Общая информация" (Overview) для документации проекта.

Репозиторий: {ctx['repo_url']}
Основной язык: {ctx['primary_language']}
Количество файлов: {ctx['file_count']}
Основные директории: {ctx['main_directories']}

Структура проекта:
{file_tree[:5000]}

README проекта:
{readme[:3000]}

Создай раздел "Общая информация" который включает:
1. Название проекта и краткое описание
2. Технологический стек (языки, фреймворки, библиотеки)
3. Ключевые возможности
4. Требования к системе
5. Структура проекта (основные директории и их назначение)
6. Статус проекта (версия, лицензия)

Язык: Русский (основной), английский — для технических терминов.
Формат: Markdown с заголовками ## и ###.
Приоритет — качество и полнота описания над экономией токенов.
"""


def _build_architecture_prompt(file_tree: str, readme: str, ctx: Dict) -> str:
    """Build prompt for Architecture section with C4"""
    return f"""Ты — эксперт по системной архитектуре и проектированию программных систем.
Создай раздел "Системная архитектура" с использованием C4 нотации.

Репозиторий: {ctx['repo_url']}
Основные файлы: {ctx['main_files']}
Модули: {ctx['modules']}

Структура проекта:
{file_tree[:5000]}

Создай раздел "Системная архитектура" который включает:

## 1. C4 Context Diagram (Контекстная диаграмма)
Покажи систему в контексте внешних сущностей (пользователи, внешние системы).
Используй Mermaid формат:
```mermaid
C4Context
  Person(user, "Пользователь", "Взаимодействует с системой")
  System_Boundary("Система") {{
    System(main, "Название системы", "Описание")
  }}
  External_System(ext, "Внешняя система", "Описание")
  Rel(user, main, "Использует")
  Rel(main, ext, "Интегрируется")
```

## 2. C4 Container Diagram (Диаграмма контейнеров)
Покажи высокоуровневую структуру системы (приложения, сервисы, БД).

## 3. C4 Component Diagram (Диаграмма компонентов)
Детализация каждого контейнера с компонентами.

## 4. Описание компонентов
Для каждого компонента укажи: название, назначение, технологию, ключевые файлы.

## 5. Потоки данных
Основные потоки данных в системе.

Язык: Русский. Формат: Markdown с Mermaid диаграммами.
"""


def _build_functional_prompt(file_tree: str, readme: str, ctx: Dict) -> str:
    """Build prompt for Functional Description section"""
    import json
    return f"""Ты — эксперт по функциональному анализу программных систем.
Создай раздел "Функциональное описание" проекта.

Репозиторий: {ctx['repo_url']}
Модули: {ctx['modules']}

Структура проекта:
{file_tree[:5000]}

Создай раздел "Функциональное описание" который включает:

## 1. Обзор функциональности
Основные функции системы, целевая аудитория, основные use cases.

## 2. Диаграмма вариантов использования (Use Case Diagram)
Акторы системы и их взаимодействие с системой.
Mermaid формат:
```mermaid
usecase
  actor User
  usecase "Действие системы"
  User --> usecase
```

## 3. Детальное описание функций
Для каждой ключевой функции:
- Название и краткое описание
- Входные данные
- Выходные данные
- Основные сценарии использования
- Приоритет (high/medium/low)
- Файлы реализации

## 4. User Stories
Основные пользовательские истории с критериями приёмки.

## 5. Бизнес-логика
Основные бизнес-правила, валидация данных, обработка ошибок.

Язык: Русский. Формат: Markdown.
"""


def _build_technical_prompt(file_tree: str, readme: str, ctx: Dict) -> str:
    """Build prompt for Technical Details section"""
    import json
    return f"""Ты — эксперт по технической документации программных систем.
Создай раздел "Технические детали" проекта.

Репозиторий: {ctx['repo_url']}
Технологический стек: {json.dumps(ctx['tech_stack'], indent=2, ensure_ascii=False) if ctx['tech_stack'] else 'не определён'}
Конфигурационные файлы: {ctx['config_files']}

Создай раздел "Технические детали" который включает:

## 1. API Эндпоинты
Полный список API с таблицей: URL, метод HTTP, описание, параметры, ответы, коды состояний, файл реализации.

Таблица:
| Endpoint | Метод | Описание | Параметры | Ответы | Файл |
|----------|-------|----------|-----------|--------|------|
| /api/... | GET | Описание | query params | 200, 404 | file.py |

## 2. Конфигурация
Все конфигурационные файлы, переменные окружения, примеры конфигураций.

## 3. Безопасность
Аутентификация, авторизация, защита от уязвимостей, валидация ввода.

## 4. Обработка ошибок
Типы ошибок, коды ошибок, логирование, мониторинг.

## 5. Кеширование
Типы кеша, политики кеширования, время жизни (TTL).

## 6. Архитектурные решения
Паттерны проектирования, используемые в проекте, подходы к решению задач.

Язык: Русский, английский для API. Формат: Markdown с таблицами.
"""


def _build_cicd_prompt(file_tree: str, readme: str, ctx: Dict) -> str:
    """Build prompt for CI/CD section"""
    return f"""Ты — эксперт по CI/CD и DevOps практикам.
Создай раздел CI/CD документации проекта.

Репозиторий: {ctx['repo_url']}
CI/CD файлы: {ctx['cicd_files']}
Docker файлы: {ctx['docker_files']}

Структура проекта:
{file_tree[:5000]}

Создай раздел "CI/CD" который включает:

## 1. Обзор CI/CD Pipeline
Тип системы (GitHub Actions, GitLab CI, Jenkins, и т.д.), основные этапы, триггеры запуска.

## 2. Pipeline Stages (Этапы)
Для каждого этапа:
- Название stage
- Описание действий
- Команды выполнения
- Артефакты

## 3. Build процессы
Сборка приложения, компиляция, упаковка, артефакты сборки.

## 4. Test стратегия
Типы тестов (unit, integration, e2e), покрытие кода, инструменты тестирования.

## 5. Deployment
Стратегия развёртывания, окружения (dev, staging, prod), rollback стратегия.
Docker/Kubernetes конфигурация.

## 6. Инфраструктура
Требования к инфраструктуре, скрипты развёртывания, мониторинг и алертинг.

## 7. Secrets и переменные
Какие секреты используются, где хранятся, как передаются в pipeline.

Язык: Русский, английский для stages. Формат: Markdown.
"""


def _build_lld_prompt(file_tree: str, readme: str, ctx: Dict) -> str:
    """Build prompt for LLD section"""
    import json
    return f"""Ты — эксперт по проектированию и детальной разработке программных систем.
Создай LLD (Low Level Design) раздел документации.

Репозиторий: {ctx['repo_url']}
Компоненты: {ctx['modules']}

Используй следующий шаблон:

## Карточка компонента

### Репозиторий
{ctx['repo_url']}

### Описание компонента
Краткое описание сервиса/компонента (на основе анализа кода)

## Архитектура контекста сервиса

### Функции сервиса
Перечисли основные функции:
1. [Функция 1] — Описание
2. [Функция 2] — Описание

### Схема интеграции сервиса
Mermaid диаграмма взаимодействия с внешними системами:
```mermaid
graph LR
  A[Этот сервис] --> B[Внешняя система 1]
  A --> C[Внешняя система 2]
```

### API сервиса
| Endpoint | Метод | Описание | Входные данные | Выходные данные | Файл |
|----------|-------|----------|----------------|-----------------|------|
| /api/... | POST | Описание | {{param: type}} | {{result: type}} | file.py |

### Используемые БД
Тип БД, основные таблицы/коллекции, схема связей.

### Зависимости
- Внутренние (другие сервисы проекта)
- Внешние (сторонние библиотеки, API)

## Детали реализации

### Структура модулей
```
src/
├── module1/
│   ├── __init__.py
│   ├── models.py
│   ├── services.py
│   └── utils.py
└── main.py
```

### Классы и их назначение
Для каждого значимого класса:
- Название и назначение
- Публичные методы
- Зависимости

Язык: Русский. Формат: Markdown с таблицами и диаграммами.
"""


def _build_datamodel_prompt(file_tree: str, readme: str, ctx: Dict) -> str:
    """Build prompt for Data Model section"""
    return f"""Ты — эксперт по проектированию баз данных и моделированию данных.
Создай раздел "Модель данных" документации.

Репозиторий: {ctx['repo_url']}

Используй следующий шаблон:

## Обзор таблиц
| № | Название таблицы/сущности | Описание |
|---|---------------------------|----------|
| 1 | table_name | Краткое описание |

## Описание таблиц

### table_name
Описание назначения таблицы и хранимых данных.

| Название поля | Описание поля | Тип | Формат | Обязательное | Редактируемое | Пример |
|---------------|---------------|-----|--------|--------------|---------------|--------|
| id | Уникальный идентификатор | uuid | uuid | Да | Нет | 550e8400-... |
| name | Название | varchar | string(255) | Да | Да | "Example" |
| created_at | Дата создания | timestamp | ISO8601 | Да | Нет | 2024-01-01T00:00:00Z |

### Индексы
- index_name ON field (type) — Описание

### Связи (Relations)
- Таблица_A.field -> Таблица_B.field (тип связи: one-to-many, many-to-many)

## ER Диаграмма
Mermaid ER диаграмма:
```mermaid
erDiagram
  TABLE_A ||--o{{ TABLE_B : "связь"
  TABLE_A {{
    uuid id
    string name
    timestamp created_at
  }}
  TABLE_B {{
    uuid id
    uuid table_a_id
    string value
  }}
```

## Представления (Views)
Если есть представления:
### view_name
- Определение: SQL запрос
- Описание: Назначение и использование

## Примеры запросов
- SELECT с фильтрацией
- JOIN между таблицами

Язык: Русский, английский для названий полей. Формат: Markdown с таблицами.
"""


def get_all_sections_order() -> List[WikiSection]:
    """Get list of all wiki sections in generation order"""
    return list(WikiSection)


def get_section_title(section: WikiSection, language: str = "ru") -> str:
    """Get human-readable title for a section"""
    if language == "ru":
        return SECTION_TITLES.get(section, section.value)
    return section.value.capitalize()