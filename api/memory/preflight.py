"""Embedder preflight (port of the fork's ``embedder_diagnostics``, design §7).

Одна проба эмбеддинга ДО долгой docgen-генерации: часовой прогон, результат
которого физически не сможет проиндексироваться (эмбеддер недоступен, модель
не загружена, размерность сменилась), должен падать через секунды — с
человекочитаемой причиной и машинным префиксом ``EMBEDDER_ERROR:`` в
``job["error"]``, а не молча терять recall в фоне.

Адаптация под единый OpenAI-compatible стек продуктариума: провайдер один
(langchain ``OpenAIEmbeddings`` из ``api.tools.embedder``), поэтому переключателя
типов нет — классификация отказа идёт по тексту ошибки (auth / model_missing /
unreachable / misconfigured). Модуль ничего не знает про HTTP и джобы: он
только пробует эмбеддер и сверяет размерность, поэтому одинаков во всех
точках вызова.

Небезопасных состояний нет: на не-pgvector инсталляции (SQLite-фолбэк,
герметичные тесты) предполёт молча пропускается — чанки всё равно не лягут в
косинус-поиск.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Машинный код диагноза (design §7.7): его читает UI/скрипты, а не человек —
# иначе русский диагноз без признака превращается в «неизвестная ошибка».
EMBEDDER_ERROR_PREFIX = "EMBEDDER_ERROR: "

_PROBE_TEXT = "productarium embedder preflight"


class EmbedderUnavailable(ValueError):
    """Наследование от ValueError намеренное: ветка обработки джоба уже ловит
    ValueError и кладёт его текст в ``job["error"]`` / ``indexing_message``,
    поэтому диагноз попадает в существующий путь без правки структуры."""

    def __init__(self, message: str, *, reason: str, detail: str = "") -> None:
        super().__init__(message)
        self.reason = reason
        self.detail = detail


def _classify(detail: str) -> str:
    """Классификация отказа по тексту ошибки (порт ``_classify`` форка)."""
    low = (detail or "").lower()
    if (
        "401" in low or "403" in low or "unauthorized" in low
        or "api key" in low or "expiredtoken" in low
        or "unrecognizedclient" in low or "invalidsignature" in low
        or "credential" in low or "forbidden" in low
    ):
        return "auth"
    if "not found" in low or "404" in low or "does not exist" in low:
        return "model_missing"
    return "unreachable"


def _resolve_probe_timeout() -> float:
    """Таймаут пробы из реестра таймаутов.

    Специализированного ``EMBEDDER_*`` ключа в реестре нет; ``provider_test``
    (``PROVIDER_TEST_TIMEOUT_SECONDS``, по умолчанию 15 c) — это ровно тот же
    сценарий «проверить, что провайдер отвечает», что и у кнопки «Test» в
    админке, поэтому проба живёт под ним.
    """
    try:
        from api.config.timeout import resolve_provider_test_timeout

        value = resolve_provider_test_timeout()
        if value and value > 0:
            return float(value)
    except Exception:  # pragma: no cover - registry is import-safe
        pass
    return 15.0


def _fail(reason: str, message: str, detail: str = "") -> None:
    """Структурированная строка в журнал + Raise с человекочитаемым текстом."""
    try:
        from api.config.settings import get_setting

        model = get_setting("models.embedder.model") or "(default)"
    except Exception:  # pragma: no cover - settings store is import-safe
        model = "(default)"
    logger.error(
        "embedder preflight failed: reason=%s model=%s detail=%s",
        reason, model, detail or "-",
    )
    raise EmbedderUnavailable(message, reason=reason, detail=detail)


def _pgvector_ready() -> bool:
    """Проба имеет смысл только там, где чанки реально лягут в косинус-поиск."""
    try:
        from api.memory.pgvector_backend import _is_pgvector_capable

        return bool(_is_pgvector_capable())
    except Exception:  # pragma: no cover - defensive
        return False


def _existing_chunk_dimension(product_id: str) -> Optional[int]:
    """Размерность самого свежего чанка продукта (pgvector ``vector_dims``).

    None = чанков нет / не pgvector / запрос не удался — сверка пропускается
    (проба всё равно состоится; это метаданные, а не требование).
    """
    try:
        from sqlalchemy import text

        from api.db import SessionLocal

        with SessionLocal() as db:
            row = db.execute(
                text(
                    "SELECT vector_dims(embedding) FROM knowledge_chunks "
                    "WHERE product_id = :pid AND embedding IS NOT NULL "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"pid": product_id},
            ).scalar()
        return int(row) if row else None
    except Exception as e:  # pragma: no cover - DB-dependent
        logger.debug("dimension precheck skipped for %s: %s", product_id, e)
        return None


async def _probe(timeout: Optional[float] = None) -> int:
    """Одна синхронная проба эмбеддинга в отдельном потоке под таймаутом.

    Возвращает размерность вектора. Любой отказ → ``EmbedderUnavailable`` с
    классифицированной причиной.
    """
    if timeout is None:
        timeout = _resolve_probe_timeout()

    def _do() -> List[List[float]]:
        from api.tools.embedder import get_embedder

        embedder = get_embedder()
        vectors = embedder.embed_documents([_PROBE_TEXT])
        out: List[List[float]] = []
        for vec in vectors or []:
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            out.append([float(x) for x in vec])
        if len(out) != 1:
            raise ValueError(f"embedder returned {len(out)} vectors for 1 text")
        return out

    try:
        vectors = await asyncio.wait_for(asyncio.to_thread(_do), timeout=timeout)
    except asyncio.TimeoutError:
        _fail(
            "unreachable",
            f"Эмбеддер не ответил за {timeout:.0f} с. Проверьте, что локальный "
            f"OpenAI-compatible сервер запущен и базовый URL верен "
            f"(LOCAL_OPENAI_BASE_URL / настройка models.embedder.base_url).",
        )
    except EmbedderUnavailable:
        raise
    except Exception as e:
        detail = str(e)
        reason = _classify(detail)
        if reason == "auth":
            _fail(
                reason,
                "Эмбеддер отклонил запрос: ключ отсутствует или недействителен "
                "(LOCAL_OPENAI_API_KEY / models.embedder.api_key).",
                detail=detail,
            )
        if reason == "model_missing":
            _fail(
                reason,
                "Сервер не знает модель эмбеддингов: проверьте, что модель "
                "загружена (настройка models.embedder.model) и имя совпадает "
                "с экспортированной сервером.",
                detail=detail,
            )
        if "no embedder configuration" in detail.lower():
            _fail(
                "misconfigured",
                "Конфигурация эмбеддера не задана (embedder.json / models.embedder.*).",
                detail=detail,
            )
        _fail(
            reason,
            f"Эмбеддер недоступен: {detail}. Проверьте, что локальный "
            f"OpenAI-compatible сервер запущен и доступен с этого хоста.",
            detail=detail,
        )

    vector = vectors[0] if vectors else []
    if not vector:
        _fail(
            "unreachable",
            "Эмбеддер вернул пустой вектор на пробном запросе — индексация "
            "получила бы нули. Проверьте модель эмбеддингов.",
        )
    return len(vector)


def _check_dimension(probe_dim: int, product_id: str) -> None:
    """Сверка размерности пробы с уже лежащими чанками продукта (design §7.5).

    Расхождение означает, что модель эмбеддингов сменилась: старые чанки и
    новые векторы несравнимы в косинус-пространстве, а pgvector-колонка уже
    запинена под старую размерность — вставка нового батча упадёт в самом
    конце многочасового прогона. Ловим это до старта.
    """
    existing = _existing_chunk_dimension(product_id)
    if existing is None:
        return
    if probe_dim != existing:
        _fail(
            "dimension_mismatch",
            f"Размерность эмбеддера ({probe_dim}) не совпадает с уже "
            f"проиндексированными чанками продукта ({existing}). Верните "
            f"прежнюю модель эмбеддингов либо переиндексируйте продукт "
            f"(удалите его чанки), иначе новый прогон не сможет записать "
            f"векторы.",
            detail=f"probe={probe_dim} existing={existing}",
        )


async def preflight_embedder(product_id: str) -> Optional[int]:
    """Предполёт эмбеддера перед долгой docgen-генерацией.

    Возвращает размерность пробы (None, когда предполёт неприменим — не
    pgvector-инсталляция). Поднимает :class:`EmbedderUnavailable` с
    классифицированной причиной; вызывающий джоб-воркер превращает её в
    ``ValueError`` с префиксом ``EMBEDDER_ERROR:`` в ``job["error"]``, ни
    один чанк не пишется.
    """
    if not _pgvector_ready():
        logger.debug("embedder preflight skipped: pgvector not active")
        return None
    probe_dim = await _probe()
    _check_dimension(probe_dim, product_id)
    return probe_dim
