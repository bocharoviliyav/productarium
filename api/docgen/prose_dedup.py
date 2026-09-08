"""Сторож межсекционного повтора зачинов (порт из форка DeepWiki, пункт 2.4).

Семь секций документации кодовой базы пишутся независимыми сабагентами, и
типовой дефект такой генерации — одинаковые вступления: каждый сабагент
начинает со своего «Система, с которой мы имеем дело, представляет собой…».
Изолированная проверка секции (``verify_section``) этого не видит и увидеть
не может: она проверяет текст без знания о соседях.

Порог 0.30 в форке не назначен, а ИЗМЕРЕН на живом корпусе: по всем 171 паре
зачинов медиана сходства 0.000, 90-й перцентиль 0.167, а между «истинный
повтор» (0.391) и «законно разные зачины» (0.222) лежит пропасть в 0.17.
Порог стоит посреди неё. Значение помечено КАЛИБРУЕМЫМ: при желании его
пересчитывают на собственных корпусах.

Контракт (как ремонт нарратива в форке): при конфликте — РОВНО одна попытка
LLM-ремонта; кандидат принят только если конфликт действительно снят; иначе
исходный текст не трогается, а дефект попадает в provenance-отчёт с меткой
``opener-duplicate``. Выключатель — ``DOCGEN_OPENER_DEDUP`` (как
``DOCGEN_JUDGE_ENABLED``: чтение на вызове, не на импорте).
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from api.utils.russian_stem import stem_phrase

logger = logging.getLogger(__name__)

#: Обоснование числа — в docstring модуля: замер, а не глазомер. Калибруемый.
OPENER_SIMILARITY_THRESHOLD = 0.30

#: Потолок зачина. Абзац без единой точки — законный ответ модели, и без
#: потолка он уезжал бы в промпт ремонта ЦЕЛИКОМ (в форке проба гейта дала
#: зачин в 11 999 знаков и промпт, выросший с 2 307 до 26 603 знаков).
OPENER_CAP_CHARS = 400

#: Потолок строки-кандидата из ремонта: один зачин, а не новый раздел.
_OPENER_REPLACEMENT_CAP_CHARS = 600

#: Точка после короткого слова в нижнем регистре — это сокращение, а не конец
#: предложения: «т. е.», «стр. 5», «рис. 2», «см. выше». Без этого признака
#: «т. е. Система…» давал зачин «т.» (два дословно одинаковых зачина
#: сходились на 0.0 — повтор проходил мимо сторожа), а «стр. 5 …» и
#: «стр. 9 …» давали «стр.» и «стр.» — сходство 1.0 и ложный ремонт на
#: пустом месте.
#:
#: Списка сокращений не заводим: он бесконечен и его пришлось бы вести.
#: Признак ловит весь класс разом, а цена ошибки мала и в обе стороны:
#: предложение, законно кончившееся коротким словом («Это так.»), склеится со
#: следующим — сравнивать чуть больше текста сторожу не мешает.
_ABBREV_TAIL_LEN = 3

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
_ANY_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def opener_dedup_enabled() -> bool:
    """True, если только ``DOCGEN_OPENER_DEDUP`` не выключил сторож.

    Тот же паттерн, что у ``judge_enabled``: чтение на вызове, чтобы ops-тумблер
    и monkeypatch в тестах работали без рестарта процесса.
    """
    return (os.environ.get("DOCGEN_OPENER_DEDUP", "true") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ends_with_abbreviation(text: str) -> bool:
    r"""Кончается ли кусок сокращением, а не концом предложения.

    Признак — последнее слово: короткое и в нижнем регистре («стр.», «т.»,
    «рис.»). Заглавная буква не в счёт: инициал «А.» и настоящее короткое
    предложение неразличимы, а ошибиться в эту сторону дешевле.

    Проверять «а кончается ли кусок точкой вообще» не нужно: разбиение идёт
    по границе `(?<=[.!?])\s`, поэтому куском без точки на конце бывает только
    последний — а после него продолжать нечем.
    """
    words = _ANY_WORD.findall(text)
    if not words:
        return False
    tail = words[-1]
    return len(tail) <= _ABBREV_TAIL_LEN and not tail[0].isupper()


def opener_of(prose: str) -> str:
    """Первое непустое предложение прозы (для сравнения с соседями).

    Берётся ДО сборки markdown: в собранной странице сверху стоят заголовки и
    якоря, у каждой страницы свои, и сторож, глядя на них, не сработал бы
    никогда.

    «Первое предложение» считается не по первой точке: кусок, кончающийся
    сокращением, приклеивается к следующему. Результат ограничен
    ``OPENER_CAP_CHARS``: текст без единой точки иначе становится зачином
    целиком.
    """
    text = (prose or "").strip()
    if not text:
        return ""
    opener = ""
    for piece in _SENTENCE_END.split(text):
        opener = f"{opener} {piece}".strip() if opener else piece.strip()
        if len(opener) >= OPENER_CAP_CHARS or not _ends_with_abbreviation(opener):
            break
    return opener[:OPENER_CAP_CHARS].strip()


def opener_span(prose: str) -> Tuple[str, int]:
    """Зачин и безопасная граница его замены в символах lstripped-текста.

    ``opener_of`` возвращает НОРМАЛИЗОВАННЫЙ зачин (разделители предложений
    схлопнуты в один пробел), который может не быть точной подстрокой
    исходника. Для точечной замены спан считается по ЦЕЛЫМ кускам до границы
    предложения, чтобы хвост секции не задевался вовсе.

    Хвост без границы предложения длиннее потолка НЕ входит в спан: замена
    «всего текста» одним предложением могла бы съесть секцию целиком.
    """
    text = (prose or "").strip()
    if not text:
        return "", 0
    opener = ""
    pos = 0
    last_end = 0
    for m in _SENTENCE_END.finditer(text):
        piece = text[pos:m.start()]
        opener = f"{opener} {piece}".strip() if opener else piece.strip()
        if len(opener) >= OPENER_CAP_CHARS or not _ends_with_abbreviation(opener):
            return opener[:OPENER_CAP_CHARS].strip(), m.start()
        pos = m.end()
        last_end = m.start()
    rest = text[pos:]
    if rest and len(rest) <= OPENER_CAP_CHARS:
        opener = f"{opener} {rest}".strip() if opener else rest.strip()
        return opener[:OPENER_CAP_CHARS].strip(), len(text)
    # Хвост-гигант без границ: зачин кончается на последней поглощённой
    # границе — ДО её разделителя, чтобы замена не съедала пробел перед
    # хвостом. Сравнение остаётся на стороне opener_of (капнутый зачин).
    return opener[:OPENER_CAP_CHARS].strip(), (last_end if opener else 0)


def _stems(text: str) -> frozenset:
    return frozenset(stem_phrase(m.group(0)) for m in _WORD.finditer(text or ""))


def opener_similarity(a: str, b: str) -> float:
    """Жаккар по ОСНОВАМ слов: склонение не должно разводить один и тот же
    зачин («Система … задачами» против «Системы … задачам»).

    Пустая сторона даёт 0.0, а не деление на ноль: секция без прозаического
    зачина ни с чем не совпадает.
    """
    sa, sb = _stems(a), _stems(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_repeated_opener(
    opener: str, prior: Sequence[Tuple[str, str]]
) -> Optional[Tuple[str, str, float]]:
    """Ближайший из уже принятых зачинов, если он ближе порога.

    Выбирается МАКСИМУМ сходства, при равенстве — первый по каноническому
    порядку (том, в котором секции обходит генератор): иначе отчёт называл бы
    разные секции от прогона к прогону при одинаковом входе.

    Пустой зачин отдельной ветки не требует: ``opener_similarity`` возвращает
    для него 0.0, и до порога он не доходит.
    """
    best: Optional[Tuple[str, str, float]] = None
    for page_id, other in prior:
        score = opener_similarity(opener, other)
        if score < OPENER_SIMILARITY_THRESHOLD:
            continue
        if best is None or score > best[2]:
            best = (page_id, other, score)
    return best


def _clean_replacement(reply: Any) -> str:
    """Кандидат на новый зачин: одна строка, без кавычек/буллетов, с потолком.

    Не-строка (None/bytes) — пустой кандидат: ремонт не состоялся, а не мусор
    в начало секции.
    """
    if not isinstance(reply, str):
        return ""
    text = " ".join(reply.split())
    if text.startswith(("- ", "* ", "1. ", "1) ")):
        text = text[2:].strip()
    text = text.strip("\"'«»").strip()
    if not text or len(text) > _OPENER_REPLACEMENT_CAP_CHARS:
        return ""
    return text


def _opener_repair_prompt(
    *, own_title: str, own_opener: str, other_title: str, other_opener: str
) -> str:
    """Промпт ремонта несёт ОБА зачина (порт правила ``_dedup_prompt``):
    без текста соседа «не повторяй зачин» — не ремонт, а новая случайная
    выборка."""
    return (
        "Два раздела одной документации открываются почти одинаковым "
        "предложением.\n\n"
        f"Зачин раздела «{other_title}»: {other_opener}\n"
        f"Зачин раздела «{own_title}»: {own_opener}\n\n"
        f"Напиши одно новое первое предложение для раздела «{own_title}»: оно "
        "должно сразу называть предмет ЭТОГО раздела, а не давать общее "
        "вступление о системе, и не должно повторять зачин соседнего раздела. "
        "Верни только это предложение — без кавычек, без нумерации и без "
        "markdown-разметки."
    )


async def _try_repair_opener(
    content: str,
    prior: Sequence[Tuple[str, str]],
    repair: Callable[[str], Awaitable[str]],
    *,
    own_title: str,
    other_title: str,
    other_opener: str,
) -> Optional[str]:
    """Одна попытка ремонта: заменить ТОЛЬКО зачин, остальной текст не трогать.

    Возврат None = ремонт не состоялся (сбой вызова, пустой/дырявый кандидат
    или конфликт всё ещё на месте) — вызывающий оставляет исходный текст и
    пишет warning. Любое исключение ремонта глотается: годный текст секции уже
    на руках, и сбой ремонта не должен его отнимать.
    """
    stripped = content.lstrip()
    opener, span_end = opener_span(stripped)
    if not opener or span_end <= 0:
        return None
    prompt = _opener_repair_prompt(
        own_title=own_title,
        own_opener=opener,
        other_title=other_title,
        other_opener=other_opener,
    )
    try:
        reply = await repair(prompt)
    except Exception:
        return None
    replacement = _clean_replacement(reply)
    if not replacement:
        return None
    lead = content[: len(content) - len(stripped)]
    candidate = f"{lead}{replacement}{stripped[span_end:]}"
    new_opener = opener_of(candidate)
    if not new_opener:
        return None
    # Кандидат принят только если конфликт действительно снят — то же правило,
    # что и ремонт нарратива в форке.
    if find_repeated_opener(new_opener, prior) is not None:
        return None
    return candidate


async def enforce_unique_openers(
    sections: Dict[str, str],
    section_order: Sequence[str],
    repair: Optional[Callable[[str], Awaitable[str]]] = None,
    *,
    repair_eligible: Optional[Callable[[str], bool]] = None,
    placeholder: str = "",
    titles: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Пройти секции в каноническом порядке и развести одинаковые зачины.

    Мутирует ``sections`` на месте при УСПЕШНОМ ремонте (заменён только зачин
    конфликтующей секции). Возвращает отчёт ``{sid: {...}}`` для provenance:

    - ``similar_to`` — sid секции, с которой совпал зачин;
    - ``similarity`` — Жаккар по основам (округлён);
    - ``repaired`` — снят ли конфликт единственной попыткой LLM-ремонта.

    ``repair_eligible(sid)`` разрешает ремонт (в codebase-потоке это «секция
    не переиспользована verbatim»: reuse-контент нельзя переписывать — LLM
    недетерминирован, и чекпойнт-стабильность repeat-прогонов важнее косметики).
    Переиспользованная секция всё равно служит «prior» и всё равно попадает в
    отчёт, если конфликтует. Плейсхолдеры и пустые секции не участвуют вовсе:
    два одинаковых заглушки — не повтор, а один и тот же маркер отказа.
    """
    report: Dict[str, Dict[str, Any]] = {}
    prior: List[Tuple[str, str]] = []
    resolved_titles = titles or {}
    for sid in section_order:
        content = (sections.get(sid) or "").strip()
        if not content or (placeholder and content == placeholder.strip()):
            continue
        opener = opener_of(content)
        conflict = find_repeated_opener(opener, prior)
        if conflict is None:
            prior.append((sid, opener))
            continue
        other_sid, other_opener, score = conflict
        entry: Dict[str, Any] = {
            "similar_to": other_sid,
            "similarity": round(score, 3),
            "repaired": False,
        }
        if repair is not None and (repair_eligible is None or repair_eligible(sid)):
            new_content = await _try_repair_opener(
                content,
                prior,
                repair,
                own_title=resolved_titles.get(sid) or sid,
                other_title=resolved_titles.get(other_sid) or other_sid,
                other_opener=other_opener,
            )
            if new_content is not None:
                sections[sid] = new_content
                entry["repaired"] = True
                opener = opener_of(new_content)
        if not entry["repaired"]:
            logger.warning(
                "opener-duplicate [%s]: зачин совпадает с «%s» (score=%.2f)",
                sid, other_sid, score,
            )
        report[sid] = entry
        # Эффективный зачин (отремонтированный или нет) становится prior для
        # следующих секций — как принятый текст в форке.
        prior.append((sid, opener))
    return report
