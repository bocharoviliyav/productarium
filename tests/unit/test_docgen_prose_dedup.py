"""Tests for the opener-duplicate guard (2.4): stem morphology, opener
extraction/span, Jaccard similarity over stems, and the
``enforce_unique_openers`` pass (one repair attempt, verbatim-tail
preservation, reuse/placeholder rules, soft failures)."""
import asyncio

import pytest

from api.docgen import prose_dedup as pd
from api.utils.russian_stem import stem_phrase


# ============================================================================
# russian_stem
# ============================================================================
class TestRussianStem:
    def test_inflections_collapse_to_same_stem(self):
        assert stem_phrase("Модель данных") == stem_phrase("Модели данных")

    def test_adjective_paradigm_collapses(self):
        assert stem_phrase("Внешний сервис") == stem_phrase("Внешнего сервиса")

    def test_noun_case_paradigm_collapses(self):
        assert stem_phrase("Система задач") == stem_phrase("Системы задач")

    def test_latin_identifiers_untouched(self):
        # Латиница не склоняется — только casefold.
        assert stem_phrase("sendSmsCode") == "sendsmscode"

    def test_empty_is_empty(self):
        assert stem_phrase("") == ""
        assert stem_phrase(None) == ""


# ============================================================================
# opener_of / opener_span
# ============================================================================
class TestOpenerOf:
    def test_first_sentence(self):
        assert pd.opener_of("Первое предложение. Второе.") == "Первое предложение."

    def test_abbreviation_joins_next_sentence(self):
        # «т.» и «т. е.» — сокращения, не концы предложения.
        assert pd.opener_of("т. е. Система работает. Дальше.") == (
            "т. е. Система работает."
        )

    def test_cap_on_text_without_sentence_end(self):
        # Ровно кап или кап-1 (если 400-й символ — пробел, strip его снимает).
        n = len(pd.opener_of("Без точки " * 200))
        assert pd.OPENER_CAP_CHARS - 1 <= n <= pd.OPENER_CAP_CHARS

    def test_empty(self):
        assert pd.opener_of("") == ""
        assert pd.opener_of("   ") == ""


class TestOpenerSpan:
    def test_span_ends_at_sentence_boundary(self):
        text = "Первое. Второе предложение хвоста."
        opener, end = pd.opener_span(text)
        assert opener == "Первое."
        assert end == len("Первое.")
        assert text[end] == " "

    def test_span_matches_opener_of_for_plain_text(self):
        text = "Зачин секции здесь. Хвост секции."
        opener, end = pd.opener_span(text)
        assert opener == pd.opener_of(text)
        assert text[:end] == opener

    def test_giant_tail_excluded_from_span(self):
        # Безграничный хвост длиннее потолка НЕ входит в спан: замена «всего
        # текста» одним предложением съела бы секцию целиком. Границы здесь —
        # сокращения в нижнем регистре («см.», «стр.»), заглавное «См.»
        # эвристика сокращением не считает.
        text = "см. стр. 5 " + ("слово " * 300).strip()
        opener, end = pd.opener_span(text)
        assert opener == "см. стр."
        assert end == len("см. стр.")
        assert text[end] == " "  # разделитель перед хвостом не съеден
        # Сравнение при этом всё ещё видит хвост (капнутый зачин).
        assert "слово" in pd.opener_of(text)

    def test_empty(self):
        assert pd.opener_span("") == ("", 0)


# ============================================================================
# opener_similarity / find_repeated_opener
# ============================================================================
class TestOpenerSimilarity:
    def test_identical_is_one(self):
        s = "Одинаковый зачин секции."
        assert pd.opener_similarity(s, s) == 1.0

    def test_inflected_duplicate_above_threshold(self):
        a = "Система управляет задачами и проектами."
        b = "Система управляет задачами и напоминаниями."
        assert pd.opener_similarity(a, b) >= pd.OPENER_SIMILARITY_THRESHOLD

    def test_different_below_threshold(self):
        a = "Сервис обрабатывает заказы через очередь сообщений."
        b = "Репозиторий содержит модульные тесты и фикстуры."
        assert pd.opener_similarity(a, b) < pd.OPENER_SIMILARITY_THRESHOLD

    def test_empty_is_zero(self):
        assert pd.opener_similarity("", "Любой зачин.") == 0.0
        assert pd.opener_similarity("Любой зачин.", "") == 0.0


class TestFindRepeatedOpener:
    def test_below_threshold_none(self):
        assert pd.find_repeated_opener(
            "Порядок и хаос.", [("x", "Таблицы и индексы.")]
        ) is None

    def test_picks_max_score(self):
        prior = [
            ("a", "Система управляет задачами."),
            ("b", "Система управляет задачами и проектами."),
        ]
        hit = pd.find_repeated_opener("Система управляет задачами и проектами.", prior)
        assert hit is not None and hit[0] == "b"

    def test_tie_picks_first_in_canonical_order(self):
        prior = [
            ("a", "Система управляет задачами."),
            ("b", "Система управляет задачами."),
        ]
        hit = pd.find_repeated_opener("Система управляет задачами.", prior)
        assert hit is not None and hit[0] == "a"


# ============================================================================
# opener_dedup_enabled
# ============================================================================
class TestOpenerDedupEnabled:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("DOCGEN_OPENER_DEDUP", raising=False)
        assert pd.opener_dedup_enabled() is True

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("DOCGEN_OPENER_DEDUP", value)
        assert pd.opener_dedup_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "off", "no", "FALSE", ""])
    def test_disabling_values(self, monkeypatch, value):
        monkeypatch.setenv("DOCGEN_OPENER_DEDUP", value)
        assert pd.opener_dedup_enabled() is False


# ============================================================================
# enforce_unique_openers
# ============================================================================
class TestEnforceUniqueOpeners:
    ORDER = ["overview", "architecture", "functional"]

    @staticmethod
    def _dup_sections():
        return {
            "overview": "Система управляет задачами. Хвост первой секции.",
            "architecture": "Система управляет задачами. Хвост второй секции.",
            "functional": "Совсем другое вступление про функции.",
        }

    def test_duplicate_reported_without_repair(self):
        sections = self._dup_sections()
        report = asyncio.run(pd.enforce_unique_openers(sections, self.ORDER))
        assert set(report) == {"architecture"}
        entry = report["architecture"]
        assert entry["similar_to"] == "overview"
        assert entry["repaired"] is False
        assert entry["similarity"] == 1.0
        # Без ремонта текст не трогается.
        assert sections["architecture"].startswith("Система управляет задачами.")

    def test_repair_replaces_only_opener(self):
        sections = self._dup_sections()
        calls = []

        async def repair(prompt):
            calls.append(prompt)
            return "Архитектура построена из модулей."

        report = asyncio.run(pd.enforce_unique_openers(sections, self.ORDER, repair))
        assert len(calls) == 1
        assert report["architecture"]["repaired"] is True
        # Зачин заменён, хвост сохранён дословно.
        assert sections["architecture"].startswith("Архитектура построена из модулей.")
        assert "Хвост второй секции." in sections["architecture"]

    def test_repair_prompt_carries_both_openers(self):
        # Инфлектированный повтор (не дословный): промпт обязан нести ОБА
        # зачина — без текста соседа «не повторяй зачин» — не ремонт, а новая
        # случайная выборка (порт правила _dedup_prompt).
        sections = {
            "overview": "Система управляет задачами и проектами. Хвост.",
            "architecture": "Система управляет задачами и напоминаниями. Хвост.",
        }
        calls = []

        async def repair(prompt):
            calls.append(prompt)
            return "Архитектура построена из модулей."

        report = asyncio.run(pd.enforce_unique_openers(
            sections, ["overview", "architecture"], repair,
            titles={"overview": "Обзор", "architecture": "Архитектура"},
        ))
        assert report["architecture"]["repaired"] is True
        assert "Система управляет задачами и проектами." in calls[0]
        assert "Система управляет задачами и напоминаниями." in calls[0]

    def test_repaired_opener_becomes_prior(self):
        sections = {
            "overview": "Система управляет задачами. Один.",
            "architecture": "Система управляет задачами. Два.",
            "functional": "Система управляет задачами. Три.",
        }

        async def repair(prompt):
            if "Архитектура" in prompt:
                return "Архитектура описывает слои системы."
            return "Функциональность покрывает сценарии использования."

        report = asyncio.run(pd.enforce_unique_openers(
            sections, self.ORDER, repair,
            titles={"overview": "Обзор", "architecture": "Архитектура",
                    "functional": "Функциональность"},
        ))
        # Третья секция конфликтует с ПЕРВОЙ (максимум счёта), а не с
        # отремонтированной второй.
        assert report["functional"]["similar_to"] == "overview"
        assert report["architecture"]["repaired"] is True
        assert report["functional"]["repaired"] is True

    def test_repair_candidate_still_conflicting_rejected(self):
        sections = self._dup_sections()

        async def repair(prompt):
            # Кандидат всё ещё повторяет зачин соседа — не принимается.
            return "Система управляет задачами и напоминаниями."

        report = asyncio.run(pd.enforce_unique_openers(sections, self.ORDER, repair))
        assert report["architecture"]["repaired"] is False
        assert sections["architecture"].startswith("Система управляет задачами.")

    def test_placeholder_sections_skipped(self):
        ph = "_(Содержимое раздела временно недоступно. Вы можете перезапустить генерацию.)_"
        sections = {"overview": ph, "architecture": ph}
        report = asyncio.run(pd.enforce_unique_openers(
            sections, ["overview", "architecture"], placeholder=ph,
        ))
        assert report == {}

    def test_empty_sections_skipped(self):
        sections = {"overview": "", "architecture": "   "}
        report = asyncio.run(pd.enforce_unique_openers(
            sections, ["overview", "architecture"],
        ))
        assert report == {}

    def test_repair_not_called_for_ineligible(self):
        sections = self._dup_sections()
        calls = []

        async def repair(prompt):
            calls.append(prompt)
            return "Уникальный зачин."

        report = asyncio.run(pd.enforce_unique_openers(
            sections, self.ORDER, repair, repair_eligible=lambda sid: False,
        ))
        assert calls == []
        assert report["architecture"]["repaired"] is False

    def test_repair_exception_is_soft(self):
        sections = self._dup_sections()

        async def repair(prompt):
            raise RuntimeError("llm down")

        report = asyncio.run(pd.enforce_unique_openers(sections, self.ORDER, repair))
        assert report["architecture"]["repaired"] is False
        assert sections["architecture"].startswith("Система управляет")

    def test_non_string_reply_rejected(self):
        sections = self._dup_sections()

        async def repair(prompt):
            return None

        report = asyncio.run(pd.enforce_unique_openers(sections, self.ORDER, repair))
        assert report["architecture"]["repaired"] is False

    def test_oversized_reply_rejected(self):
        sections = self._dup_sections()

        async def repair(prompt):
            return "слово " * 300

        report = asyncio.run(pd.enforce_unique_openers(sections, self.ORDER, repair))
        assert report["architecture"]["repaired"] is False

    def test_quoted_and_bulleted_reply_cleaned(self):
        sections = self._dup_sections()

        async def repair(prompt):
            return "- «Архитектура: слои и зависимости.»"

        report = asyncio.run(pd.enforce_unique_openers(sections, self.ORDER, repair))
        assert report["architecture"]["repaired"] is True
        assert sections["architecture"].startswith("Архитектура: слои и зависимости.")
