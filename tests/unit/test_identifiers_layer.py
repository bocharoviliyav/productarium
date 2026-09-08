"""Unit tests for the identifier layer (fork "layer A" port, Wave 1.3).

Covers:
- ``split_identifier``: the fork's camel/Pascal/snake/dot cases.
- ``extract_identifiers``: backticked names first, compound bare tokens,
  prose noise filtered, dedupe + limit.
- ``build_embed_payload`` / ``build_query_payload``: chunk prefix kept
  verbatim, bounded suffix, no-identifier text unchanged.
- pgvector integration: the text sent to the embedder differs from the
  stored chunk; the stored chunk stays verbatim; the query embeds with the
  human form of identifier tokens in the question.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.utils.identifiers import (
    build_embed_payload,
    build_query_payload,
    extract_identifiers,
    split_identifier,
)


# ===========================================================================
# split_identifier (fork port)
# ===========================================================================
class TestSplitIdentifier:
    def test_camel_case(self):
        assert split_identifier("sendSmsCode") == "send sms code"

    def test_pascal_case(self):
        assert split_identifier("SmsGatewayFacade") == "sms gateway facade"

    def test_acronym_boundary(self):
        assert split_identifier("APIGateway") == "api gateway"
        assert split_identifier("parseHTTPResponse") == "parse http response"

    def test_snake_and_dots(self):
        assert split_identifier("CHATBOT_TASK") == "chatbot task"
        assert split_identifier("ru.x.TaskService") == "ru x task service"

    def test_path_like(self):
        assert split_identifier("src/main.py") == "src main py"

    def test_kebab(self):
        assert split_identifier("docker-compose") == "docker compose"

    def test_empty_and_plain(self):
        assert split_identifier("") == ""
        assert split_identifier(None) == ""
        assert split_identifier("users") == "users"


# ===========================================================================
# extract_identifiers
# ===========================================================================
class TestExtractIdentifiers:
    def test_backticked_first(self):
        ids = extract_identifiers("The table `user_accounts` holds rows.")
        assert ids == ["user_accounts"]

    def test_backticked_path(self):
        ids = extract_identifiers("See `src/api/router.py` for routes.")
        assert "src/api/router.py" in ids

    def test_bare_compound_tokens(self):
        ids = extract_identifiers("calls sendSmsCode via SmsGatewayFacade")
        assert "sendSmsCode" in ids
        assert "SmsGatewayFacade" in ids

    def test_prose_noise_filtered(self):
        ids = extract_identifiers("The quick brown fox jumps over rows")
        assert ids == []  # single words are not identifiers

    def test_dedupe(self):
        ids = extract_identifiers("`user_accounts` and user_accounts again")
        assert ids.count("user_accounts") == 1

    def test_limit(self):
        text = " ".join(f"token{i}Value" for i in range(50))
        assert len(extract_identifiers(text, limit=5)) == 5


# ===========================================================================
# payload builders
# ===========================================================================
class TestPayloadBuilders:
    def test_chunk_kept_verbatim_as_prefix(self):
        chunk = "Handles delivery via `sendSmsCode` in the facade."
        payload = build_embed_payload(chunk)
        assert payload.startswith(chunk)
        assert "sendSmsCode = send sms code" in payload

    def test_no_identifiers_unchanged(self):
        chunk = "Просто прозa без идентификаторов тут."
        assert build_embed_payload(chunk) == chunk

    def test_heading_included(self):
        chunk = "## Схема базы данных\n\nТаблица `order_items` хранит позиции."
        payload = build_embed_payload(chunk)
        assert "Схема базы данных" in payload
        assert "order_items = order items" in payload

    def test_suffix_bounded(self):
        chunk = " ".join(f"`table_{i}_name`" for i in range(40))
        payload = build_embed_payload(chunk, budget=100)
        suffix = payload.split("\n", 1)[1]
        assert len(suffix) <= 160  # budget + separator slack

    def test_query_payload_human_form(self):
        q = "как работает sendSmsCode в шлюзе?"
        payload = build_query_payload(q)
        assert payload.startswith(q)
        assert "sendSmsCode = send sms code" in payload

    def test_plain_query_unchanged(self):
        assert build_query_payload("как устроен сервис?") == "как устроен сервис?"


# ===========================================================================
# pgvector integration: embed text != stored chunk
# ===========================================================================
class TestPgVectorPayloadIntegration:
    def _prepare(self, isolated_db):
        from api.models import ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_a", name="P", description=""))
            db.commit()
        finally:
            db.close()

    def test_embedded_text_differs_from_stored_chunk(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        from api.models import KnowledgeChunkORM

        self._prepare(isolated_db)
        captured = []

        async def _fake_embed(texts):
            captured.extend(texts)
            return [[float(i)] * 4 for i in range(len(texts))]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        content = "Раздел про `user_accounts`: таблица и её индексы. " * 20
        n = asyncio.run(be.index(
            content, "prod_a", source_type="codebase", source_id="cb_1",
        ))
        assert n > 0
        assert captured, "embedder saw no texts"
        # Every embedded text carries the identifier suffix...
        assert any("user_accounts = user accounts" in t for t in captured)
        # ...but NO stored chunk contains the synthetic suffix.
        db = isolated_db.SessionLocal()
        try:
            rows = db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.source_id == "cb_1"
            ).all()
            assert rows
            for row in rows:
                assert "user_accounts = user accounts" not in row.content
                assert "user_accounts" in row.content  # the prose itself stays
        finally:
            db.close()

    def test_query_embeds_human_form(self, monkeypatch):
        from api.memory import pgvector_backend as pb

        captured = []

        async def _fake_embed(texts):
            captured.extend(texts)
            return [[0.0] * 4]

        async def _fake_search(self, qvec, product_id, top_k):
            return "result"

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)
        # _cosine_search is a backend METHOD — patch it on the class.
        monkeypatch.setattr(pb.PgVectorMemoryBackend, "_cosine_search", _fake_search)
        monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

        be = pb.PgVectorMemoryBackend()
        out = asyncio.run(be.query("как работает sendSmsCode?", "prod_a"))
        assert out == "result"
        assert any("send sms code" in t for t in captured)
