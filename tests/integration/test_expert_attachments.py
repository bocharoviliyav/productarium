"""Integration tests for expert chat attachments (upload / download / ask).

Covers the ``api.expert.attachments`` conversion helpers (UTF-8 passthrough,
the 50k cap, binary rejection, HTML-escaped prompt blocks), the upload
endpoint contract (metadata, 400/413/501 paths), download (sanitized
Content-Disposition, 404, non-admin owner scoping) and the ask wiring: the
runner query is augmented with ``<attachment>`` blocks while the persisted
transcript keeps the RAW query and surfaces attachment chips instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.conftest import build_test_client  # noqa: E402


@pytest.fixture
def client(isolated_db, monkeypatch):
    import api.auth.deps as deps
    import api.routers.expert as expert_router
    from api.utils.rate_limit import reset_rate_limits

    reset_rate_limits()
    monkeypatch.setattr(deps, "AUTH_PROVIDER", "none")
    app, client = build_test_client(isolated_db, [expert_router])
    return client


def _seed(db_mod, product_id: str = "prod_1"):
    from api.models import ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name="Acme"))
        db.commit()


def _upload(client, *files):
    """POST files as (name, bytes, mime) tuples to the attachments endpoint."""
    return client.post(
        "/api/products/prod_1/ask/attachments",
        files=[("files", (name, data, mime)) for name, data, mime in files],
    )


def _ev(content: str):
    from api.expert.types import ExpertStreamEvent

    return ExpertStreamEvent("content", content)


# --- conversion module --------------------------------------------------------
class TestConversion:
    def test_utf8_passthrough_and_cap(self):
        from api.expert.attachments import MAX_ATTACHMENT_CHARS, convert_attachment

        text = "hello attachment"
        assert convert_attachment(text.encode("utf-8"), "notes.txt") == text
        # Works via markitdown OR the UTF-8 passthrough; the cap holds either way.
        assert len(convert_attachment(b"x" * 99_000, "big.txt")) == MAX_ATTACHMENT_CHARS

    def test_binary_raises_value_error(self):
        from api.expert.attachments import convert_attachment

        # No UTF-16 BOM (markitdown would legitimately convert that) — raw
        # non-UTF-8 garbage that no converter accepts.
        with pytest.raises(ValueError):
            convert_attachment(b"\x00\x01\x02\x03\xff", "blob.bin")

    def test_blocks_escape_filename(self):
        from api.expert.attachments import build_attachment_blocks

        class _Row:
            filename = 'evil".txt>'
            content_md = " body "

        rendered = build_attachment_blocks([_Row()])
        assert rendered == '<attachment name="evil&quot;.txt&gt;">\nbody\n</attachment>'

    def test_upload_max_bytes_env(self, monkeypatch):
        import api.expert.attachments as att

        monkeypatch.setenv("UPLOAD_MAX_BYTES", "123")
        assert att.upload_max_bytes() == 123
        monkeypatch.delenv("UPLOAD_MAX_BYTES")
        assert att.upload_max_bytes() == 50 * 1024 * 1024


# --- upload --------------------------------------------------------------------
class TestUpload:
    def test_upload_returns_metadata_and_persists(self, client, isolated_db):
        _seed(isolated_db)
        resp = _upload(client, ("notes.txt", b"deploy notes", "text/plain"))
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        meta = body[0]
        assert meta["filename"] == "notes.txt"
        assert meta["size_bytes"] == len(b"deploy notes")
        assert meta["content_chars"] == len("deploy notes")
        assert meta["id"].startswith("atch_")

        from api.models import ChatAttachmentORM

        with isolated_db.SessionLocal() as db:
            row = db.get(ChatAttachmentORM, meta["id"])
            assert row is not None
            # The default test admin is not persisted -> unowned row.
            assert row.user_id is None
            assert row.content_md == "deploy notes"
            assert row.product_id == "prod_1"

    def test_too_many_files_400(self, client):
        resp = _upload(client, *[(f"f{i}.txt", b"x", "text/plain") for i in range(6)])
        assert resp.status_code == 400
        assert "At most 5" in resp.json()["detail"]

    def test_empty_file_400(self, client):
        resp = _upload(client, ("empty.txt", b"", "text/plain"))
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Uploaded file is empty"

    def test_binary_501(self, client):
        resp = _upload(
            client, ("blob.bin", b"\x00\x01\x02\x03\xff", "application/octet-stream")
        )
        assert resp.status_code == 501

    def test_oversized_413(self, client, monkeypatch):
        monkeypatch.setenv("UPLOAD_MAX_BYTES", "10")
        resp = _upload(client, ("big.txt", b"x" * 40, "text/plain"))
        assert resp.status_code == 413


# --- download -------------------------------------------------------------------
class TestDownload:
    def test_download_returns_markdown_with_sanitized_name(self, client):
        up = _upload(client, ("my notes v1.txt", b"the content", "text/plain")).json()
        resp = client.get(f"/api/products/prod_1/ask/attachments/{up[0]['id']}")
        assert resp.status_code == 200
        assert resp.text == "the content"
        assert resp.headers["content-type"].startswith("text/markdown")
        # Unsafe filename chars are replaced, never echoed into the header.
        assert resp.headers["content-disposition"] == (
            'attachment; filename="my_notes_v1.txt"'
        )

    def test_unknown_attachment_404(self, client):
        assert (
            client.get("/api/products/prod_1/ask/attachments/atch_none").status_code
            == 404
        )

    def test_other_users_attachment_404_for_non_admin(self, client, isolated_db):
        import api.routers.expert as expert_router
        from api.models import ChatAttachmentORM, UserORM

        with isolated_db.SessionLocal() as db:
            db.add(UserORM(id="u_a", username="a", role="user"))
            db.add(UserORM(id="u_b", username="b", role="user"))
            db.commit()

        # Transient ORMs suffice (role + id are read in-memory); the upload
        # links user_id because the persisted row exists.
        def _as(uid):
            return lambda: UserORM(id=uid, username=uid, role="user")

        client.app.dependency_overrides[expert_router.get_current_user] = _as("u_a")
        up = _upload(client, ("mine.txt", b"secret", "text/plain")).json()
        atch_id = up[0]["id"]

        with isolated_db.SessionLocal() as db:
            assert db.get(ChatAttachmentORM, atch_id).user_id == "u_a"

        client.app.dependency_overrides[expert_router.get_current_user] = _as("u_b")
        assert (
            client.get(f"/api/products/prod_1/ask/attachments/{atch_id}").status_code
            == 404
        )
        ask = client.post(
            "/api/products/prod_1/ask",
            json={"query": "q", "attachment_ids": [atch_id]},
        )
        assert ask.status_code == 400
        assert ask.json()["detail"] == "Unknown attachment id"


# --- ask wiring -------------------------------------------------------------------
class TestAskAugmentation:
    def test_runner_query_augmented_transcript_keeps_raw_query(
        self, client, isolated_db, monkeypatch
    ):
        import api.routers.expert as expert_router

        _seed(isolated_db)
        up = _upload(client, ("notes.txt", b"deploy notes", "text/plain")).json()
        atch_id = up[0]["id"]

        captured: dict = {}

        def _capture(product_id, query, **kwargs):
            captured["query"] = query

            async def gen():
                yield _ev("ok")

            return gen()

        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _capture)

        resp = client.post(
            "/api/products/prod_1/ask",
            json={"query": "summarize", "attachment_ids": [atch_id]},
        )
        assert resp.status_code == 200
        assert captured["query"] == (
            "summarize\n\n<attachment name=\"notes.txt\">\ndeploy notes\n</attachment>"
        )

        # session_id is announced second; transcript is durable after [DONE].
        frames = [
            line[len("data: "):]
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        session_id = next(
            f.split('"session_id": "')[1].split('"')[0]
            for f in frames
            if '"session_id"' in f
        )
        messages = client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/messages"
        ).json()
        user_rows = [m for m in messages if m["role"] == "user"]
        assert len(user_rows) == 1
        # Raw query persisted — attachment content never lands in the transcript.
        assert user_rows[0]["content"] == "summarize"
        assert user_rows[0]["attachments"] == [
            {"id": atch_id, "filename": "notes.txt", "size_bytes": 12}
        ]
        # Assistant rows never carry attachment chips.
        assert all(
            m["attachments"] == [] for m in messages if m["role"] != "user"
        )

    def test_unknown_attachment_id_400(self, client, isolated_db, monkeypatch):
        import api.routers.expert as expert_router

        _seed(isolated_db)
        monkeypatch.setattr(
            expert_router,
            "run_agent_chat_stream",
            lambda *a, **k: _aiter([]),
        )
        resp = client.post(
            "/api/products/prod_1/ask",
            json={"query": "q", "attachment_ids": ["atch_missing"]},
        )
        assert resp.status_code == 400

    def test_too_many_attachment_ids_422(self, client):
        ids = [f"atch_{i}" for i in range(6)]
        resp = client.post(
            "/api/products/prod_1/ask",
            json={"query": "q", "attachment_ids": ids},
        )
        assert resp.status_code == 422


def _aiter(events):
    async def gen():
        for e in events:
            yield e

    return gen()
