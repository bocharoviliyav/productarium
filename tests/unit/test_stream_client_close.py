"""Unit tests for per-stream httpx client closing (P1-14 fixup).

``stream_chat`` / ``stream_chat_fields`` build a ChatOpenAI (with a dedicated
``httpx.AsyncClient``) per call via ``build_chat_model``; the generator must
close that client when the stream finishes — normally AND on early disconnect
(``agen.aclose()`` / consumer ``break``).
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import api.llm.stream as stream_mod


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class _FakeChunk:
    def __init__(self, content: str, reasoning: str = "") -> None:
        # Shape consumed by _chunk_text / _chunk_fields.
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}


class _FakeChat:
    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self.http_async_client = _FakeAsyncClient()

    async def astream(self, _messages):
        for chunk in self._chunks:
            yield chunk


def _install(monkeypatch, chunks) -> _FakeChat:
    chat = _FakeChat(chunks)
    monkeypatch.setattr(stream_mod, "build_chat_model", lambda **_kwargs: chat)
    return chat


class TestStreamChatClosesClient:
    def test_closes_after_full_drain(self, monkeypatch):
        chat = _install(
            monkeypatch,
            [_FakeChunk("hello "), _FakeChunk("world")],
        )

        async def run() -> list:
            out = []
            async for text in stream_mod.stream_chat("prompt"):
                out.append(text)
            return out

        out = asyncio.run(run())
        assert out == ["hello ", "world"]
        assert chat.http_async_client.closed == 1

    def test_closes_on_early_disconnect(self, monkeypatch):
        chat = _install(monkeypatch, [_FakeChunk("a"), _FakeChunk("b"), _FakeChunk("c")])

        async def run() -> list:
            agen = stream_mod.stream_chat("prompt")
            out = []
            out.append(await agen.__anext__())
            await agen.aclose()  # consumer aborted the SSE stream
            return out

        out = asyncio.run(run())
        assert out == ["a"]
        assert chat.http_async_client.closed == 1

    def test_closes_when_stream_raises(self, monkeypatch):
        class _Boom(Exception):
            pass

        class _BrokenChat(_FakeChat):
            async def astream(self, _messages):
                yield _FakeChunk("partial")
                raise _Boom()

        chat = _BrokenChat([])
        monkeypatch.setattr(stream_mod, "build_chat_model", lambda **_kwargs: chat)

        async def run():
            try:
                async for _ in stream_mod.stream_chat("prompt"):
                    pass
            except _Boom:
                return "raised"
            return "no-raise"

        assert asyncio.run(run()) == "raised"
        assert chat.http_async_client.closed == 1

    def test_close_failure_is_non_fatal(self, monkeypatch):
        class _BadClient:
            async def aclose(self) -> None:
                raise RuntimeError("already closed")

        class _Chat(_FakeChat):
            def __init__(self, chunks) -> None:
                super().__init__(chunks)
                self.http_async_client = _BadClient()

        chat = _Chat([_FakeChunk("x")])
        monkeypatch.setattr(stream_mod, "build_chat_model", lambda **_kwargs: chat)

        async def run() -> list:
            out = []
            async for text in stream_mod.stream_chat("prompt"):
                out.append(text)
            return out

        assert asyncio.run(run()) == ["x"]  # cleanup error must not surface


class TestStreamChatFieldsClosesClient:
    def test_closes_after_full_drain(self, monkeypatch):
        chat = _install(
            monkeypatch,
            [
                _FakeChunk("answer", reasoning="thinking..."),
                _FakeChunk("!"),
            ],
        )

        async def run() -> list:
            out = []
            async for pair in stream_mod.stream_chat_fields("prompt"):
                out.append(pair)
            return out

        out = asyncio.run(run())
        assert out == [("answer", "thinking..."), ("!", "")]
        assert chat.http_async_client.closed == 1

    def test_closes_on_early_disconnect(self, monkeypatch):
        chat = _install(monkeypatch, [_FakeChunk("a"), _FakeChunk("b")])

        async def run():
            agen = stream_mod.stream_chat_fields("prompt")
            pair = await agen.__anext__()
            await agen.aclose()
            return pair

        assert asyncio.run(run()) == ("a", "")
        assert chat.http_async_client.closed == 1

    def test_chat_without_client_attribute_is_fine(self, monkeypatch):
        class _BareChat(_FakeChat):
            def __init__(self, chunks) -> None:
                super().__init__(chunks)
                del self.http_async_client  # duck-typed: attr may be absent

        chat = _BareChat([_FakeChunk("ok")])
        monkeypatch.setattr(stream_mod, "build_chat_model", lambda **_kwargs: chat)

        async def run() -> list:
            out = []
            async for text in stream_mod.stream_chat("prompt"):
                out.append(text)
            return out

        assert asyncio.run(run()) == ["ok"]
