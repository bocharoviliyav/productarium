from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from adalflow.core.types import Document
from api.clients.ollama import OllamaDocumentProcessor


class _FakeEmbedderResult:
    """Mimics adalflow EmbedderOutput: result.data[0].embedding."""
    def __init__(self, data):
        self.data = data


class _FakeEmbedding:
    def __init__(self, embedding):
        self.embedding = embedding


def _make_result(embedding_or_none):
    if embedding_or_none is None:
        return _FakeEmbedderResult([])
    return _FakeEmbedderResult([_FakeEmbedding(embedding_or_none)])


class _FakeEmbedder:
    """A fake adal.Embedder whose __call__ returns canned embedding data."""

    def __init__(self, responses):
        # responses: list of _FakeEmbedderResult, one per call
        self._responses = list(responses)
        self.calls = []

    def __call__(self, input):
        self.calls.append(input)
        if self._responses:
            return self._responses.pop(0)
        return _make_result([0.1, 0.2, 0.3])


def _make_doc(text, file_path=None):
    doc = Document(text=text)
    doc.meta_data = {"file_path": file_path} if file_path else {}
    return doc


class TestOllamaDocumentProcessor:
    def test_processes_single_document(self):
        embedder = _FakeEmbedder([_make_result([0.1, 0.2, 0.3])])
        proc = OllamaDocumentProcessor(embedder)
        docs = [_make_doc("hello", "f1.txt")]
        result = proc(docs)
        assert len(result) == 1
        assert result[0].vector == [0.1, 0.2, 0.3]

    def test_processes_multiple_consistent_documents(self):
        embedder = _FakeEmbedder([
            _make_result([0.1, 0.2]),
            _make_result([0.3, 0.4]),
        ])
        proc = OllamaDocumentProcessor(embedder)
        docs = [_make_doc("a", "f1.txt"), _make_doc("b", "f2.txt")]
        result = proc(docs)
        assert len(result) == 2
        assert result[0].vector == [0.1, 0.2]
        assert result[1].vector == [0.3, 0.4]

    def test_skips_inconsistent_embedding_size(self):
        embedder = _FakeEmbedder([
            _make_result([0.1, 0.2, 0.3]),  # size 3 -> expected
            _make_result([0.1, 0.2]),         # size 2 -> skipped
            _make_result([0.4, 0.5, 0.6]),    # size 3 -> ok
        ])
        proc = OllamaDocumentProcessor(embedder)
        docs = [
            _make_doc("a", "f1.txt"),
            _make_doc("b", "f2.txt"),
            _make_doc("c", "f3.txt"),
        ]
        result = proc(docs)
        assert len(result) == 2
        assert result[0].vector == [0.1, 0.2, 0.3]
        assert result[1].vector == [0.4, 0.5, 0.6]

    def test_skips_document_with_empty_embedding(self):
        embedder = _FakeEmbedder([_make_result(None)])
        proc = OllamaDocumentProcessor(embedder)
        docs = [_make_doc("a", "f1.txt")]
        result = proc(docs)
        assert len(result) == 0

    def test_skips_document_on_embedder_exception(self):
        class _ErrorEmbedder:
            def __call__(self, input):
                raise RuntimeError("embedding failed")

        proc = OllamaDocumentProcessor(_ErrorEmbedder())
        docs = [_make_doc("a", "f1.txt")]
        result = proc(docs)
        assert len(result) == 0

    def test_empty_input(self):
        embedder = _FakeEmbedder([])
        proc = OllamaDocumentProcessor(embedder)
        result = proc([])
        assert result == []

    def test_sets_expected_size_from_first_successful(self):
        embedder = _FakeEmbedder([
            _make_result(None),               # first fails -> no expected size yet
            _make_result([0.1, 0.2]),          # second sets expected=2
            _make_result([0.3, 0.4]),          # third matches
        ])
        proc = OllamaDocumentProcessor(embedder)
        docs = [
            _make_doc("a", "f1.txt"),
            _make_doc("b", "f2.txt"),
            _make_doc("c", "f3.txt"),
        ]
        result = proc(docs)
        assert len(result) == 2

    def test_document_without_meta_data(self):
        """Documents without meta_data should not crash on file_path lookup."""
        embedder = _FakeEmbedder([_make_result([0.1])])
        proc = OllamaDocumentProcessor(embedder)
        doc = Document(text="no meta")
        doc.meta_data = None
        result = proc([doc])
        assert len(result) == 1
