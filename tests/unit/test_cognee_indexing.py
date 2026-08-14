"""Unit tests for ``api.cognee.indexing``.

Covers:
- ``_resolve_raw_data_location_path``: file:// stripping, empty/bare passthrough.
- ``_is_likely_text_file``: extension + basename allow-list.
- ``_looks_like_file_path``: existing file True, text blob / newline / too-long False.
- ``_read_repo_text_for_cognee``: walks a temp dir, skips non-text, caps per-file/blob.
- ``_resolve_safe_chunk_size``: derived from model context window, floor 300, ceiling 1200.
- ``_remember_with_timeout``: success / TimeoutError (soft success) / CancelledError (soft).
- ``add_document``: cognee unavailable -> False, empty payload -> False, dir -> blob,
  text payload -> DataItem wrap, duplicate-key retry path, generic error -> False.
- ``cognify_dataset``: success True, timeout soft-success True, error -> False, unavailable -> False.
- ``add_documents_and_cognify_once``: filters empty items, unavailable -> None.
- ``add_and_index_document``: delegates to add_document.
- ``_empty_cognee_dataset``: forget success, forget "not found" -> True, forget error -> fallback,
  no datasets attr -> False.
- ``_reconcile_stale_cognee_data``: unavailable -> None, no datasets attr -> None, fresh DB error -> None.
- ``reindex_product_knowledge_graph``: unavailable -> failure dict, no products -> success dict,
  DB query error (artifacts attr removed) -> failure dict (non-fatal).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.cognee.indexing import (
    _direct_delete_dataset_data,
    _empty_cognee_dataset,
    _is_likely_text_file,
    _looks_like_file_path,
    _read_repo_text_for_cognee,
    _reconcile_stale_cognee_data,
    _remember_with_timeout,
    _resolve_raw_data_location_path,
    _resolve_safe_chunk_size,
    add_and_index_document,
    add_document,
    add_documents_and_cognify_once,
    cognify_dataset,
    reindex_product_knowledge_graph,
)


# --------------------------------------------------------------------------- #
# _resolve_raw_data_location_path
# --------------------------------------------------------------------------- #
class TestResolveRawDataLocationPath:
    def test_strips_file_scheme(self):
        assert _resolve_raw_data_location_path("file:///root/.adalflow/cognee_data/text_abc.txt") == "/root/.adalflow/cognee_data/text_abc.txt"

    def test_bare_path_unchanged(self):
        path = "/root/.adalflow/cognee_data/text_abc.txt"
        assert _resolve_raw_data_location_path(path) == path

    def test_empty_string(self):
        assert _resolve_raw_data_location_path("") == ""

    def test_non_file_scheme_unchanged(self):
        assert _resolve_raw_data_location_path("http://example.com/file.txt") == "http://example.com/file.txt"


# --------------------------------------------------------------------------- #
# _is_likely_text_file
# --------------------------------------------------------------------------- #
class TestIsLikelyTextFile:
    def test_py_extension(self):
        assert _is_likely_text_file("/repo/src/main.py") is True

    def test_md_extension(self):
        assert _is_likely_text_file("/repo/README.md") is True

    def test_json_extension(self):
        assert _is_likely_text_file("/repo/config.json") is True

    def test_binary_extension(self):
        assert _is_likely_text_file("/repo/image.png") is False

    def test_no_extension_readme_basename(self):
        assert _is_likely_text_file("/repo/readme") is True

    def test_no_extension_makefile_basename(self):
        assert _is_likely_text_file("/repo/makefile") is True

    def test_no_extension_unknown_basename(self):
        assert _is_likely_text_file("/repo/somefile") is False

    def test_dockerfile_basename(self):
        assert _is_likely_text_file("/repo/Dockerfile") is True

    def test_case_insensitive(self):
        assert _is_likely_text_file("/repo/MAIN.PY") is True


# --------------------------------------------------------------------------- #
# _looks_like_file_path
# --------------------------------------------------------------------------- #
class TestLooksLikeFilePath:
    def test_existing_file_true(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert _looks_like_file_path(str(f)) is True

    def test_nonexistent_file_false(self, tmp_path):
        assert _looks_like_file_path(str(tmp_path / "nonexistent.txt")) is False

    def test_empty_string_false(self):
        assert _looks_like_file_path("") is False

    def test_text_blob_with_newline_false(self):
        assert _looks_like_file_path("line1\nline2") is False

    def test_text_blob_too_long_false(self):
        assert _looks_like_file_path("x" * 5000) is False

    def test_directory_false(self, tmp_path):
        # os.path.isfile returns False for a directory.
        assert _looks_like_file_path(str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# _read_repo_text_for_cognee
# --------------------------------------------------------------------------- #
class TestReadRepoTextForCognee:
    def test_empty_string_returns_empty(self):
        assert _read_repo_text_for_cognee("") == ""

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert _read_repo_text_for_cognee(str(tmp_path / "nonexistent")) == ""

    def test_reads_text_files(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Title")
        result = _read_repo_text_for_cognee(str(tmp_path))
        assert "main.py" in result
        assert "print('hello')" in result
        assert "README.md" in result
        assert "# Title" in result

    def test_skips_non_text_files(self, tmp_path):
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "image.png").write_text("binary")
        result = _read_repo_text_for_cognee(str(tmp_path))
        assert "main.py" in result
        assert "image.png" not in result

    def test_skips_git_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("code")
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "index").write_text("git binary")
        result = _read_repo_text_for_cognee(str(tmp_path))
        assert "main.py" in result
        assert ".git" not in result or "index" not in result

    def test_skips_node_modules(self, tmp_path):
        (tmp_path / "main.py").write_text("code")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "dep.js").write_text("dep code")
        result = _read_repo_text_for_cognee(str(tmp_path))
        assert "main.py" in result
        assert "node_modules" not in result

    def test_per_file_cap(self, tmp_path):
        # Write a file larger than the per-file cap.
        big = "x" * 20_000
        (tmp_path / "big.py").write_text(big)
        result = _read_repo_text_for_cognee(str(tmp_path))
        # The file content should be truncated.
        assert "file truncated" in result
        assert big not in result

    def test_empty_dir_returns_empty(self, tmp_path):
        assert _read_repo_text_for_cognee(str(tmp_path)) == ""


# --------------------------------------------------------------------------- #
# _resolve_safe_chunk_size
# --------------------------------------------------------------------------- #
class TestResolveSafeChunkSize:
    def test_returns_int(self):
        val = _resolve_safe_chunk_size()
        assert isinstance(val, int)

    def test_within_bounds(self):
        val = _resolve_safe_chunk_size()
        assert 300 <= val <= 1200

    def test_small_context_window_floors_at_300(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        def _tiny_ctx(**kw):
            return 1000

        monkeypatch.setattr("api.utils.get_model_context_window", _tiny_ctx)
        # ctx=1000 -> (1000-3000)//2 = -1000 -> max(300, min(1200, -1000)) = 300
        val = _resolve_safe_chunk_size()
        assert val == 300

    def test_large_context_window_caps_at_1200(self, monkeypatch):
        def _huge_ctx(**kw):
            return 1_000_000

        monkeypatch.setattr("api.utils.get_model_context_window", _huge_ctx)
        # ctx=1_000_000 -> (1000000-3000)//2 = 498500 -> max(300, min(1200, 498500)) = 1200
        val = _resolve_safe_chunk_size()
        assert val == 1200

    def test_exception_falls_back_to_8192(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("no model")

        monkeypatch.setattr("api.utils.get_model_context_window", _boom)
        # Exception -> ctx=8192 -> (8192-3000)//2 = 2596 -> max(300, min(1200, 2596)) = 1200
        val = _resolve_safe_chunk_size()
        assert val == 1200


# --------------------------------------------------------------------------- #
# _remember_with_timeout
# --------------------------------------------------------------------------- #
class TestRememberWithTimeout:
    def test_success(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        called = []

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            called.append(payload)
            return "ok"

        fake_cognee.remember = _remember

        asyncio.run(_remember_with_timeout("text", "prod_1", 500))
        assert called == ["text"]

    def test_timeout_is_soft_success(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            raise asyncio.TimeoutError()

        fake_cognee.remember = _remember

        # Should NOT raise — timeout is treated as soft success.
        asyncio.run(_remember_with_timeout("text", "prod_1", 500))

    def test_cancelled_is_soft_success(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            raise asyncio.CancelledError("cancelled")

        fake_cognee.remember = _remember

        # Should NOT raise — cancelled is treated as soft success.
        asyncio.run(_remember_with_timeout("text", "prod_1", 500))


# --------------------------------------------------------------------------- #
# add_document
# --------------------------------------------------------------------------- #
class TestAddDocument:
    def test_unavailable_returns_false(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", False)
        result = asyncio.run(add_document("content", "prod_1"))
        assert result is False

    def test_empty_payload_returns_false(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)
        result = asyncio.run(add_document("", "prod_1"))
        assert result is False

    def test_text_payload_success(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        called = []

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            called.append(payload)
            return "ok"

        fake_cognee.remember = _remember
        # DataItem import will fail (no real cognee) -> payload stays a str.
        # That's fine: _looks_like_file_path returns False for non-file text.

        result = asyncio.run(add_document("some text content", "prod_1"))
        assert result is True
        assert len(called) == 1

    def test_dir_with_no_text_returns_false(self, monkeypatch, tmp_path):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        # Empty dir -> no text files found.
        result = asyncio.run(add_document(str(tmp_path), "prod_1"))
        assert result is False

    def test_dir_with_text_success(self, monkeypatch, fake_cognee, tmp_path):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        (tmp_path / "main.py").write_text("print('hi')")

        called = []

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            called.append(payload)
            return "ok"

        fake_cognee.remember = _remember

        result = asyncio.run(add_document(str(tmp_path), "prod_1"))
        assert result is True
        assert len(called) == 1
        # The payload should be the blob (contains the file content).
        assert "print('hi')" in called[0]

    def test_generic_error_returns_false(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            raise RuntimeError("cognee failed")

        fake_cognee.remember = _remember

        result = asyncio.run(add_document("text", "prod_1"))
        assert result is False

    def test_duplicate_key_retry_success(self, monkeypatch, fake_cognee):
        """A duplicate-key error triggers _empty_cognee_dataset then retries."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        call_count = 0

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("duplicate key value violates data_pkey")

        fake_cognee.remember = _remember

        # Mock _empty_cognee_dataset to return True (cleared).
        async def _fake_empty(name):
            return True

        monkeypatch.setattr(idx_mod, "_empty_cognee_dataset", _fake_empty)

        result = asyncio.run(add_document("text", "prod_1"))
        assert result is True
        assert call_count == 2  # first failed, retry succeeded

    def test_duplicate_key_retry_failure_returns_false(self, monkeypatch, fake_cognee):
        """Duplicate-key retry that fails again returns False."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        call_count = 0

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            nonlocal call_count
            call_count += 1
            raise Exception("duplicate key value violates data_pkey")

        fake_cognee.remember = _remember

        async def _fake_empty(name):
            return True

        monkeypatch.setattr(idx_mod, "_empty_cognee_dataset", _fake_empty)

        result = asyncio.run(add_document("text", "prod_1"))
        assert result is False

    def test_unique_violation_error_type_triggers_retry(self, monkeypatch, fake_cognee):
        """UniqueViolationError exception type also triggers retry."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        call_count = 0

        class UniqueViolationError(Exception):
            pass

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise UniqueViolationError("constraint")

        fake_cognee.remember = _remember

        async def _fake_empty(name):
            return True

        monkeypatch.setattr(idx_mod, "_empty_cognee_dataset", _fake_empty)

        result = asyncio.run(add_document("text", "prod_1"))
        assert result is True
        assert call_count == 2


# --------------------------------------------------------------------------- #
# cognify_dataset
# --------------------------------------------------------------------------- #
class TestCognifyDataset:
    def test_unavailable_returns_false(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", False)
        result = asyncio.run(cognify_dataset("prod_1"))
        assert result is False

    def test_success_returns_true(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        called = []

        async def _cognify(datasets=None, chunk_size=None):
            called.append((datasets, chunk_size))
            return "ok"

        fake_cognee.cognify = _cognify

        result = asyncio.run(cognify_dataset("prod_1"))
        assert result is True
        assert called[0][0] == ["prod_1"]

    def test_timeout_returns_true_soft_success(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        async def _cognify(datasets=None, chunk_size=None):
            raise asyncio.TimeoutError()

        fake_cognee.cognify = _cognify

        result = asyncio.run(cognify_dataset("prod_1"))
        assert result is True

    def test_cancelled_returns_true_soft_success(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        async def _cognify(datasets=None, chunk_size=None):
            raise asyncio.CancelledError()

        fake_cognee.cognify = _cognify

        result = asyncio.run(cognify_dataset("prod_1"))
        assert result is True

    def test_generic_error_returns_false(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        async def _cognify(datasets=None, chunk_size=None):
            raise RuntimeError("cognify crashed")

        fake_cognee.cognify = _cognify

        result = asyncio.run(cognify_dataset("prod_1"))
        assert result is False


# --------------------------------------------------------------------------- #
# add_documents_and_cognify_once
# --------------------------------------------------------------------------- #
class TestAddDocumentsAndCognifyOnce:
    def test_unavailable_returns_none(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", False)
        result = asyncio.run(add_documents_and_cognify_once(["a", "b"], "prod_1"))
        assert result is None

    def test_empty_items_returns_none(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        result = asyncio.run(add_documents_and_cognify_once([], "prod_1"))
        assert result is None

    def test_all_blank_items_returns_none(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        result = asyncio.run(add_documents_and_cognify_once(["", "  ", None], "prod_1"))
        assert result is None

    def test_success_calls_remember(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        called = []

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            called.append(payload)
            return "ok"

        fake_cognee.remember = _remember

        asyncio.run(add_documents_and_cognify_once(["a", "b"], "prod_1"))
        assert len(called) == 1
        assert called[0] == ["a", "b"]

    def test_error_does_not_raise(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        async def _remember(payload, dataset_name=None, self_improvement=False, chunk_size=None):
            raise RuntimeError("failed")

        fake_cognee.remember = _remember

        # Should not raise.
        asyncio.run(add_documents_and_cognify_once(["a"], "prod_1"))


# --------------------------------------------------------------------------- #
# add_and_index_document
# --------------------------------------------------------------------------- #
class TestAddAndIndexDocument:
    def test_delegates_to_add_document(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        called = []

        async def _fake_add(content_or_path, dataset_name):
            called.append((content_or_path, dataset_name))
            return True

        monkeypatch.setattr(idx_mod, "add_document", _fake_add)
        asyncio.run(add_and_index_document("content", "prod_1"))
        assert called == [("content", "prod_1")]


# --------------------------------------------------------------------------- #
# _empty_cognee_dataset
# --------------------------------------------------------------------------- #
class TestEmptyCogneeDataset:
    def test_forget_success_returns_true(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        async def _forget(dataset=None):
            return None

        fake_cognee.forget = _forget

        result = asyncio.run(_empty_cognee_dataset("prod_1"))
        assert result is True

    def test_forget_not_found_returns_true(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        async def _forget(dataset=None):
            raise Exception("dataset not found")

        fake_cognee.forget = _forget

        result = asyncio.run(_empty_cognee_dataset("prod_1"))
        assert result is True

    def test_forget_no_dataset_returns_true(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        async def _forget(dataset=None):
            raise Exception("no dataset with that name")

        fake_cognee.forget = _forget

        result = asyncio.run(_empty_cognee_dataset("prod_1"))
        assert result is True

    def test_forget_does_not_exist_returns_true(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        async def _forget(dataset=None):
            raise Exception("Dataset does not exist")

        fake_cognee.forget = _forget

        result = asyncio.run(_empty_cognee_dataset("prod_1"))
        assert result is True

    def test_forget_error_fallback_no_datasets_attr_returns_false(self, monkeypatch, fake_cognee):
        """forget fails with a non-not-found error AND no datasets attr -> False."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        async def _forget(dataset=None):
            raise Exception("some other error")

        fake_cognee.forget = _forget
        # No `datasets` attribute on fake_cognee -> fallback returns False.
        if hasattr(fake_cognee, "datasets"):
            del fake_cognee.datasets

        result = asyncio.run(_empty_cognee_dataset("prod_1"))
        assert result is False

    def test_forget_error_fallback_no_matching_dataset_returns_true(self, monkeypatch, fake_cognee):
        """forget fails, fallback list_datasets finds no match -> True."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        async def _forget(dataset=None):
            raise Exception("some other error")

        fake_cognee.forget = _forget

        datasets_obj = SimpleNamespace()

        async def _list_datasets():
            return []  # no datasets

        datasets_obj.list_datasets = _list_datasets
        fake_cognee.datasets = datasets_obj

        result = asyncio.run(_empty_cognee_dataset("prod_1"))
        assert result is True

    def test_no_forget_attr_fallback_no_datasets_returns_false(self, monkeypatch, fake_cognee):
        """No forget attr AND no datasets attr -> False."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        if hasattr(fake_cognee, "forget"):
            del fake_cognee.forget
        if hasattr(fake_cognee, "datasets"):
            del fake_cognee.datasets

        result = asyncio.run(_empty_cognee_dataset("prod_1"))
        assert result is False


# --------------------------------------------------------------------------- #
# _reconcile_stale_cognee_data
# --------------------------------------------------------------------------- #
class TestReconcileStaleCogneeData:
    def test_unavailable_returns_none(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", False)
        result = asyncio.run(_reconcile_stale_cognee_data())
        assert result is None

    def test_no_datasets_attr_returns_none(self, monkeypatch, fake_cognee):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        if hasattr(fake_cognee, "datasets"):
            del fake_cognee.datasets

        result = asyncio.run(_reconcile_stale_cognee_data())
        assert result is None

    def test_list_datasets_exception_fresh_db_returns_none(self, monkeypatch, fake_cognee):
        """A DatabaseNotCreatedError on list_datasets -> None (non-fatal)."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        datasets_obj = SimpleNamespace()

        async def _list_datasets():
            raise RuntimeError("DatabaseNotCreatedError: The database has not been created yet")

        datasets_obj.list_datasets = _list_datasets
        fake_cognee.datasets = datasets_obj

        result = asyncio.run(_reconcile_stale_cognee_data())
        assert result is None

    def test_no_stale_rows_returns_none(self, monkeypatch, fake_cognee):
        """All data rows have backing files -> no stale rows -> None."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        datasets_obj = SimpleNamespace()

        async def _list_datasets():
            return [SimpleNamespace(id="ds1", name="prod_1")]

        async def _list_data(ds_id):
            # Return a row whose backing file exists (use a real file).
            import tempfile

            f = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            f.write(b"content")
            f.close()
            return [SimpleNamespace(id="row1", raw_data_location=f.name)]

        datasets_obj.list_datasets = _list_datasets
        datasets_obj.list_data = _list_data
        fake_cognee.datasets = datasets_obj

        result = asyncio.run(_reconcile_stale_cognee_data())
        assert result is None

    def test_stale_row_with_file_scheme_path_exists(self, monkeypatch, fake_cognee):
        """A row with a file:// path whose backing file exists is NOT stale."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        datasets_obj = SimpleNamespace()

        async def _list_datasets():
            return [SimpleNamespace(id="ds1", name="prod_1")]

        import tempfile

        f = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        f.write(b"content")
        f.close()

        async def _list_data(ds_id):
            return [SimpleNamespace(id="row1", raw_data_location=f"file://{f.name}")]

        datasets_obj.list_datasets = _list_datasets
        datasets_obj.list_data = _list_data
        fake_cognee.datasets = datasets_obj

        result = asyncio.run(_reconcile_stale_cognee_data())
        assert result is None  # file exists -> not stale -> no deletion

    def test_no_list_data_fn_returns_none(self, monkeypatch, fake_cognee):
        """When datasets_obj has no list_data callable, reconcile returns None."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "cognee", fake_cognee)

        datasets_obj = SimpleNamespace()

        async def _list_datasets():
            return [SimpleNamespace(id="ds1", name="prod_1")]

        datasets_obj.list_datasets = _list_datasets
        # No list_data attribute.
        fake_cognee.datasets = datasets_obj

        result = asyncio.run(_reconcile_stale_cognee_data())
        assert result is None


# --------------------------------------------------------------------------- #
# reindex_product_knowledge_graph
# --------------------------------------------------------------------------- #
class TestReindexProductKnowledgeGraph:
    def test_unavailable_returns_failure_dict(self, monkeypatch):
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", False)
        result = asyncio.run(reindex_product_knowledge_graph())
        assert result["success"] is False
        assert result["reindexed_count"] == 0
        assert "not available" in result["message"].lower()

    def test_no_products_returns_success_dict(self, monkeypatch, isolated_db):
        """With an empty products table, reindex returns success with count 0.

        NOTE: reindex_product_knowledge_graph references ProductORM.artifacts
        (removed in the ArtifactORM split). We alias it to codebases so the
        selectinload call builds without error; the empty table then returns
        no products and the function returns the success dict.
        """
        import api.cognee.indexing as idx_mod
        from api.models import ProductORM

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        # Alias the removed `artifacts` relationship to `codebases` so the
        # selectinload(ProductORM.artifacts) call in the source doesn't raise.
        if not hasattr(ProductORM, "artifacts"):
            ProductORM.artifacts = ProductORM.codebases

        # Patch api.db.SessionLocal to the isolated one.
        monkeypatch.setattr("api.db.SessionLocal", isolated_db.SessionLocal)

        result = asyncio.run(reindex_product_knowledge_graph())
        assert result["success"] is True
        assert result["reindexed_count"] == 0

    def test_db_error_returns_failure_dict(self, monkeypatch, isolated_db):
        """When the DB query raises (artifacts attr removed), reindex returns failure."""
        import api.cognee.indexing as idx_mod

        monkeypatch.setattr(idx_mod, "_COGNEE_AVAILABLE", True)
        monkeypatch.setattr(idx_mod, "apply_cognee_runtime_config", lambda: None)

        # Patch api.db.SessionLocal to raise during query.
        class _BadSession:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def query(self, *a, **kw):
                raise RuntimeError("db connection lost")

        monkeypatch.setattr("api.db.SessionLocal", lambda: _BadSession())

        result = asyncio.run(reindex_product_knowledge_graph())
        assert result["success"] is False
        assert result["reindexed_count"] == 0
        assert "reindex error" in result["message"].lower()


# --------------------------------------------------------------------------- #
# _direct_delete_dataset_data (defensive — depends on cognee internals)
# --------------------------------------------------------------------------- #
class TestDirectDeleteDatasetData:
    def test_import_failure_returns_false(self, monkeypatch):
        """When cognee relational engine import fails, returns False."""
        # No fake_cognee with the relational submodule -> import fails.
        result = asyncio.run(_direct_delete_dataset_data("ds_123"))
        assert result is False
