#!/usr/bin/env python3
"""Unit tests for api.utils.repo_url.validate_repo_url (P0-1).

The validator is the gate that keeps dangerous git "URLs" (ext:: transport
helpers, file://, ssh://, SCP-like syntax, arbitrary local paths) away from
``git clone``. Matrix covers accepted http(s), every rejected class, and the
opt-in managed-local-path exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from api.utils.repo_url import (
    is_http_repo_url,
    local_clones_allowed,
    validate_repo_url,
)


class TestAccepted:
    def test_https(self):
        assert validate_repo_url("https://github.com/acme/repo.git") == "https://github.com/acme/repo.git"

    def test_http(self):
        assert validate_repo_url("http://gitlab.local/acme/repo.git") is not None

    def test_strip_whitespace(self):
        assert validate_repo_url("  https://github.com/acme/repo  ") == "https://github.com/acme/repo"

    def test_is_http_repo_url_true(self):
        assert is_http_repo_url("https://example.com/a/b") is True

    def test_is_http_repo_url_false_for_ssh(self):
        assert is_http_repo_url("ssh://git@example.com/a/b") is False


class TestRejected:
    @pytest.mark.parametrize(
        "url",
        [
            "ext::sh -c id",                       # transport helper → RCE
            "ext::/usr/bin/id",
            "file:///etc/passwd",                  # local filesystem read
            "/etc/passwd",                         # absolute path
            "../../etc/passwd",                    # relative traversal
            "ssh://git@github.com/acme/repo.git",
            "git://github.com/acme/repo.git",
            "git@github.com:acme/repo.git",        # SCP-like syntax
            "ftp://example.com/repo",
            "localhost",                           # hostname without scheme
            "localhost:1234/x",                    # scheme-less host:port
            "",                                    # empty
            "   ",                                 # whitespace-only
        ],
    )
    def test_rejected_by_default(self, url, monkeypatch):
        monkeypatch.delenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", raising=False)
        with pytest.raises(ValueError):
            validate_repo_url(url)

    def test_empty_message(self, monkeypatch):
        monkeypatch.delenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", raising=False)
        with pytest.raises(ValueError, match="empty"):
            validate_repo_url("")


class TestManagedLocalPathOptIn:
    def test_local_rejected_without_flag(self, monkeypatch):
        monkeypatch.delenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", raising=False)
        with pytest.raises(ValueError, match="PRODUCTARIUM_ALLOW_LOCAL_CLONES"):
            validate_repo_url("/tmp/some/repo")

    def test_flag_parsing(self, monkeypatch):
        monkeypatch.setenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", "1")
        assert local_clones_allowed() is True
        monkeypatch.setenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", "TRUE")
        assert local_clones_allowed() is True
        monkeypatch.setenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", "0")
        assert local_clones_allowed() is False
        monkeypatch.delenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", raising=False)
        assert local_clones_allowed() is False

    def test_managed_path_allowed_with_flag(self, monkeypatch, tmp_path):
        # Point the managed root at an isolated temp dir by faking the
        # adalflow helper, so the test never touches the real ~/.adalflow.
        import api.utils.repo_url as mod

        root = tmp_path / "managed"
        root.mkdir()
        repo = root / "repo"
        repo.mkdir()
        monkeypatch.setattr(mod, "_managed_root", lambda: str(root))
        monkeypatch.setenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", "1")

        assert validate_repo_url(str(repo)) == str(repo)
        # Nested path inside the managed root is fine too.
        assert validate_repo_url(str(repo / "sub")) == str(repo / "sub")

    def test_outside_managed_root_rejected_even_with_flag(self, monkeypatch, tmp_path):
        import api.utils.repo_url as mod

        root = tmp_path / "managed"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setattr(mod, "_managed_root", lambda: str(root))
        monkeypatch.setenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", "1")

        with pytest.raises(ValueError, match="managed state dir"):
            validate_repo_url(str(outside))

    def test_traversal_out_of_managed_root_rejected(self, monkeypatch, tmp_path):
        """A path that STARTS inside the root but realpath-escapes is rejected."""
        import api.utils.repo_url as mod

        root = tmp_path / "managed"
        (root / "repo").mkdir(parents=True)
        monkeypatch.setattr(mod, "_managed_root", lambda: str(root))
        monkeypatch.setenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", "1")

        # realpath(managed/repo/../../outside) is outside the managed root.
        with pytest.raises(ValueError):
            validate_repo_url(str(root / "repo" / ".." / ".." / "outside"))
