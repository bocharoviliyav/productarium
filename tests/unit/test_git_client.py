"""Unit tests for the git client + Git-host integration connectors.

Covers (>=85% each):
- ``api.clients.git``: token sanitization, authed clone-URL building, download_repo
  (reuse / force-refresh / re-clone / error mapping), GitHub/GitLab file-content
  API fetches, and the dispatcher.
- ``api.integrations._git_base``: ``GitConnector`` clone-or-pull, listing with
  pagination, repo-name extraction, clone-dir resolution, markdown building.
- ``api.integrations.github``: ``GitHubConnector`` test/list/pull.
- ``api.integrations.gitlab``: ``GitLabConnector`` test/list/pull.

No real git, no real network. ``subprocess.run`` is monkeypatched on the
``api.clients.git`` module; ``requests.get`` is monkeypatched on the use-site
modules where the calling code looks it up.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Project-root sys.path insert (required by every unit test file).
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeResp:
    """Minimal stand-in for a ``requests.Response``."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        raise_exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text if text else (
            json.dumps(json_data) if json_data is not None else ""
        )
        self._raise_exc = raise_exc

    def json(self) -> Any:
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            raise self._raise_exc
        if self.status_code >= 400:
            from requests.exceptions import HTTPError

            raise HTTPError(f"HTTP {self.status_code}")


def _completed(stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    """A successful ``subprocess.CompletedProcess``-like object."""
    m = MagicMock()
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = 0
    return m


def _cpe(stderr: bytes = b"boom") -> subprocess.CalledProcessError:
    """A ``CalledProcessError`` carrying the given stderr bytes."""
    return subprocess.CalledProcessError(returncode=1, cmd=["git"], stderr=stderr)


# ===========================================================================
# api.clients.git — helpers
# ===========================================================================
class TestSanitizeGitStderr:
    def test_strips_raw_token(self):
        from api.clients.git import _sanitize_git_stderr

        text = "fatal: bad remote https://token@host/repo"
        assert _sanitize_git_stderr(text, "token") == "fatal: bad remote https://***TOKEN***@host/repo"

    def test_strips_url_encoded_token(self):
        from api.clients.git import _sanitize_git_stderr
        from urllib.parse import quote

        token = "p@ss/w:ord"
        encoded = quote(token, safe="")
        text = f"fatal: {token} and {encoded}"
        out = _sanitize_git_stderr(text, token)
        assert token not in out
        assert encoded not in out
        assert out.count("***TOKEN***") == 2

    def test_empty_token_returns_text(self):
        from api.clients.git import _sanitize_git_stderr

        assert _sanitize_git_stderr("some text", "") == "some text"
        assert _sanitize_git_stderr("some text", None) == "some text"

    def test_empty_text_returns_text(self):
        from api.clients.git import _sanitize_git_stderr

        assert _sanitize_git_stderr("", "token") == ""


class TestIsGitRepo:
    def test_empty_path(self, tmp_path):
        from api.clients.git import _is_git_repo

        assert _is_git_repo("") is False

    def test_not_a_repo(self, tmp_path):
        from api.clients.git import _is_git_repo

        assert _is_git_repo(str(tmp_path)) is False

    def test_valid_repo(self, tmp_path):
        from api.clients.git import _is_git_repo

        (tmp_path / ".git").mkdir()
        assert _is_git_repo(str(tmp_path)) is True


class TestBuildAuthedCloneUrl:
    def test_github_injects_token(self):
        from api.clients.git import _build_authed_clone_url

        url = _build_authed_clone_url("https://github.com/owner/repo", "github", "tok123")
        assert "tok123@github.com" in url
        assert url.startswith("https://")
        assert "/owner/repo" in url

    def test_gitlab_injects_oauth2_prefix(self):
        from api.clients.git import _build_authed_clone_url

        url = _build_authed_clone_url("https://gitlab.com/group/project", "gitlab", "tok123")
        assert "oauth2:tok123@gitlab.com" in url
        assert "/group/project" in url

    def test_no_token_returns_unchanged(self):
        from api.clients.git import _build_authed_clone_url

        assert _build_authed_clone_url("https://github.com/o/r", "github", "") == "https://github.com/o/r"
        assert _build_authed_clone_url("https://github.com/o/r", "github", None) == "https://github.com/o/r"

    def test_unknown_repo_type_returns_unchanged(self):
        from api.clients.git import _build_authed_clone_url

        assert _build_authed_clone_url("https://example.com/o/r", "bitbucket", "tok") == "https://example.com/o/r"

    def test_token_with_special_chars_is_encoded(self):
        from api.clients.git import _build_authed_clone_url
        from urllib.parse import quote

        token = "p@ss"
        url = _build_authed_clone_url("https://github.com/o/r", "github", token)
        assert quote(token, safe="") in url
        assert token + "@" not in url  # raw token not present


# ===========================================================================
# api.clients.git — download_repo
# ===========================================================================
class TestDownloadRepo:
    def test_fresh_clone_success(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _completed(stdout=b"cloned", stderr=b"")

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
        local = str(tmp_path / "repo")
        out = git_mod.download_repo("https://github.com/o/r", local, repo_type="github")

        assert out == "cloned"
        # git --version, then clone
        assert calls[0] == ["git", "--version"]
        clone_cmd = calls[-1]
        assert clone_cmd[:4] == ["git", "clone", "--depth=1", "--single-branch"]
        assert clone_cmd[-2].startswith("https://")
        assert clone_cmd[-1] == local

    def test_existing_repo_reuse_no_force(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        local = str(tmp_path / "repo")
        os.makedirs(os.path.join(local, ".git"))
        # Put a file so os.listdir is non-empty.
        Path(local, "README.md").write_text("hi")

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _completed()

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
        out = git_mod.download_repo("https://github.com/o/r", local, repo_type="github")

        assert "Using existing repository" in out
        # Only the git --version check; no clone.
        assert len(calls) == 1

    def test_force_refresh_success(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        local = str(tmp_path / "repo")
        os.makedirs(os.path.join(local, ".git"))
        Path(local, "README.md").write_text("hi")

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return _completed()

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
        out = git_mod.download_repo(
            "https://github.com/o/r", local, repo_type="github",
            access_token="tok", force_refresh=True,
        )

        assert "Refreshed existing repository" in out
        # git --version, fetch, reset, clean
        assert calls[0] == ["git", "--version"]
        # Inspect each subcommand by its verb rather than exact slice.
        fetch_cmd = next(c for c in calls if "fetch" in c)
        assert fetch_cmd[:3] == ["git", "-C", local]
        assert "--depth=1" in fetch_cmd and "origin" in fetch_cmd
        reset_cmd = next(c for c in calls if "reset" in c)
        assert reset_cmd[:3] == ["git", "-C", local]
        assert reset_cmd[-1] == "FETCH_HEAD"
        clean_cmd = next(c for c in calls if "clean" in c)
        assert clean_cmd[:3] == ["git", "-C", local]
        assert "-fdx" in clean_cmd

    def test_force_refresh_failure_re_clones(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        local = str(tmp_path / "repo")
        os.makedirs(os.path.join(local, ".git"))
        Path(local, "README.md").write_text("hi")

        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            # git --version succeeds; fetch fails; clone succeeds.
            if "fetch" in cmd:
                raise _cpe(stderr=b"fetch failed")
            if "clone" in cmd:
                os.makedirs(os.path.join(local, ".git"))
                return _completed(stdout=b"cloned fresh")
            return _completed(stdout=b"cloned fresh")

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
        out = git_mod.download_repo(
            "https://github.com/o/r", local, repo_type="github", force_refresh=True,
        )

        assert out == "cloned fresh"
        # The original dir was removed + re-created by the clone.
        assert os.path.isdir(os.path.join(local, ".git"))

    def test_non_git_dir_force_refresh_removes_and_clones(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        local = str(tmp_path / "repo")
        os.makedirs(local)
        Path(local, "junk.txt").write_text("not a repo")

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            # Create the .git dir so it looks like a clone happened.
            if "clone" in cmd:
                os.makedirs(os.path.join(local, ".git"))
            return _completed(stdout=b"cloned")

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
        out = git_mod.download_repo(
            "https://github.com/o/r", local, repo_type="github", force_refresh=True,
        )

        assert out == "cloned"
        assert not Path(local, "junk.txt").exists()

    def test_non_git_dir_no_force_returns_existing(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        local = str(tmp_path / "repo")
        os.makedirs(local)
        Path(local, "junk.txt").write_text("not a repo")

        monkeypatch.setattr(git_mod.subprocess, "run", lambda *a, **k: _completed())
        out = git_mod.download_repo("https://github.com/o/r", local, repo_type="github")

        assert "Using existing repository" in out

    def test_git_not_installed_raises_value_error(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        def fake_run(cmd, **kwargs):
            raise _cpe(stderr=b"git not found")

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
        with pytest.raises(ValueError, match="Error during cloning"):
            git_mod.download_repo("https://github.com/o/r", str(tmp_path / "r"), "github")

    def test_generic_exception_raises_value_error(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        def fake_run(cmd, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
        with pytest.raises(ValueError, match="An unexpected error occurred"):
            git_mod.download_repo("https://github.com/o/r", str(tmp_path / "r"), "github")

    def test_clone_called_process_error_sanitized(self, tmp_path, monkeypatch):
        from api.clients import git as git_mod

        token = "secret123"

        def fake_run(cmd, **kwargs):
            if "clone" in cmd:
                raise _cpe(stderr=token.encode())
            return _completed()

        monkeypatch.setattr(git_mod.subprocess, "run", fake_run)
        with pytest.raises(ValueError) as ei:
            git_mod.download_repo(
                "https://github.com/o/r", str(tmp_path / "r"), "github",
                access_token=token,
            )
        assert token not in str(ei.value)
        assert "***TOKEN***" in str(ei.value)


# ===========================================================================
# api.clients.git — GitHub file content
# ===========================================================================
class TestGetGithubFileContent:
    def test_success_public_github(self, monkeypatch):
        from api.clients import git as git_mod

        content = "print('hello')"
        b64 = base64.b64encode(content.encode()).decode()
        resp = _FakeResp(json_data={"content": b64, "encoding": "base64"})
        monkeypatch.setattr(git_mod.requests, "get", lambda *a, **k: resp)

        out = git_mod.get_github_file_content(
            "https://github.com/owner/repo", "src/main.py",
        )
        assert out == content

    def test_token_header_sent(self, monkeypatch):
        from api.clients import git as git_mod

        b64 = base64.b64encode(b"x").decode()
        resp = _FakeResp(json_data={"content": b64, "encoding": "base64"})
        captured: dict = {}

        def fake_get(url, headers=None, **kw):
            captured["headers"] = headers
            captured["url"] = url
            return resp

        monkeypatch.setattr(git_mod.requests, "get", fake_get)
        git_mod.get_github_file_content(
            "https://github.com/o/r", "f.py", access_token="tok",
        )
        assert captured["headers"]["Authorization"] == "token tok"

    def test_enterprise_url(self, monkeypatch):
        from api.clients import git as git_mod

        b64 = base64.b64encode(b"x").decode()
        resp = _FakeResp(json_data={"content": b64, "encoding": "base64"})
        captured: dict = {}

        def fake_get(url, headers=None, **kw):
            captured["url"] = url
            return resp

        monkeypatch.setattr(git_mod.requests, "get", fake_get)
        git_mod.get_github_file_content(
            "https://github.company.com/o/r", "f.py",
        )
        assert "github.company.com/api/v3" in captured["url"]

    def test_404_raises_value_error(self, monkeypatch):
        from api.clients import git as git_mod
        from requests.exceptions import HTTPError

        resp = _FakeResp(status_code=404, raise_exc=HTTPError("404"))
        monkeypatch.setattr(git_mod.requests, "get", lambda *a, **k: resp)
        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_github_file_content("https://github.com/o/r", "f.py")

    def test_api_error_response_raises(self, monkeypatch):
        from api.clients import git as git_mod

        resp = _FakeResp(json_data={"message": "Not Found", "documentation_url": "x"})
        monkeypatch.setattr(git_mod.requests, "get", lambda *a, **k: resp)
        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_github_file_content("https://github.com/o/r", "f.py")

    def test_unexpected_encoding_raises(self, monkeypatch):
        from api.clients import git as git_mod

        resp = _FakeResp(json_data={"content": "x", "encoding": "hex"})
        monkeypatch.setattr(git_mod.requests, "get", lambda *a, **k: resp)
        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_github_file_content("https://github.com/o/r", "f.py")

    def test_missing_content_raises(self, monkeypatch):
        from api.clients import git as git_mod

        resp = _FakeResp(json_data={"foo": "bar"})
        monkeypatch.setattr(git_mod.requests, "get", lambda *a, **k: resp)
        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_github_file_content("https://github.com/o/r", "f.py")

    def test_invalid_url_raises(self, monkeypatch):
        from api.clients import git as git_mod

        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_github_file_content("not-a-url", "f.py")

    def test_invalid_json_response(self, monkeypatch):
        from api.clients import git as git_mod

        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        resp.raise_for_status = lambda: None
        monkeypatch.setattr(git_mod.requests, "get", lambda *a, **k: resp)
        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_github_file_content("https://github.com/o/r", "f.py")


# ===========================================================================
# api.clients.git — GitLab file content
# ===========================================================================
class TestGetGitlabFileContent:
    def test_success_gitlab_com(self, monkeypatch):
        from api.clients import git as git_mod

        project_info = _FakeResp(json_data={"default_branch": "develop"})
        file_resp = _FakeResp(text="file contents here")
        responses = iter([project_info, file_resp])
        monkeypatch.setattr(git_mod.requests, "get", lambda *a, **k: next(responses))

        out = git_mod.get_gitlab_file_content(
            "https://gitlab.com/group/project", "src/main.py",
        )
        assert out == "file contents here"

    def test_token_header_sent(self, monkeypatch):
        from api.clients import git as git_mod

        project_info = _FakeResp(json_data={"default_branch": "main"})
        file_resp = _FakeResp(text="content")
        responses = iter([project_info, file_resp])
        captured: list[dict] = []

        def fake_get(url, headers=None, **kw):
            captured.append({"url": url, "headers": headers})
            return next(responses)

        monkeypatch.setattr(git_mod.requests, "get", fake_get)
        git_mod.get_gitlab_file_content(
            "https://gitlab.com/g/p", "f.py", access_token="tok",
        )
        # Both calls should carry the PRIVATE-TOKEN header.
        for c in captured:
            assert c["headers"]["PRIVATE-TOKEN"] == "tok"

    def test_self_hosted_with_port(self, monkeypatch):
        from api.clients import git as git_mod

        project_info = _FakeResp(json_data={"default_branch": "main"})
        file_resp = _FakeResp(text="content")
        responses = iter([project_info, file_resp])
        captured: list[str] = []

        def fake_get(url, headers=None, **kw):
            captured.append(url)
            return next(responses)

        monkeypatch.setattr(git_mod.requests, "get", fake_get)
        git_mod.get_gitlab_file_content(
            "https://gitlab.local:8443/g/p", "f.py",
        )
        assert "gitlab.local:8443" in captured[0]

    def test_project_info_failure_falls_back_to_main(self, monkeypatch):
        from api.clients import git as git_mod

        project_info = _FakeResp(status_code=500)
        file_resp = _FakeResp(text="content")
        responses = iter([project_info, file_resp])
        captured: list[str] = []

        def fake_get(url, headers=None, **kw):
            captured.append(url)
            return next(responses)

        monkeypatch.setattr(git_mod.requests, "get", fake_get)
        out = git_mod.get_gitlab_file_content(
            "https://gitlab.com/g/p", "f.py",
        )
        assert out == "content"
        # The file-fetch URL should use ref=main (the fallback branch).
        assert "ref=main" in captured[1]

    def test_project_info_exception_falls_back_to_main(self, monkeypatch):
        from api.clients import git as git_mod

        file_resp = _FakeResp(text="content")
        responses = iter([None, file_resp])

        def fake_get(url, headers=None, **kw):
            val = next(responses)
            if val is None:
                raise ConnectionError("boom")
            return val

        monkeypatch.setattr(git_mod.requests, "get", fake_get)
        out = git_mod.get_gitlab_file_content(
            "https://gitlab.com/g/p", "f.py",
        )
        assert out == "content"

    def test_gitlab_api_error_message_in_body(self, monkeypatch):
        from api.clients import git as git_mod

        project_info = _FakeResp(json_data={"default_branch": "main"})
        file_resp = _FakeResp(text='{"message": "404 File Not Found"}')
        responses = iter([project_info, file_resp])
        monkeypatch.setattr(git_mod.requests, "get", lambda *a, **k: next(responses))
        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_gitlab_file_content(
                "https://gitlab.com/g/p", "f.py",
            )

    def test_file_fetch_request_exception_raises(self, monkeypatch):
        from api.clients import git as git_mod
        from requests.exceptions import RequestException

        project_info = _FakeResp(json_data={"default_branch": "main"})
        responses = iter([project_info, None])

        def fake_get(url, headers=None, **kw):
            val = next(responses)
            if val is None:
                raise RequestException("network down")
            return val

        monkeypatch.setattr(git_mod.requests, "get", fake_get)
        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_gitlab_file_content(
                "https://gitlab.com/g/p", "f.py",
            )

    def test_invalid_url_raises(self, monkeypatch):
        from api.clients import git as git_mod

        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_gitlab_file_content("not-a-url", "f.py")

    def test_too_few_path_parts_raises(self, monkeypatch):
        from api.clients import git as git_mod

        with pytest.raises(ValueError, match="Failed to get file content"):
            git_mod.get_gitlab_file_content("https://gitlab.com", "f.py")


# ===========================================================================
# api.clients.git — dispatcher
# ===========================================================================
class TestGetFileContentDispatcher:
    def test_github_dispatches(self, monkeypatch):
        from api.clients import git as git_mod

        called: list = []
        monkeypatch.setattr(
            git_mod, "get_github_file_content",
            lambda *a, **k: called.append(a) or "gh-content",
        )
        out = git_mod.get_file_content("u", "f", "github")
        assert out == "gh-content"
        assert called == [("u", "f", None)]

    def test_gitlab_dispatches(self, monkeypatch):
        from api.clients import git as git_mod

        monkeypatch.setattr(
            git_mod, "get_gitlab_file_content", lambda *a, **k: "gl-content",
        )
        assert git_mod.get_file_content("u", "f", "gitlab") == "gl-content"

    def test_unsupported_type_raises(self):
        from api.clients.git import get_file_content

        with pytest.raises(ValueError, match="Unsupported repository type"):
            get_file_content("u", "f", "bitbucket")


# ===========================================================================
# api.integrations._git_base — GitConnector
# ===========================================================================
class TestGitConnectorExtractRepoName:
    def test_github_url(self):
        from api.integrations._git_base import GitConnector

        assert GitConnector.extract_repo_name(
            "https://github.com/owner/repo", "github") == "owner_repo"

    def test_gitlab_url_with_git_suffix(self):
        from api.integrations._git_base import GitConnector

        assert GitConnector.extract_repo_name(
            "https://gitlab.com/group/project.git", "gitlab") == "group_project"

    def test_trailing_slash(self):
        from api.integrations._git_base import GitConnector

        assert GitConnector.extract_repo_name(
            "https://github.com/owner/repo/", "github") == "owner_repo"

    def test_unknown_type(self):
        from api.integrations._git_base import GitConnector

        # Unknown type: just the last segment.
        assert GitConnector.extract_repo_name(
            "https://example.com/owner/repo", "bitbucket") == "repo"


class TestGitConnectorCloneDir:
    def test_clone_dir_uses_adalflow_root(self, tmp_path, monkeypatch):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"

            def __init__(self, config=None):
                self.config = dict(config or {})

        conn = _Conn(config={})
        d = conn._clone_dir("https://github.com/o/repo")
        assert "repos" in d
        assert "o_repo" in d

    def test_clone_dir_falls_back_to_expanduser(self, tmp_path, monkeypatch):
        # When adalflow import fails, _clone_dir uses os.path.expanduser.
        from api.integrations import _git_base
        import builtins

        class _Conn(_git_base.GitConnector):
            repo_type = "gitlab"

            def __init__(self, config=None):
                self.config = dict(config or {})

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "adalflow.utils":
                raise ImportError("no adalflow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        conn = _Conn(config={})
        d = conn._clone_dir("https://gitlab.com/g/p")
        assert "repos" in d
        assert "g_p" in d


class TestGitConnectorFileTree:
    def test_file_tree_bounded(self, tmp_path):
        from api.integrations._git_base import GitConnector

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b")
        tree = GitConnector._file_tree(str(tmp_path), max_depth=1, max_entries=10)
        assert "a.txt" in tree
        assert "sub/" in tree

    def test_file_tree_truncated(self, tmp_path):
        from api.integrations._git_base import GitConnector

        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("x")
        tree = GitConnector._file_tree(str(tmp_path), max_depth=1, max_entries=2)
        assert "truncated" in tree

    def test_find_readme(self, tmp_path):
        from api.integrations._git_base import GitConnector

        (tmp_path / "README.md").write_text("readme content")
        assert GitConnector._find_readme(str(tmp_path)) == "readme content"

    def test_find_readme_none(self, tmp_path):
        from api.integrations._git_base import GitConnector

        assert GitConnector._find_readme(str(tmp_path)) is None


class TestGitConnectorBuildMarkdown:
    def test_with_readme(self, tmp_path):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"

            def __init__(self, config=None):
                self.config = dict(config or {})

        (tmp_path / "README.md").write_text("# Title")
        (tmp_path / "src.py").write_text("code")
        md = _Conn(config={})._build_markdown("https://github.com/o/repo", str(tmp_path))
        assert "o/repo" in md
        assert "## README" in md
        assert "## File tree" in md

    def test_without_readme(self, tmp_path):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "gitlab"

            def __init__(self, config=None):
                self.config = dict(config or {})

        md = _Conn(config={})._build_markdown("https://gitlab.com/g/p", str(tmp_path))
        assert "No README found" in md


class TestGitConnectorGitPull:
    def test_pull_builds_markdown(self, tmp_path, monkeypatch):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"
            default_api_base = "https://api.github.com"

            def __init__(self, config=None):
                self.config = dict(config or {})

        conn = _Conn(config={})

        # Patch _clone_dir to point at our tmp_path so download_repo writes there.
        monkeypatch.setattr(conn, "_clone_dir", lambda url: str(tmp_path))

        # Patch download_repo inside git_pull to create the .git dir.
        def fake_download(repo_url, local_path, **kw):
            os.makedirs(os.path.join(local_path, ".git"), exist_ok=True)
            Path(local_path, "README.md").write_text("# hello")
            return "cloned"

        monkeypatch.setattr(
            "api.clients.git.download_repo", fake_download,
        )
        result = conn.git_pull("https://github.com/o/repo")
        assert result["title"] == "o/repo"
        assert result["repo_type"] == "github"
        assert result["repo_url"] == "https://github.com/o/repo"
        assert "## README" in result["markdown"]
        assert result["local_path"] == str(tmp_path)

    def test_pull_with_opts_token_override(self, tmp_path, monkeypatch):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "gitlab"
            default_api_base = "https://gitlab.com"

            def __init__(self, config=None):
                self.config = dict(config or {})

        conn = _Conn(config={})
        monkeypatch.setattr(conn, "_clone_dir", lambda url: str(tmp_path))

        captured: dict = {}

        def fake_download(repo_url, local_path, repo_type=None, access_token=None, **kw):
            captured["token"] = access_token
            os.makedirs(os.path.join(local_path, ".git"), exist_ok=True)
            return "ok"

        monkeypatch.setattr("api.clients.git.download_repo", fake_download)
        conn.git_pull("https://gitlab.com/g/p", opts={"token": "override-tok"})
        assert captured["token"] == "override-tok"


class TestGitConnectorListSpaces:
    def test_no_token_returns_empty(self):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"
            default_api_base = "https://api.github.com"

            def __init__(self, config=None):
                self.config = dict(config or {})

        conn = _Conn(config={})
        assert conn.git_list_spaces() == []

    def test_list_with_pagination(self, monkeypatch):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"
            default_api_base = "https://api.github.com"

            def __init__(self, config=None):
                self.config = dict(config or {})

            def list_repos_url(self, api_base):
                return f"{api_base}/user/repos?per_page=100"

            def auth_headers(self):
                return {"Authorization": "token tok"}

        conn = _Conn(config={"token": "tok"})

        # Return a list (GitHub-style); the first page yields one repo and no
        # next-link so listing stops after page 1.
        page1 = [
            {"id": 1, "full_name": "o/a", "html_url": "https://github.com/o/a"},
            {"id": 2, "full_name": "o/b", "html_url": "https://github.com/o/b"},
        ]
        responses = iter([
            _FakeResp(status_code=200, json_data=page1),
        ])

        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)

        def fake_get(url, **kw):
            return next(responses)

        # Patch the local ``requests`` import inside git_list_spaces.
        monkeypatch.setitem(sys.modules, "requests", MagicMock(get=fake_get))
        out = conn.git_list_spaces()
        assert len(out) == 2
        assert out[0]["title"] == "o/a"
        assert out[1]["title"] == "o/b"

    def test_list_http_error_breaks(self, monkeypatch):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"
            default_api_base = "https://api.github.com"

            def __init__(self, config=None):
                self.config = dict(config or {})

            def list_repos_url(self, api_base):
                return f"{api_base}/user/repos"

            def auth_headers(self):
                return {}

        conn = _Conn(config={"token": "tok"})

        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)

        resp = _FakeResp(status_code=401, text="unauthorized")
        monkeypatch.setitem(sys.modules, "requests", MagicMock(get=lambda *a, **k: resp))
        assert conn.git_list_spaces() == []

    def test_list_non_list_results_breaks(self, monkeypatch):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"
            default_api_base = "https://api.github.com"

            def __init__(self, config=None):
                self.config = dict(config or {})

            def list_repos_url(self, api_base):
                return f"{api_base}/user/repos"

            def auth_headers(self):
                return {}

        conn = _Conn(config={"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)
        # A dict without "results" key -> non-list -> break.
        resp = _FakeResp(status_code=200, json_data={"error": "weird"})
        monkeypatch.setitem(sys.modules, "requests", MagicMock(get=lambda *a, **k: resp))
        assert conn.git_list_spaces() == []

    def test_list_follows_next_link(self, monkeypatch):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"
            default_api_base = "https://api.github.com"

            def __init__(self, config=None):
                self.config = dict(config or {})

            def list_repos_url(self, api_base):
                return f"{api_base}/page1"

            def auth_headers(self):
                return {}

        conn = _Conn(config={"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)

        p1 = _FakeResp(status_code=200, json_data={
            "results": [{"id": 1, "full_name": "o/a", "html_url": "u"}],
            "next": "https://api.github.com/page2",
        })
        p2 = _FakeResp(status_code=200, json_data={"results": []})
        responses = iter([p1, p2])
        monkeypatch.setitem(
            sys.modules, "requests",
            MagicMock(get=lambda *a, **k: next(responses)),
        )
        out = conn.git_list_spaces()
        assert len(out) == 1

    def test_list_next_as_dict_href(self, monkeypatch):
        from api.integrations import _git_base

        class _Conn(_git_base.GitConnector):
            repo_type = "github"
            default_api_base = "https://api.github.com"

            def __init__(self, config=None):
                self.config = dict(config or {})

            def list_repos_url(self, api_base):
                return f"{api_base}/page1"

            def auth_headers(self):
                return {}

        conn = _Conn(config={"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)

        p1 = _FakeResp(status_code=200, json_data={
            "results": [{"id": 1, "full_name": "o/a", "html_url": "u"}],
            "links": {"next": {"href": "https://api.github.com/page2"}},
        })
        p2 = _FakeResp(status_code=200, json_data={"results": []})
        responses = iter([p1, p2])
        monkeypatch.setitem(
            sys.modules, "requests",
            MagicMock(get=lambda *a, **k: next(responses)),
        )
        out = conn.git_list_spaces()
        assert len(out) == 1


# ===========================================================================
# api.integrations.github — GitHubConnector
# ===========================================================================
class TestGitHubConnector:
    def _make(self, config=None):
        from api.integrations.github import GitHubConnector
        return GitHubConnector(config or {})

    def test_is_configured_with_token(self):
        assert self._make({"token": "tok"}).is_configured() is True

    def test_is_configured_with_url(self):
        assert self._make({"url": "https://gh.enterprise"}).is_configured() is True

    def test_is_configured_empty(self):
        assert self._make({}).is_configured() is False

    def test_list_repos_url_public(self):
        c = self._make()
        u = c.list_repos_url("https://api.github.com")
        assert "api.github.com/user/repos" in u
        assert "per_page=100" in u

    def test_list_repos_url_enterprise(self):
        c = self._make()
        u = c.list_repos_url("https://gh.enterprise")
        assert "gh.enterprise/api/v3/user/repos" in u

    def test_auth_headers_with_token(self):
        c = self._make({"token": "tok"})
        assert c.auth_headers() == {"Authorization": "token tok"}

    def test_auth_headers_no_token(self):
        c = self._make()
        assert c.auth_headers() == {}

    def test_test_no_token(self):
        c = self._make({})
        out = c.test()
        assert out["success"] is True
        assert "public repos only" in out["message"]

    def test_test_success(self, monkeypatch):
        c = self._make({"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)
        resp = _FakeResp(status_code=200, json_data={"login": "user1"})
        monkeypatch.setitem(
            sys.modules, "requests", MagicMock(get=lambda *a, **k: resp),
        )
        out = c.test()
        assert out["success"] is True
        assert "user1" in out["message"]

    def test_test_http_error(self, monkeypatch):
        c = self._make({"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)
        resp = _FakeResp(status_code=401)
        monkeypatch.setitem(
            sys.modules, "requests", MagicMock(get=lambda *a, **k: resp),
        )
        out = c.test()
        assert out["success"] is False
        assert "HTTP 401" in out["message"]

    def test_test_exception(self, monkeypatch):
        c = self._make({"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)

        def boom(*a, **k):
            raise ConnectionError("down")

        monkeypatch.setitem(sys.modules, "requests", MagicMock(get=boom))
        out = c.test()
        assert out["success"] is False
        assert "connection failed" in out["message"]

    def test_test_enterprise_url_adds_api_v3(self, monkeypatch):
        c = self._make({"token": "tok", "url": "https://gh.enterprise"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)

        captured: list[str] = []

        def fake_get(url, **kw):
            captured.append(url)
            return _FakeResp(status_code=200, json_data={"login": "x"})

        monkeypatch.setitem(sys.modules, "requests", MagicMock(get=fake_get))
        c.test()
        assert any("/api/v3/user" in u for u in captured)

    def test_list_spaces_delegates(self, monkeypatch):
        c = self._make({"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)
        resp = _FakeResp(status_code=200, json_data=[
            {"id": 1, "full_name": "o/a", "html_url": "u"},
        ])
        monkeypatch.setitem(sys.modules, "requests", MagicMock(get=lambda *a, **k: resp))
        out = c.list_spaces()
        assert len(out) == 1
        assert out[0]["title"] == "o/a"

    def test_pull(self, tmp_path, monkeypatch):
        c = self._make({"token": "tok"})
        monkeypatch.setattr(c, "_clone_dir", lambda url: str(tmp_path))

        def fake_download(repo_url, local_path, **kw):
            os.makedirs(os.path.join(local_path, ".git"), exist_ok=True)
            return "ok"

        monkeypatch.setattr("api.clients.git.download_repo", fake_download)
        result = c.pull("https://github.com/o/repo")
        assert result["repo_type"] == "github"
        assert result["title"] == "o/repo"


# ===========================================================================
# api.integrations.gitlab — GitLabConnector
# ===========================================================================
class TestGitLabConnector:
    def _make(self, config=None):
        from api.integrations.gitlab import GitLabConnector
        return GitLabConnector(config or {})

    def test_is_configured_with_token(self):
        assert self._make({"token": "tok"}).is_configured() is True

    def test_is_configured_empty(self):
        assert self._make({}).is_configured() is False

    def test_list_repos_url(self):
        c = self._make()
        u = c.list_repos_url("https://gitlab.com")
        assert "gitlab.com/api/v4/projects" in u
        assert "membership=true" in u
        assert "per_page=100" in u

    def test_auth_headers_with_token(self):
        c = self._make({"token": "tok"})
        assert c.auth_headers() == {"PRIVATE-TOKEN": "tok"}

    def test_auth_headers_no_token(self):
        c = self._make()
        assert c.auth_headers() == {}

    def test_parse_repo_entry_gitlab_http_url(self):
        c = self._make()
        item = {
            "id": 5,
            "path_with_namespace": "group/proj",
            "web_url": "https://gitlab.com/group/proj",
            "http_url_to_repo": "https://gitlab.com/group/proj.git",
        }
        entry = c._parse_repo_entry(item)
        assert entry["url"] == "https://gitlab.com/group/proj"
        assert entry["title"] == "group/proj"
        assert entry["repo_id"] == "5"

    def test_parse_repo_entry_uses_http_url_when_web_url_missing(self):
        c = self._make()
        item = {
            "id": 5,
            "path_with_namespace": "group/proj",
            "http_url_to_repo": "https://gitlab.com/group/proj.git",
        }
        entry = c._parse_repo_entry(item)
        assert entry["url"] == "https://gitlab.com/group/proj.git"

    def test_test_no_token(self):
        c = self._make({})
        out = c.test()
        assert out["success"] is True
        assert "public repos only" in out["message"]

    def test_test_success(self, monkeypatch):
        c = self._make({"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)
        resp = _FakeResp(status_code=200, json_data={"username": "user1"})
        monkeypatch.setitem(
            sys.modules, "requests", MagicMock(get=lambda *a, **k: resp),
        )
        out = c.test()
        assert out["success"] is True
        assert "user1" in out["message"]

    def test_test_http_error(self, monkeypatch):
        c = self._make({"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)
        resp = _FakeResp(status_code=401)
        monkeypatch.setitem(
            sys.modules, "requests", MagicMock(get=lambda *a, **k: resp),
        )
        out = c.test()
        assert out["success"] is False
        assert "HTTP 401" in out["message"]

    def test_test_exception(self, monkeypatch):
        c = self._make({"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)

        def boom(*a, **k):
            raise ConnectionError("down")

        monkeypatch.setitem(sys.modules, "requests", MagicMock(get=boom))
        out = c.test()
        assert out["success"] is False
        assert "connection failed" in out["message"]

    def test_list_spaces_delegates(self, monkeypatch):
        c = self._make({"token": "tok"})
        import api.config.ssl as ssl_mod
        monkeypatch.setattr(ssl_mod, "requests_verify", lambda: True)
        resp = _FakeResp(status_code=200, json_data=[
            {"id": 1, "path_with_namespace": "g/p", "web_url": "u", "http_url_to_repo": "u2"},
        ])
        monkeypatch.setitem(sys.modules, "requests", MagicMock(get=lambda *a, **k: resp))
        out = c.list_spaces()
        assert len(out) == 1
        assert out[0]["title"] == "g/p"

    def test_pull(self, tmp_path, monkeypatch):
        c = self._make({"token": "tok"})
        monkeypatch.setattr(c, "_clone_dir", lambda url: str(tmp_path))

        def fake_download(repo_url, local_path, **kw):
            os.makedirs(os.path.join(local_path, ".git"), exist_ok=True)
            return "ok"

        monkeypatch.setattr("api.clients.git.download_repo", fake_download)
        result = c.pull("https://gitlab.com/g/p")
        assert result["repo_type"] == "gitlab"
        assert result["title"] == "g/p"
