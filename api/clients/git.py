"""Git repository cloning + remote file content APIs (GitHub / GitLab).

Split out of the former ``api/data_pipeline.py`` (Step 5). Handles:
- Authenticated shallow clone (``--depth=1 --single-branch``) with token
  injection per host (GitHub ``<token>@``, GitLab ``oauth2:<token>@``).
- Force-refresh of an existing clone (fetch + reset --hard + clean) so
  artifact regeneration reads the latest remote tip.
- Remote file content retrieval via the GitHub / GitLab REST APIs (public
  ``github.com`` / ``gitlab.com`` or enterprise / self-hosted instances).

Token hygiene: ``_sanitize_git_stderr`` strips raw + URL-encoded tokens from
git stderr/stdout so clone errors logged or surfaced to the UI never leak the
credential. All network/subprocess calls are inside the functions (not at
import time) so this module imports cleanly without git/requests available.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
from urllib.parse import urlparse, urlunparse, quote

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


def _sanitize_git_stderr(text: str, access_token: str) -> str:
    """Strip raw + URL-encoded access tokens from a git stderr/stdout string."""
    if not access_token or not text:
        return text
    encoded_token = quote(access_token, safe='')
    return text.replace(access_token, "***TOKEN***").replace(encoded_token, "***TOKEN***")


def _is_git_repo(path: str) -> bool:
    """True if ``path`` looks like the root of a valid git working tree."""
    return bool(path) and os.path.isdir(os.path.join(path, ".git"))


def _build_authed_clone_url(repo_url: str, repo_type: str, access_token: str) -> str:
    """Inject an access token into a GitHub/GitLab HTTPS URL for clone/fetch."""
    if not access_token:
        return repo_url
    parsed = urlparse(repo_url)
    encoded_token = quote(access_token, safe='')
    if repo_type == "github":
        return urlunparse((parsed.scheme, f"{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))
    if repo_type == "gitlab":
        return urlunparse((parsed.scheme, f"oauth2:{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))
    return repo_url


def download_repo(
    repo_url: str,
    local_path: str,
    repo_type: str = None,
    access_token: str = None,
    force_refresh: bool = False,
) -> str:
    """
    Downloads a Git repository (GitHub or GitLab) to a specified local path.

    Args:
        repo_type(str): Type of repository
        repo_url (str): The URL of the Git repository to clone.
        local_path (str): The local directory where the repository will be cloned.
        access_token (str, optional): Access token for private repositories.
        force_refresh (bool): When True and ``local_path`` already holds a valid
            git repo, fetch the latest tip and hard-reset the working tree so a
            (re)generation reads the current remote commit instead of the stale
            first clone. Required for artifact regeneration freshness; left False
            on the Ask/RAG path to avoid a per-request fetch.

    Returns:
        str: The output message from the `git` command.
    """
    try:
        # Check if Git is installed
        logger.info(f"Preparing to clone repository to {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        dir_exists = os.path.exists(local_path) and os.listdir(local_path)

        # Existing valid repo: either refresh it (force_refresh) or reuse as-is.
        if dir_exists and _is_git_repo(local_path):
            if force_refresh:
                # Update the shallow clone to the latest remote tip. Best-effort:
                # on any failure we fall back to the existing checkout rather than
                # failing the whole generation, then optionally re-clone fresh.
                try:
                    fetch_url = _build_authed_clone_url(repo_url, repo_type, access_token)
                    logger.info("Refreshing existing repository (force_refresh): %s", local_path)
                    subprocess.run(
                        ["git", "-C", local_path, "fetch", "--depth=1", "origin", fetch_url],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    subprocess.run(
                        ["git", "-C", local_path, "reset", "--hard", "FETCH_HEAD"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    # Drop untracked/ignored files so removed files don't linger.
                    subprocess.run(
                        ["git", "-C", local_path, "clean", "-fdx"],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    logger.info("Repository refreshed to latest remote tip.")
                    return f"Refreshed existing repository at {local_path}"
                except subprocess.CalledProcessError as e:
                    err = _sanitize_git_stderr(e.stderr.decode('utf-8'), access_token)
                    logger.warning(
                        "force_refresh fetch/reset failed for %s (%s); removing and "
                        "re-cloning fresh.", local_path, err,
                    )
                    shutil.rmtree(local_path, ignore_errors=True)
                    # Fall through to the fresh-clone path below.
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "force_refresh refresh raised for %s (%s); removing and "
                        "re-cloning fresh.", local_path, e,
                    )
                    shutil.rmtree(local_path, ignore_errors=True)
            else:
                logger.info("Repository already exists at %s. Using existing repository.", local_path)
                return f"Using existing repository at {local_path}"
        elif dir_exists:
            # Directory exists but is not a valid git repo (corrupt/partial copy).
            if force_refresh:
                logger.warning(
                    "Existing path %s is not a git repo; removing and re-cloning.",
                    local_path,
                )
                shutil.rmtree(local_path, ignore_errors=True)
            else:
                logger.warning("Repository already exists at %s. Using existing repository.", local_path)
                return f"Using existing repository at {local_path}"

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)

        # Prepare the clone URL with access token if provided
        clone_url = _build_authed_clone_url(repo_url, repo_type, access_token)
        if access_token:
            logger.info("Using access token for authentication")

        # Clone the repository
        logger.info(f"Cloning repository from {repo_url} to {local_path}")
        # We use repo_url in the log to avoid exposing the token in logs
        result = subprocess.run(
            ["git", "clone", "--depth=1", "--single-branch", clone_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("Repository cloned successfully")
        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        error_msg = _sanitize_git_stderr(e.stderr.decode('utf-8'), access_token)
        raise ValueError(f"Error during cloning: {error_msg}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}")

# Alias for backward compatibility
download_github_repo = download_repo

def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitHub repository using the GitHub API.
    Supports both public GitHub (github.com) and GitHub Enterprise (custom domains).

    Args:
        repo_url (str): The URL of the GitHub repository
                       (e.g., "https://github.com/username/repo" or "https://github.company.com/username/repo")
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): GitHub personal access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not a valid GitHub URL
    """
    try:
        # Parse the repository URL to support both github.com and enterprise GitHub
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitHub repository URL")

        # Check if it's a GitHub-like URL structure
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format - expected format: https://domain/owner/repo")

        owner = path_parts[-2]
        repo = path_parts[-1].replace(".git", "")

        # Determine the API base URL
        if parsed_url.netloc == "github.com":
            # Public GitHub
            api_base = "https://api.github.com"
        else:
            # GitHub Enterprise - API is typically at https://domain/api/v3/
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v3"

        # Use GitHub API to get file content
        # The API endpoint for getting file content is: /repos/{owner}/{repo}/contents/{path}
        api_url = f"{api_base}/repos/{owner}/{repo}/contents/{file_path}"

        # Fetch file content from GitHub API
        headers = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        from api.timeout_config import resolve_git_file_content_timeout
        logger.info(f"Fetching file content from GitHub API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers, timeout=resolve_git_file_content_timeout())
            response.raise_for_status()
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")
        try:
            content_data = response.json()
        except json.JSONDecodeError:
            raise ValueError("Invalid response from GitHub API")

        # Check if we got an error response
        if "message" in content_data and "documentation_url" in content_data:
            raise ValueError(f"GitHub API error: {content_data['message']}")

        # GitHub API returns file content as base64 encoded string
        if "content" in content_data and "encoding" in content_data:
            if content_data["encoding"] == "base64":
                # The content might be split into lines, so join them first
                content_base64 = content_data["content"].replace("\n", "")
                content = base64.b64decode(content_base64).decode("utf-8")
                return content
            else:
                raise ValueError(f"Unexpected encoding: {content_data['encoding']}")
        else:
            raise ValueError("File content not found in GitHub API response")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitLab repository (cloud or self-hosted).

    Args:
        repo_url (str): The GitLab repo URL (e.g., "https://gitlab.com/username/repo" or "http://localhost/group/project")
        file_path (str): File path within the repository (e.g., "src/main.py")
        access_token (str, optional): GitLab personal access token

    Returns:
        str: File content

    Raises:
        ValueError: If anything fails
    """
    try:
        # Parse and validate the URL
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitLab repository URL")

        gitlab_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if parsed_url.port not in (None, 80, 443):
            gitlab_domain += f":{parsed_url.port}"
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("Invalid GitLab URL format — expected something like https://gitlab.domain.com/group/project")

        # Build project path and encode for API
        project_path = "/".join(path_parts).replace(".git", "")
        encoded_project_path = quote(project_path, safe='')

        # Encode file path
        encoded_file_path = quote(file_path, safe='')

        from api.timeout_config import resolve_git_file_content_timeout
        # Try to get the default branch from the project info
        default_branch = None
        try:
            project_info_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}"
            project_headers = {}
            if access_token:
                project_headers["PRIVATE-TOKEN"] = access_token

            project_response = requests.get(project_info_url, headers=project_headers, timeout=resolve_git_file_content_timeout())
            if project_response.status_code == 200:
                project_data = project_response.json()
                default_branch = project_data.get('default_branch', 'main')
                logger.info(f"Found default branch: {default_branch}")
            else:
                logger.warning(f"Could not fetch project info, using 'main' as default branch")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"Error fetching project info: {e}, using 'main' as default branch")
            default_branch = 'main'

        api_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}/repository/files/{encoded_file_path}/raw?ref={default_branch}"
        # Fetch file content from GitLab API
        headers = {}
        if access_token:
            headers["PRIVATE-TOKEN"] = access_token
        logger.info(f"Fetching file content from GitLab API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers, timeout=resolve_git_file_content_timeout())
            response.raise_for_status()
            content = response.text
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

        # Check for GitLab error response (JSON instead of raw file)
        if content.startswith("{") and '"message":' in content:
            try:
                error_data = json.loads(content)
                if "message" in error_data:
                    raise ValueError(f"GitLab API error: {error_data['message']}")
            except json.JSONDecodeError:
                pass

        return content

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_file_content(repo_url: str, file_path: str, repo_type: str = None, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Git repository (GitHub or GitLab).

    Args:
        repo_type (str): Type of repository
        repo_url (str): The URL of the repository
        file_path (str): The path to the file within the repository
        access_token (str, optional): Access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not valid
    """
    if repo_type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif repo_type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    else:
        raise ValueError("Unsupported repository type. Only GitHub and GitLab are supported.")
