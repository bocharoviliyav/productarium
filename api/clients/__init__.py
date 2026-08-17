"""LLM client layer — external integration with OpenAI-compatible providers.

Members:
- ``openai_client`` — OpenAIClient (adalflow ModelClient wrapper for any
  OpenAI-compatible local/remote endpoint: LM Studio, llama.cpp, vLLM, etc.).
- ``git``          — GitHub/GitLab clone + remote file content APIs
  (``download_repo``, ``get_file_content``, ...).
"""

from api.clients.git import (
    download_repo,
    get_file_content,
    get_github_file_content,
    get_gitlab_file_content,
)

__all__ = [
    "download_repo",
    "get_file_content",
    "get_github_file_content",
    "get_gitlab_file_content",
]
