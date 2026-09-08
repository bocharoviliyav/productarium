"""Client layer — external integrations.

Members:
- ``git`` — GitHub/GitLab clone + remote file content APIs
  (``download_repo``, ``get_file_content``, ...).

The LLM client now lives in ``api.llm`` (langchain ``ChatOpenAI`` for any
OpenAI-compatible local/remote endpoint: LM Studio, llama.cpp, vLLM, etc.).
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
