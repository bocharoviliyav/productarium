"""LLM client layer — external integration with OpenAI-compatible providers.

Members:
- ``openai_client`` — OpenAIClient (adalflow ModelClient wrapper for any
  OpenAI-compatible local/remote endpoint: LM Studio, llama.cpp, vLLM, etc.).
- ``ollama``       — OllamaDocumentProcessor (single-doc embedding workaround
  for adalflow's Ollama client, which lacks batch embedding support).
- ``git``          — GitHub/GitLab clone + remote file content APIs
  (``download_repo``, ``get_file_content``, ...).
"""

from api.clients.git import (
    download_repo,
    download_github_repo,
    get_file_content,
    get_github_file_content,
    get_gitlab_file_content,
)

__all__ = [
    "download_repo",
    "download_github_repo",
    "get_file_content",
    "get_github_file_content",
    "get_gitlab_file_content",
]
