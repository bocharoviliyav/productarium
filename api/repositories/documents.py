"""Document reading + clone orchestration (DatabaseManager).

Split out of the former ``api/data_pipeline.py`` (Step 5). Owns:
- ``count_tokens``: tiktoken-based token estimation.
- ``read_all_documents``: recursive directory walk with include/exclude
  filters, producing lightweight ``Document`` records with file metadata.
- ``DatabaseManager``: orchestrates clone (via ``api.clients.git.download_repo``)
  -> read, with paths under ``~/.adalflow`` kept for backward compatibility
  with existing on-disk clones.

Embedding/indexing no longer happens here: generated docs are indexed into
the pgvector memory backend (``api.memory``) by the docgen pipeline, and the
former FAISS ``LocalDB`` persistence was removed together with adalflow.

The git clone lives in ``api.clients.git`` (one-way dependency: this module
imports ``download_repo`` from there; git does NOT import this module).
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import tiktoken

from api.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES

logger = logging.getLogger(__name__)

# Maximum token limit for OpenAI embedding models
MAX_EMBEDDING_TOKENS = 8192
# Cap on one file read (chars) during document collection. Files anywhere near
# the token limits are skipped anyway; the cap only stops a huge file (or a
# special file reached through a symlinked directory) from being slurped into
# memory whole before that decision is made.
_READ_MAX_CHARS = 1_000_000


def _read_confined_file(root_real: str, file_path: str) -> Optional[str]:
    """Symlink-safe, root-confined read of one clone file (capped), or None.

    Git preserves symlinks, so a malicious repo can plant ``evil.py ->
    /etc/passwd`` inside a clone. Every collected file must therefore be a
    regular file INSIDE the clone root: symlinked entries are skipped, the
    realpath must stay under ``root_real`` (already realpath-resolved by the
    caller), and the final open goes through ``open_read_nofollow`` (O_NOFOLLOW)
    so a symlink swapped onto the last component cannot win the race.
    """
    from api.utils.fs import open_read_nofollow

    try:
        if os.path.islink(file_path):
            return None
        real = os.path.realpath(file_path)
        if os.path.commonpath([root_real, real]) != root_real:
            return None
        with open_read_nofollow(real) as f:
            return f.read(_READ_MAX_CHARS)
    except (OSError, ValueError):
        # Unreadable/undecodable/escaping entries are skipped, never fatal.
        return None

# Root directory for clones and legacy artifacts. Historically the
# adalflow default root; kept as the same path so existing clones on disk
# keep working after the langchain migration.
DEFAULT_REPO_ROOT = os.path.expanduser("~/.adalflow")


@dataclass
class Document:
    """A single source file read from a repository.

    Minimal replacement for the former adalflow ``Document``: plain text plus
    the file metadata the docgen pipeline (blob building, file analysis,
    chunking) consumes.
    """

    text: str
    meta_data: Dict[str, Any] = field(default_factory=dict)

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text (str): The text to count tokens for.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Use the OpenAI embedding model encoding (text-embedding-3-small).
        # LM Studio/vLLM embedders all accept cl100k_base-equivalent
        # tokenization, so a single encoding is accurate enough for budgeting.
        encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        # Rough approximation: 4 characters per token
        return len(text) // 4

def read_all_documents(path: str,
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.
        excluded_dirs
            Overrides the default configuration if provided.
        excluded_files (List[str], optional): List of file patterns to exclude from processing.
            Overrides the default configuration if provided.
        included_dirs (List[str], optional): List of directories to include exclusively.
            When provided, only files in these directories will be processed.
        included_files (List[str], optional): List of file patterns to include exclusively.
            When provided, only files matching these patterns will be processed.

    Returns:
        list: A list of Document objects with metadata.
    """
    documents = []
    # File extensions to look for, prioritizing code files
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    # Determine filtering mode: inclusion or exclusion
    use_inclusion_mode = (included_dirs is not None and len(included_dirs) > 0) or (included_files is not None and len(included_files) > 0)

    if use_inclusion_mode:
        # Inclusion mode: only process specified directories and files
        final_included_dirs = set(included_dirs) if included_dirs else set()
        final_included_files = set(included_files) if included_files else set()

        logger.info(f"Using inclusion mode")
        logger.info(f"Included directories: {list(final_included_dirs)}")
        logger.info(f"Included files: {list(final_included_files)}")

        # Convert to lists for processing
        included_dirs = list(final_included_dirs)
        included_files = list(final_included_files)
        excluded_dirs = []
        excluded_files = []
    else:
        # Exclusion mode: use default exclusions plus any additional ones
        final_excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        final_excluded_files = set(DEFAULT_EXCLUDED_FILES)

        # Add any additional excluded directories from config
        if "file_filters" in configs and "excluded_dirs" in configs["file_filters"]:
            final_excluded_dirs.update(configs["file_filters"]["excluded_dirs"])

        # Add any additional excluded files from config
        if "file_filters" in configs and "excluded_files" in configs["file_filters"]:
            final_excluded_files.update(configs["file_filters"]["excluded_files"])

        # Add any explicitly provided excluded directories and files
        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)

        if excluded_files is not None:
            final_excluded_files.update(excluded_files)

        # Convert back to lists for compatibility
        excluded_dirs = list(final_excluded_dirs)
        excluded_files = list(final_excluded_files)
        included_dirs = []
        included_files = []

        logger.info(f"Using exclusion mode")
        logger.info(f"Excluded directories: {excluded_dirs}")
        logger.info(f"Excluded files: {excluded_files}")

    logger.info(f"Reading documents from {path}")

    def should_process_file(file_path: str, use_inclusion: bool, included_dirs: List[str], included_files: List[str],
                           excluded_dirs: List[str], excluded_files: List[str]) -> bool:
        """
        Determine if a file should be processed based on inclusion/exclusion rules.

        Args:
            file_path (str): The file path to check
            use_inclusion (bool): Whether to use inclusion mode
            included_dirs (List[str]): List of directories to include
            included_files (List[str]): List of files to include
            excluded_dirs (List[str]): List of directories to exclude
            excluded_files (List[str]): List of files to exclude

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        file_path_parts = os.path.normpath(file_path).split(os.sep)
        file_name = os.path.basename(file_path)

        if use_inclusion:
            # Inclusion mode: file must be in included directories or match included files
            is_included = False

            # Check if file is in an included directory
            if included_dirs:
                for included in included_dirs:
                    clean_included = included.strip("./").rstrip("/")
                    if clean_included in file_path_parts:
                        is_included = True
                        break

            # Check if file matches included file patterns
            if not is_included and included_files:
                for included_file in included_files:
                    if file_name == included_file or file_name.endswith(included_file):
                        is_included = True
                        break

            # If no inclusion rules are specified for a category, allow all files from that category
            if not included_dirs and not included_files:
                is_included = True
            elif not included_dirs and included_files:
                # Only file patterns specified, allow all directories
                pass  # is_included is already set based on file patterns
            elif included_dirs and not included_files:
                # Only directory patterns specified, allow all files in included directories
                pass  # is_included is already set based on directory patterns

            return is_included
        else:
            # Exclusion mode: file must not be in excluded directories or match excluded files
            is_excluded = False

            # Check if file is in an excluded directory
            for excluded in excluded_dirs:
                clean_excluded = excluded.strip("./").rstrip("/")
                if clean_excluded in file_path_parts:
                    is_excluded = True
                    break

            # Check if file matches excluded file patterns
            if not is_excluded:
                for excluded_file in excluded_files:
                    if file_name == excluded_file:
                        is_excluded = True
                        break

            return not is_excluded

    # Clone root (resolved once) used to confine every collected file.
    root_real = os.path.realpath(path)

    # Process code files first
    for ext in code_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                content = _read_confined_file(root_real, file_path)
                if content is None:
                    logger.debug("Skipping unreadable/symlinked file %s", file_path)
                    continue
                relative_path = os.path.relpath(file_path, path)

                # Determine if this is an implementation file
                is_implementation = (
                    not relative_path.startswith("test_")
                    and not relative_path.startswith("app_")
                    and "test" not in relative_path.lower()
                )

                # Check token count
                token_count = count_tokens(content)
                if token_count > MAX_EMBEDDING_TOKENS * 10:
                    logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                    continue

                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:],
                        "is_code": True,
                        "is_implementation": is_implementation,
                        "title": relative_path,
                        "token_count": token_count,
                    },
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                content = _read_confined_file(root_real, file_path)
                if content is None:
                    logger.debug("Skipping unreadable/symlinked file %s", file_path)
                    continue
                relative_path = os.path.relpath(file_path, path)

                # Check token count
                token_count = count_tokens(content)
                if token_count > MAX_EMBEDDING_TOKENS:
                    logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                    continue

                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:],
                        "is_code": False,
                        "is_implementation": False,
                        "title": relative_path,
                        "token_count": token_count,
                    },
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    return documents

def prepare_data_pipeline():
    """Deprecated alias kept for backward compatibility (returns None).

    The adalflow TextSplitter + ToEmbeddings pipeline was removed in the
    langchain migration; indexing now goes through ``api.memory``
    (pgvector). Kept as a stub so older imports keep working.
    """
    return None


class DatabaseManager:
    """
    Manages repository clones and document reading (no local index).

    Historically owned FAISS ``LocalDB`` creation/loading/persistence; that
    was removed with adalflow. What remains: cloning (via
    ``api.clients.git.download_repo``) and reading files into ``Document``
    records for the docgen pipeline. Existing ``.pkl`` database files under
    ``~/.adalflow/databases`` are ignored (stale FAISS artifacts).
    """

    def __init__(self):
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None,
                         excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                         included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        Create a new database from the repository.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively

        Returns:
            List[Document]: List of Document objects
        """
        self.reset_database()
        self._create_repo(repo_url_or_path, repo_type, access_token)
        return self.prepare_db_index(excluded_dirs=excluded_dirs, excluded_files=excluded_files,
                                   included_dirs=included_dirs, included_files=included_files)

    def reset_database(self):
        """
        Reset the manager to its initial state.
        """
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        # Extract owner and repo name to create unique identifier
        url_parts = repo_url_or_path.rstrip('/').split('/')

        if repo_type in ["github", "gitlab"] and len(url_parts) >= 5:
            # GitHub URL format: https://github.com/owner/repo
            # GitLab URL format: https://gitlab.com/owner/repo or https://gitlab.com/group/subgroup/repo
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name

    def _create_repo(
        self,
        repo_url_or_path: str,
        repo_type: str = None,
        access_token: str = None,
        force_refresh: bool = False,
    ) -> None:
        """
        Download and prepare all paths.
        Paths:
        ~/.adalflow/repos/{owner}_{repo_name} (for url, local path will be the same)

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories
            force_refresh (bool): forwarded to ``download_repo`` so an existing
                clone is fetched+reset to the latest remote tip instead of being
                reused as-is. Used by artifact (re)generation; the Ask/RAG path
                leaves it False.
        """
        logger.info(f"Preparing repo storage for {repo_url_or_path}...")

        try:
            # Strip whitespace to handle URLs with leading/trailing spaces
            repo_url_or_path = repo_url_or_path.strip()

            root_path = DEFAULT_REPO_ROOT

            os.makedirs(root_path, exist_ok=True)
            # url
            if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
                # Extract the repository name from the URL
                repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type)
                logger.info(f"Extracted repo name: {repo_name}")

                save_repo_dir = os.path.join(root_path, "repos", repo_name)

                # download_repo now handles both fresh clones AND refresh of an
                # existing clone when force_refresh is set, so we always call it;
                # with force_refresh=False it short-circuits on an existing dir.
                # Lazy import: git clone is the only cross-package dependency
                # (repositories -> clients/git), kept one-way.
                from api.clients.git import download_repo

                download_repo(
                    repo_url_or_path,
                    save_repo_dir,
                    repo_type,
                    access_token,
                    force_refresh=force_refresh,
                )
            else:  # local path
                repo_name = os.path.basename(repo_url_or_path)
                save_repo_dir = repo_url_or_path

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
            }
            self.repo_url_or_path = repo_url_or_path
            logger.info(f"Repo paths: {self.repo_paths}")

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def prepare_db_index(self,
                        excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        Read the repository files into ``Document`` records.

        Retains its historical name (the FAISS index build it used to perform
        was removed with adalflow); now a thin wrapper over
        ``read_all_documents`` using the clone directory captured by
        ``_create_repo``.

        Args:
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively

        Returns:
            List[Document]: List of Document records
        """
        if not self.repo_paths:
            raise ValueError("No repository prepared; call _create_repo first.")
        return read_all_documents(
            self.repo_paths["save_repo_dir"],
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )

    def prepare_retriever(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None):
        """
        Prepare the retriever for a repository.
        This is a compatibility method for the isolated API.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories

        Returns:
            List[Document]: List of Document objects
        """
        return self.prepare_database(repo_url_or_path, repo_type, access_token)
