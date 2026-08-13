from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from api.cognee._runtime import _COGNEE_AVAILABLE, _resolve_cognify_timeout, cognee
from api.cognee.config import apply_cognee_runtime_config

logger = logging.getLogger(__name__)

async def _empty_cognee_dataset(dataset_name: str) -> bool:
    """Clear all data + graph for a cognee dataset by name. Returns True on success.

    V1 path: ``cognee.forget(dataset=name)`` is the unified deletion API --
    it deletes the whole dataset (relational records, graph nodes/edges, vector
    embeddings) in one call. This replaces the legacy ``datasets.empty_dataset``
    + manual junction-row cleanup.

    Fallback: if ``forget`` raises (e.g. the embedding/tokenizer machinery it
    drags in for graph cleanup hits a KeyError, or the dataset name cannot be
    resolved), we resolve the dataset UUID via ``datasets.list_datasets`` and
    fall back to a DIRECT SQLAlchemy delete of the ``data`` + ``dataset_data``
    junction rows. That is enough to let a re-add succeed (cognify only needs
    the ``data`` rows gone); any orphaned graph nodes from a PARTIALLY
    cognified dataset are harmless (they reference a now-absent dataset).
    Best-effort: returns False only if BOTH paths fail.
    """
    try:
        forget_fn = getattr(cognee, "forget", None)
        if callable(forget_fn):
            try:
                await forget_fn(dataset=dataset_name)
                return True
            except Exception as e:
                # A "dataset not found" / "nothing to delete" is a success --
                # there is nothing to clear, and a re-add should proceed cleanly.
                msg = str(e).lower()
                if (
                    "not found" in msg
                    or "no dataset" in msg
                    or "does not exist" in msg
                    or "notexist" in msg.replace(" ", "")
                ):
                    return True
                logger.warning(
                    "_empty_cognee_dataset(%r): forget failed (%s); "
                    "falling back to direct DB delete of data rows.",
                    dataset_name, e,
                )
        # Fallback: resolve the dataset UUID by name, then direct DB delete.
        datasets_obj = getattr(cognee, "datasets", None)
        if datasets_obj is None:
            return False
        datasets = await datasets_obj.list_datasets()
        target = None
        for d in datasets:
            if getattr(d, "name", None) == dataset_name:
                target = d
                break
        if target is None:
            # Nothing to clear -- treat as success (dataset doesn't exist yet).
            return True
        return await _direct_delete_dataset_data(target.id)
    except Exception as e:  # pragma: no cover - depends on cognee version
        logger.warning("_empty_cognee_dataset(%r) failed: %s", dataset_name, e)
        return False


async def _direct_delete_dataset_data(dataset_id) -> bool:
    """Direct SQLAlchemy delete of a dataset's ``data`` + ``dataset_data`` rows.

    Bypasses ``cognee.datasets.delete_data`` / ``empty_dataset`` (which pull in
    the embedding pipeline and can fail with a tokenizer KeyError). Enough to
    let a re-add succeed -- cognify only needs the ``data`` rows gone. Returns
    True on success, False on error. Does NOT touch graph nodes (orphans are
    harmless: they reference an absent dataset).
    """
    try:
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.data.models import Data
        from cognee.modules.data.models.Dataset import DatasetData
        from sqlalchemy import delete, select
        eng = get_relational_engine()
        async with eng.get_async_session() as session:
            # Collect this dataset's data_ids via the junction, then delete.
            res = await session.scalars(
                select(DatasetData.data_id).where(DatasetData.dataset_id == dataset_id)
            )
            data_ids = list(res.all())
            if data_ids:
                await session.execute(
                    delete(DatasetData).where(DatasetData.dataset_id == dataset_id)
                )
                await session.execute(
                    delete(Data).where(Data.id.in_(data_ids))
                )
            await session.commit()
        return True
    except Exception as e:  # pragma: no cover - depends on cognee version
        logger.warning("_direct_delete_dataset_data(%s) failed: %s", dataset_id, e)
        return False


def _resolve_raw_data_location_path(raw_data_location: str) -> str:
    """Normalize a cognee ``raw_data_location`` into a plain filesystem path.

    cognee 1.2.x stores ``raw_data_location`` WITH a ``file://`` URI scheme
    prefix (e.g. ``file:///root/.adalflow/cognee_data/text_<hash>.txt``), but
    older rows and some code paths store a bare absolute path. ``os.path.exists``
    returns False on a ``file://`` string, so we MUST strip the scheme before
    any existence check -- otherwise the reconciler would prune every healthy
    row. Returns the original string if it's not a ``file://`` URI.
    """
    if not raw_data_location:
        return raw_data_location
    if raw_data_location.startswith("file://"):
        return raw_data_location[len("file://"):]
    return raw_data_location


async def _reconcile_stale_cognee_data() -> None:
    """One-time startup cleanup of cognee ``Data`` rows whose backing file is gone.

    Background: cognee stores each ingested document as a row in the ``data``
    table with ``raw_data_location`` pointing at a ``text_<hash>.txt`` file under
    ``data_root_directory``. If that directory is ephemeral (the old default was
    inside the installed cognee package, wiped on every Docker rebuild) but the
    Postgres ``data`` table is on a persistent volume, the table ends up
    referencing files that no longer exist. The next ``cognify`` then raises
    ``File not found: .../text_<hash>.txt`` for every stale row.

    This scans every dataset's data rows and deletes the ones whose
    ``raw_data_location`` file is missing, so the dataset can be re-ingested
    cleanly. Best-effort and non-fatal: a failure here only means some stale
    rows linger until the cat 3 / file-not-found retry path clears them lazily.

    Implementation notes:
    - ``raw_data_location`` may carry a ``file://`` URI scheme (cognee 1.2.x) OR
      a bare path. ``_resolve_raw_data_location_path`` normalizes it before the
      existence check so healthy rows are never pruned by mistake.
    - We delete the row DIRECTLY via SQLAlchemy (``data`` + ``dataset_data``
      junction) rather than ``cognee.datasets.delete_data``, because the latter
      drags in the embedding/tokenizer machinery (``has_data_related_nodes`` ->
      tokenize) and fails with a tokenizer KeyError when the embedding pipeline
      is misconfigured. A stale row (missing backing file) by definition never
      completed ``cognify``, so it has NO graph nodes/edges -- there is nothing
      to clean up in the graph, only the relational rows.
    """
    if not _COGNEE_AVAILABLE:
        return
    try:
        datasets_obj = getattr(cognee, "datasets", None)
        if datasets_obj is None:
            return
        datasets = await datasets_obj.list_datasets()
    except Exception as e:
        # On a fresh DB the schema may not have been created yet when this runs
        # (cognee.init()/setup() can defer table creation to the first write).
        # cognee raises ``DatabaseNotCreatedError: The database has not been
        # created yet. Please call `await setup()` first`` in that case. There
        # is nothing stale to reconcile on a fresh DB, so log at debug (not
        # error) and bail out cleanly instead of spamming the startup log.
        msg = str(e)
        if (
            "DatabaseNotCreatedError" in type(e).__name__
            or "has not been created" in msg
            or "call `await setup()`" in msg
            or "DatabaseNotCreatedError" in msg
        ):
            logger.debug(
                "cognee stale-data reconciliation skipped: relational schema not "
                "created yet (nothing stale on a fresh DB). Detail: %s", e,
            )
            return
        raise
    try:
        # Collect stale data_ids first, then delete in one session.
        stale_ids = []
        for d in datasets:
            list_data_fn = getattr(datasets_obj, "list_data", None)
            if not callable(list_data_fn):
                continue
            try:
                data_rows = await list_data_fn(d.id)
            except Exception as e:  # pragma: no cover - depends on cognee version
                logger.debug("reconcile: list_data(%s) failed: %s", d.id, e)
                continue
            for row in data_rows:
                loc = getattr(row, "raw_data_location", None)
                if not loc:
                    continue
                # Strip a ``file://`` URI scheme before the existence check so
                # healthy rows with the scheme-prefixed location are not pruned.
                path = _resolve_raw_data_location_path(loc)
                if os.path.exists(path):
                    continue  # backing file present -- healthy row
                stale_ids.append((d.id, row.id, loc))
        if not stale_ids:
            logger.debug("cognee stale-data reconciliation: no stale rows found.")
            return
        # Direct DB delete of stale ``data`` + ``dataset_data`` rows. Bypasses
        # cognee.datasets.delete_data (which needs the embedding pipeline).
        try:
            from cognee.infrastructure.databases.relational import get_relational_engine
            from cognee.modules.data.models import Data
            from cognee.modules.data.models.Dataset import DatasetData
            from sqlalchemy import delete
            eng = get_relational_engine()
            data_ids = [row_id for (_ds, row_id, _loc) in stale_ids]
            async with eng.get_async_session() as session:
                await session.execute(
                    delete(DatasetData).where(DatasetData.data_id.in_(data_ids))
                )
                await session.execute(
                    delete(Data).where(Data.id.in_(data_ids))
                )
                await session.commit()
        except Exception as e:  # pragma: no cover - depends on cognee version
            logger.warning(
                "cognee stale-data reconciliation: direct DB delete failed: %s. "
                "Stale rows will be cleared lazily by the add/retry path.", e,
            )
            return
        logger.info(
            "cognee stale-data reconciliation: pruned %d row(s) with missing "
            "backing files across %d dataset(s).",
            len(stale_ids), len(datasets),
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("cognee stale-data reconciliation failed (non-fatal): %s", e)


# --- Repo-path text extraction (cat 2: exclude .git + binary files) -------------
# cognee's ``text_loader.load`` does ``open(file_path, encoding=utf-8).read()``
# on every file it traverses, which raises UnicodeDecodeError on binary files
# like ``.git/index``. When the caller hands us a directory (the cloned repo),
# we read the text files ourselves (skipping .git/binary/non-text) and hand
# cognee the concatenated text blob instead of the raw path.
_COGNEE_TEXT_EXTENSIONS = {
    # code
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".cs", ".scala", ".clj", ".ex",
    ".exs", ".erl", ".hs", ".ml", ".fs", ".lua", ".pl", ".r", ".dart", ".vue",
    ".svelte", ".m", ".mm",
    # docs / config (text)
    ".md", ".markdown", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".properties", ".xml", ".html", ".htm", ".css", ".scss",
    ".sass", ".less", ".csv", ".tsv", ".env.example", ".gitignore", ".dockerignore",
    ".editorconfig", ".sql", ".graphql", ".gql", ".proto", ".thrift", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".makefile", ".mk",
}
# Basenames without an extension that are worth indexing.
_COGNEE_TEXT_BASENAMES = {
    "readme", "readme.md", "readme.rst", "readme.txt", "readme",
    "license", "license.md", "contributing", "contributing.md",
    "changelog", "changelog.md", "makefile", "dockerfile", "rakefile",
    "gemfile", "procfile", "vagrantfile", "jenkinsfile", "brewfile",
}
# Directory names to always skip when walking a repo for cognee indexing.
# ``.git`` is the primary culprit (its ``index``/``objects`` are binary and
# raise UnicodeDecodeError in cognee's text loader).
_COGNEE_SKIP_DIRS = {
    ".git", ".svn", ".hg", ".bzr", "node_modules", "bower_components",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".venv", "venv", "env", "virtualenv", "dist", "build", "out", "target",
    "bin", "obj", ".idea", ".vscode", ".vs", ".next", ".nuxt", ".cache",
    ".adalflow", "logs", "log", "tmp", "temp", "coverage", ".coverage",
}
# Per-file read cap so a giant minified file doesn't dominate the blob.
_COGNEE_PER_FILE_MAX_CHARS = 16_000
# Cap the whole blob so cognee ingestion stays bounded.
_COGNEE_BLOB_MAX_CHARS = 200_000


def _is_likely_text_file(file_path: str) -> bool:
    """Quick extension/basename allow-list check for text files."""
    import os as _os
    name = _os.path.basename(file_path)
    low = name.lower()
    if low in _COGNEE_TEXT_BASENAMES:
        return True
    ext = _os.path.splitext(low)[1]
    return ext in _COGNEE_TEXT_EXTENSIONS


def _looks_like_file_path(payload: str) -> bool:
    """True if ``payload`` is an existing file cognee should load from disk.

    ``add_and_index_document`` accepts BOTH genuine file/dir paths (which
    cognee's file loader reads from disk) AND raw text blobs (the repo text we
    read ourselves, or markdown from a caller). The two need different
    treatment when wrapping in ``DataItem``: a file path must stay a string so
    cognee opens the file, while a text blob must be wrapped in ``DataItem``
    (see cat 4 in ``add_and_index_document``) or the pipeline crashes with
    ``'str' object has no attribute '__dict__'``.

    We treat a payload as a file path only when it is a short-ish string that
    resolves to an EXISTING file on disk. Multi-KB text blobs (even if they
    happen to contain a newline-free path-like substring) never satisfy this.
    """
    import os as _os
    if not payload or len(payload) > 4096 or "\n" in payload:
        return False
    try:
        return _os.path.isfile(payload)
    except (OSError, ValueError):
        return False


def _read_repo_text_for_cognee(repo_dir: str) -> str:
    """Walk ``repo_dir`` and return a concatenated text blob of text files.

    Skips ``.git`` and other non-source dirs (``_COGNEE_SKIP_DIRS``) and any
    file whose extension isn't in the text allow-list. Each file is read as
    UTF-8 (errors replaced) and capped per-file + per-blob. Returns "" if the
    path isn't a directory or no text files were found.
    """
    import os as _os
    if not repo_dir or not _os.path.isdir(repo_dir):
        return ""
    parts = []
    total = 0
    for root, dirs, files in _os.walk(repo_dir):
        # Mutate dirs in place to prune skipped directories (os.walk contract).
        dirs[:] = [d for d in dirs if d not in _COGNEE_SKIP_DIRS]
        for fname in files:
            fpath = _os.path.join(root, fname)
            if not _is_likely_text_file(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(_COGNEE_PER_FILE_MAX_CHARS + 1)
                if len(text) > _COGNEE_PER_FILE_MAX_CHARS:
                    text = text[:_COGNEE_PER_FILE_MAX_CHARS] + "\n... (file truncated)\n"
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("cognee repo read skipped %s: %s", fpath, e)
                continue
            if not text:
                continue
            rel = _os.path.relpath(fpath, repo_dir)
            block = f"### File: {rel}\n``\n{text}\n```\n"
            if total + len(block) > _COGNEE_BLOB_MAX_CHARS:
                remaining = _COGNEE_BLOB_MAX_CHARS - total
                if remaining > 200:
                    parts.append(block[:remaining] + "\n... (repo blob truncated)\n")
                parts = [p for p in parts if p]  # keep what we have
                logger.info(
                    "cognee repo blob capped at %d chars while reading %s.",
                    _COGNEE_BLOB_MAX_CHARS, repo_dir,
                )
                return "\n".join(parts)
            parts.append(block)
            total += len(block)
    return "\n".join(parts)


def _resolve_safe_chunk_size() -> int:
    """Compute a cognify chunk size that fits the configured model's context window.

    Used by both ``remember`` (which cognifies internally) and the legacy
    ``cognify_dataset`` op. cognee auto-calculates when ``chunk_size`` is None,
    but we pass an explicit value so a misreported context window cannot push
    past the model's real limit on a local inference server.
    """
    try:
        from api.utils import get_model_context_window
        ctx_win = get_model_context_window(task="cognee")
    except Exception:
        ctx_win = 8192
    return max(300, min(1200, (ctx_win - 3000) // 2))


async def _remember_with_timeout(payload, dataset_name: str, chunk_size: int) -> None:
    """Run ``cognee.remember()`` capped by the cognify timeout.

    ``remember()`` runs add + cognify (graph extraction) in one call. cognify
    has no top-level timeout and can legitimately run for hours on a local
    model (many chunks, each a structured-output LLM call); a hung pipeline
    stage (e.g. ``extract_chunks_from_documents``) could otherwise stall the
    background indexer forever. We wrap it in ``asyncio.wait_for`` with the
    resolved cognify timeout (default 7200s).

    On timeout we do NOT raise: cognee commits graph nodes + vectors
    incrementally per chunk, so the chunks processed before the timeout are
    already persisted. Treating a timeout as success avoids re-running the
    whole dataset (which would duplicate already-indexed chunks and re-hit the
    rate limit). A genuine error (non-timeout exception) propagates to the
    caller, which logs it and returns False.
    """
    cognify_timeout = _resolve_cognify_timeout()
    try:
        logger.info(
            "Cognee remember (dataset %r, chunk_size: %d, timeout: %.0fs)...",
            dataset_name, chunk_size, cognify_timeout,
        )
        await asyncio.wait_for(
            cognee.remember(
                payload,
                dataset_name=dataset_name,
                self_improvement=False,
                chunk_size=chunk_size,
            ),
            timeout=cognify_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Cognee remember timed out after %.0fs for dataset %r; partial graph "
            "may be indexed (cognee persists incrementally per chunk).",
            cognify_timeout, dataset_name,
        )
        # Soft success: do not re-raise. See docstring.
    except asyncio.CancelledError as e:
        # The outer timeout (cognee_cognify) cancelling this task shows up as a
        # CancelledError bubbling out of ``extract_chunks_from_documents`` after
        # the per-chunk graph-extraction path already logged
        # "graph extraction skipped chunk due to TimeoutError". Cognee persists
        # graph nodes + vectors incrementally per chunk, so the chunks processed
        # before the cancellation are already committed. Treat as soft success
        # (same as TimeoutError) instead of leaking the CancelledError, which
        # previously surfaced as a confusing bare exception in the logs.
        logger.warning(
            "Cognee remember cancelled for dataset %r (%s); partial graph may be "
            "indexed (cognee persists incrementally per chunk).",
            dataset_name, e,
        )
        # Do not re-raise: see TimeoutError branch above.


async def add_document(content_or_path: str, dataset_name: str) -> bool:
    """Add a document to a cognee dataset and build its knowledge graph (V1).

    Uses the V1 ``cognee.remember()`` entry point, which runs add + cognify in
    a single call. ``self_improvement=False`` skips the follow-up ``improve()``
    pass to match the prior add+cognify-only behavior and avoid extra LLM cost
    on local models. Returns True if the payload was ingested successfully.
    Safe and non-fatal.
    """
    if not _COGNEE_AVAILABLE or not content_or_path:
        return False
    apply_cognee_runtime_config()

    payload = content_or_path
    import os as _os
    if content_or_path and _os.path.isdir(content_or_path):
        blob = _read_repo_text_for_cognee(content_or_path)
        if not blob:
            logger.warning("cognee index: no text files found in %r; skipping dataset %r.", content_or_path, dataset_name)
            return False
        payload = blob
        logger.info("cognee index: read %d chars of text from repo %r for dataset %r.", len(blob), content_or_path, dataset_name)

    ingest_payload = payload
    if isinstance(payload, str) and not _looks_like_file_path(payload):
        try:
            from cognee.tasks.ingestion.data_item import DataItem
            ingest_payload = DataItem(data=payload)
        except Exception:
            pass

    chunk_size = _resolve_safe_chunk_size()
    try:
        await _remember_with_timeout(ingest_payload, dataset_name, chunk_size)
        return True
    except Exception as e:
        msg = str(e)
        if "duplicate key value" in msg or "data_pkey" in msg or "UniqueViolationError" in type(e).__name__:
            cleared = await _empty_cognee_dataset(dataset_name)
            if cleared:
                try:
                    await _remember_with_timeout(ingest_payload, dataset_name, chunk_size)
                    return True
                except Exception:
                    pass
        logger.warning("cognee add_document failed for dataset %r: %s", dataset_name, e)
        return False


async def cognify_dataset(dataset_name: str) -> bool:
    """Run cognee.cognify() ONCE over an entire dataset to build knowledge graph.

    This is the V1 legacy cognify op (still supported). Retained for callers
    that need to re-cognify an already-ingested dataset without re-adding.
    The V1 ``remember()`` path cognifies internally, so ``add_and_index_document``
    no longer calls this separately.
    """
    if not _COGNEE_AVAILABLE:
        return False
    apply_cognee_runtime_config()

    safe_chunk_size = _resolve_safe_chunk_size()
    cognify_timeout = _resolve_cognify_timeout()
    try:
        logger.info(
            "Cognifying Cognee dataset %r (chunk_size: %d, timeout: %.0fs)...",
            dataset_name, safe_chunk_size, cognify_timeout,
        )
        await asyncio.wait_for(
            cognee.cognify(datasets=[dataset_name], chunk_size=safe_chunk_size),
            timeout=cognify_timeout,
        )
        logger.info("Cognee: Ingested and cognified dataset %r successfully.", dataset_name)
        return True
    except asyncio.TimeoutError:
        logger.warning(
            "Cognify timed out after %.0fs for dataset %r; partial graph may be indexed.",
            cognify_timeout, dataset_name,
        )
        # A timeout does NOT mean total failure: cognee commits graph nodes +
        # vectors incrementally per chunk, so the chunks processed before the
        # timeout are already persisted. Treat as a soft success so the caller
        # does not retry the whole dataset from scratch (which would duplicate
        # the already-indexed chunks and re-hit the rate limit).
        return True
    except asyncio.CancelledError as e:
        # Same rationale as the TimeoutError branch: the outer timeout cancelling
        # ``extract_chunks_from_documents`` surfaces as CancelledError after the
        # per-chunk graph-extraction path logs "skipped chunk due to TimeoutError".
        # Cognee persists per chunk, so the partial graph is already committed.
        # Treat as soft success instead of propagating a bare CancelledError.
        logger.warning(
            "Cognify cancelled for dataset %r (%s); partial graph may be indexed.",
            dataset_name, e,
        )
        return True
    except Exception as e:
        logger.error("Error cognifying dataset %r: %s", dataset_name, e, exc_info=True)
        return False


async def add_documents_and_cognify_once(items: List[str], dataset_name: str) -> None:
    """Add multiple documents to a dataset and build the graph in one V1 call.

    Uses ``cognee.remember(list, ...)`` -- the idiomatic V1 batch-ingest path
    that adds all items and runs cognify once over the dataset.
    ``self_improvement=False`` skips the follow-up improve pass. Capped by the
    cognify timeout with soft-success on timeout (cognee persists per chunk).
    """
    if not _COGNEE_AVAILABLE or not items:
        return
    # Filter to non-empty items; remember accepts a list of strings natively.
    payloads: List[str] = [item.strip() for item in items if item and item.strip()]
    if not payloads:
        return
    apply_cognee_runtime_config()
    chunk_size = _resolve_safe_chunk_size()
    try:
        await _remember_with_timeout(payloads, dataset_name, chunk_size)
    except Exception as e:
        logger.warning("cognee add_documents_and_cognify_once failed for dataset %r: %s", dataset_name, e)


async def add_and_index_document(content_or_path: str, dataset_name: str) -> None:
    """Convenience function: add a single document and build its graph (V1).

    ``add_document`` now uses ``cognee.remember()`` which cognifies internally,
    so no separate cognify step is needed (calling ``cognify_dataset`` here
    would double-build the graph).
    """
    await add_document(content_or_path, dataset_name)

async def reindex_product_knowledge_graph(product_id: Optional[str] = None) -> Dict[str, Any]:
    """Force re-indexing of product artifacts and knowledge nodes into Cognee."""
    if not _COGNEE_AVAILABLE:
        return {"success": False, "message": "Cognee package is not available.", "reindexed_count": 0}

    apply_cognee_runtime_config()

    try:
        from api.db import SessionLocal
        from api.models import ProductORM
        from sqlalchemy.orm import selectinload

        with SessionLocal() as db:
            if product_id:
                products = db.query(ProductORM).options(
                    selectinload(ProductORM.artifacts),
                    selectinload(ProductORM.knowledge_nodes),
                ).filter(ProductORM.id == product_id).all()
            else:
                products = db.query(ProductORM).options(
                    selectinload(ProductORM.artifacts),
                    selectinload(ProductORM.knowledge_nodes),
                ).all()

        if not products:
            return {"success": True, "message": "No products found to reindex.", "reindexed_count": 0}

        reindexed_count = 0
        for p in products:
            dataset_name = f"prod_{p.id}"
            items = []
            for a in p.artifacts:
                docs = getattr(a, "generated_docs", None) or getattr(a, "content", None) or ""
                if docs and docs.strip():
                    items.append(docs.strip())
            for n in p.knowledge_nodes:
                md = getattr(n, "content", None) or getattr(n, "content_md", None) or ""
                if md and md.strip():
                    items.append(md.strip())

            if items:
                logger.info("Force re-indexing %d items for product %s (%s)...", len(items), p.name, dataset_name)
                await _empty_cognee_dataset(dataset_name)
                await add_documents_and_cognify_once(items, dataset_name)
                reindexed_count += 1

        return {
            "success": True,
            "message": f"Successfully reindexed {reindexed_count} product(s) into Cognee knowledge graph.",
            "reindexed_count": reindexed_count,
        }
    except Exception as e:
        logger.error("Error during force cognee reindex: %s", e, exc_info=True)
        return {"success": False, "message": f"Reindex error: {e}", "reindexed_count": 0}
