"""Ollama embedding helpers.

``OllamaDocumentProcessor`` processes documents one at a time because adalflow's
Ollama client does not support batch embedding. It also validates embedding-size
consistency across documents and skips any that diverge.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Sequence

import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.component import DataComponent
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OllamaDocumentProcessor(DataComponent):
    """Process documents for Ollama embeddings one document at a time.

    Adalflow's Ollama client does not support batch embedding, so each document
    is embedded individually. Embedding-size consistency is validated; documents
    whose embedding size differs from the first successful one are skipped.
    """

    def __init__(self, embedder: adal.Embedder) -> None:
        super().__init__()
        self.embedder = embedder

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        output = deepcopy(documents)
        logger.info("Processing %d documents individually for Ollama embeddings", len(output))

        successful_docs = []
        expected_embedding_size = None

        for i, doc in enumerate(tqdm(output, desc="Processing documents for Ollama embeddings")):
            try:
                # Get embedding for a single document
                result = self.embedder(input=doc.text)
                if result.data and len(result.data) > 0:
                    embedding = result.data[0].embedding

                    # Validate embedding size consistency
                    if expected_embedding_size is None:
                        expected_embedding_size = len(embedding)
                        logger.info("Expected embedding size set to: %d", expected_embedding_size)
                    elif len(embedding) != expected_embedding_size:
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                        logger.warning("Document '%s' has inconsistent embedding size %d != %d, skipping", file_path, len(embedding), expected_embedding_size)
                        continue

                    # Assign the embedding to the document
                    output[i].vector = embedding
                    successful_docs.append(output[i])
                else:
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning("Failed to get embedding for document '%s', skipping", file_path)
            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.error("Error processing document '%s': %s, skipping", file_path, e)

        logger.info("Successfully processed %d/%d documents with consistent embeddings", len(successful_docs), len(output))
        return successful_docs
