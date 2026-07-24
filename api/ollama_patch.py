from typing import Sequence, List
from copy import deepcopy
from tqdm import tqdm
import logging
import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.component import DataComponent
import requests
import os

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class OllamaModelNotFoundError(Exception):
    """Custom exception for when Ollama model is not found"""
    pass

def check_ollama_model_exists(model_name: str, ollama_host: str = None) -> tuple:
    """
    Check if an Ollama model exists before attempting to use it.
    
    Args:
        model_name: Name of the model to check
        ollama_host: Ollama host URL, defaults to localhost:11434
        
    Returns:
        tuple: (exists: bool, error_message: str or None, available_models: list)
    """
    if ollama_host is None:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    logger.info(f"Checking Ollama model '{model_name}' at {ollama_host}")
    
    try:
        # Remove /api prefix if present and add it back
        if ollama_host.endswith('/api'):
            ollama_host = ollama_host[:-4]
        
        api_url = f"{ollama_host}/api/tags"
        logger.debug(f"Calling Ollama API: {api_url}")
        
        from api.ssl_config import requests_verify
        response = requests.get(api_url, timeout=10, verify=requests_verify())
        logger.debug(f"Ollama API response status: {response.status_code}")
        
        if response.status_code == 200:
            models_data = response.json()
            # Get full model names and base names
            all_models = [model.get('name', '') for model in models_data.get('models', [])]
            available_models = [name.split(':')[0] for name in all_models]
            model_base_name = model_name.split(':')[0]  # Remove tag if present
            
            logger.info(f"Available Ollama models: {all_models}")
            logger.info(f"Looking for base name: '{model_base_name}' in {available_models}")
            
            is_available = model_base_name in available_models
            if is_available:
                logger.info(f"Ollama model '{model_name}' is available")
                return (True, None, all_models)
            else:
                error_msg = f"Model '{model_name}' not found. Available models: {all_models}"
                logger.warning(error_msg)
                return (False, error_msg, all_models)
        else:
            error_msg = f"Ollama API returned status {response.status_code}"
            logger.warning(error_msg)
            return (False, error_msg, [])
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Cannot connect to Ollama at {ollama_host}. Is Ollama running? Error: {e}"
        logger.error(error_msg)
        return (False, error_msg, [])
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout connecting to Ollama at {ollama_host}: {e}"
        logger.error(error_msg)
        return (False, error_msg, [])
    except Exception as e:
        error_msg = f"Error checking Ollama model availability: {e}"
        logger.error(error_msg)
        return (False, error_msg, [])

class OllamaDocumentProcessor(DataComponent):
    """
    Process documents for Ollama embeddings by processing one document at a time.
    Adalflow Ollama Client does not support batch embedding, so we need to process each document individually.
    """
    def __init__(self, embedder: adal.Embedder) -> None:
        super().__init__()
        self.embedder = embedder

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        output = deepcopy(documents)
        logger.info(f"Processing {len(output)} documents individually for Ollama embeddings")

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
                        logger.info(f"Expected embedding size set to: {expected_embedding_size}")
                    elif len(embedding) != expected_embedding_size:
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                        logger.warning(f"Document '{file_path}' has inconsistent embedding size {len(embedding)} != {expected_embedding_size}, skipping")
                        continue

                    # Assign the embedding to the document
                    output[i].vector = embedding
                    successful_docs.append(output[i])
                else:
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"Failed to get embedding for document '{file_path}', skipping")
            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.error(f"Error processing document '{file_path}': {e}, skipping")

        logger.info(f"Successfully processed {len(successful_docs)}/{len(output)} documents with consistent embeddings")
        return successful_docs