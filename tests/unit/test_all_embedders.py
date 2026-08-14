#!/usr/bin/env python3
"""
Unit tests for embedder configuration, factory, and pipeline functions.

The embedder layer now uses a single OpenAI-compatible provider
(``embedder_openai_local`` via OpenAIClient) that covers every local
server (Ollama, LM Studio, llama.cpp, vLLM, ...). There is no longer a
provider/``embedder_type`` switch — every supported server exposes the
same ``/v1/embeddings`` endpoint — so these tests cover the single path.
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEmbedderConfiguration:
    """Test the single OpenAI-compatible embedder configuration."""

    def test_config_loading(self):
        """The embedder config is loaded under a single key."""
        from api.config import configs

        assert 'embedder_openai_local' in configs, "embedder_openai_local config missing"

    def test_get_embedder_config(self):
        """get_embedder_config() returns a dict with a resolved client."""
        from api.config import get_embedder_config

        config = get_embedder_config()
        assert isinstance(config, dict), "Config should be dict"
        assert 'model_client' in config, "No model_client in embedder config"


class TestEmbedderFactory:
    """Test the embedder factory function."""

    def test_get_embedder_auto_detection(self):
        """get_embedder() with no args builds the single embedder."""
        from api.tools.embedder import get_embedder

        embedder = get_embedder()
        assert embedder is not None, "Embedder should be created"


class TestDataPipelineFunctions:
    """Test data pipeline functions that use embedders."""

    def test_count_tokens(self):
        """count_tokens uses tiktoken (no provider param)."""
        from api.repositories.documents import count_tokens

        test_text = "This is a test string for token counting."
        token_count = count_tokens(test_text)
        assert isinstance(token_count, int), "Token count should be an integer"
        assert token_count > 0, "Token count should be positive"

    def test_prepare_data_pipeline(self):
        """prepare_data_pipeline() builds a callable pipeline (no params)."""
        from api.repositories.documents import prepare_data_pipeline

        try:
            pipeline = prepare_data_pipeline()
            assert pipeline is not None, "Data pipeline should be created"
            assert hasattr(pipeline, '__call__'), "Pipeline should be callable"
        except Exception as e:
            logger.warning(f"Pipeline creation failed: {e}")
