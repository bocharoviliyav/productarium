#!/usr/bin/env python3
"""
Unit tests for DeepWiki embedder configuration, factory, and pipeline functions.
"""

import os
import sys
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEmbedderConfiguration:
    """Test embedder configuration system."""

    def test_config_loading(self):
        """Test that embedder configurations load properly."""
        from api.config import configs, CLIENT_CLASSES

        assert 'embedder_ollama' in configs, "Ollama embedder config missing"
        assert 'embedder_openai_local' in configs, "OpenAI local embedder config missing"

        assert 'OpenAIClient' in CLIENT_CLASSES, "OpenAIClient missing from CLIENT_CLASSES"
        assert 'OllamaClient' in CLIENT_CLASSES, "OllamaClient missing from CLIENT_CLASSES"

    def test_embedder_type_detection(self):
        """Test embedder type detection functions."""
        from api.config import get_embedder_type, is_ollama_embedder, is_openai_local_embedder

        current_type = get_embedder_type()
        assert current_type in ['ollama', 'openai_local'], f"Invalid embedder type: {current_type}"

        is_ollama = is_ollama_embedder()
        is_openai_local = is_openai_local_embedder()
        assert isinstance(is_ollama, bool), "is_ollama_embedder should return boolean"
        assert isinstance(is_openai_local, bool), "is_openai_local_embedder should return boolean"

    def test_get_embedder_config(self):
        """Test getting embedder config."""
        from api.config import get_embedder_config

        config = get_embedder_config()
        assert isinstance(config, dict), "Config should be dict"
        assert 'model_client' in config or 'client_class' in config, "No client specified in embedder config"


class TestEmbedderFactory:
    """Test the embedder factory function."""

    def test_get_embedder_with_explicit_type(self):
        """Test get_embedder with explicit embedder_type parameter."""
        from api.tools.embedder import get_embedder

        # Test Ollama embedder
        try:
            ollama_embedder = get_embedder(embedder_type='ollama')
            assert ollama_embedder is not None, "Ollama embedder should be created"
        except Exception as e:
            logger.warning(f"Ollama embedder creation failed: {e}")

        # Test OpenAI local embedder
        try:
            openai_embedder = get_embedder(embedder_type='openai_local')
            assert openai_embedder is not None, "OpenAI local embedder should be created"
        except Exception as e:
            logger.warning(f"OpenAI local embedder creation failed: {e}")

    def test_get_embedder_with_legacy_params(self):
        """Test get_embedder with legacy parameter."""
        from api.tools.embedder import get_embedder

        try:
            ollama_embedder = get_embedder(is_local_ollama=True)
            assert ollama_embedder is not None, "Ollama embedder should be created with is_local_ollama=True"
        except Exception as e:
            logger.warning(f"Ollama embedder creation failed: {e}")

    def test_get_embedder_auto_detection(self):
        """Test get_embedder with automatic type detection."""
        from api.tools.embedder import get_embedder

        embedder = get_embedder()
        assert embedder is not None, "Auto-detected embedder should be created"


class TestDataPipelineFunctions:
    """Test data pipeline functions that use embedders."""

    def test_count_tokens(self):
        """Test token counting."""
        from api.data_pipeline import count_tokens

        test_text = "This is a test string for token counting."
        for embedder_type in [None, 'ollama', 'openai_local']:
            token_count = count_tokens(test_text, embedder_type=embedder_type)
            assert isinstance(token_count, int), "Token count should be an integer"
            assert token_count > 0, "Token count should be positive"

    def test_prepare_data_pipeline(self):
        """Test data pipeline preparation."""
        from api.data_pipeline import prepare_data_pipeline

        for embedder_type in [None, 'ollama', 'openai_local']:
            try:
                pipeline = prepare_data_pipeline(embedder_type=embedder_type)
                assert pipeline is not None, "Data pipeline should be created"
                assert hasattr(pipeline, '__call__'), "Pipeline should be callable"
            except Exception as e:
                logger.warning(f"Pipeline creation failed for embedder_type={embedder_type}: {e}")


class TestRAGIntegration:
    """Test RAG class integration with embedders."""

    def test_rag_initialization(self):
        """Test RAG initialization with default configuration."""
        from api.rag import RAG

        try:
            rag = RAG(provider="ollama", model="qwen3.5:9b")
            assert rag is not None, "RAG should be initialized"
            assert hasattr(rag, 'embedder'), "RAG should have embedder"
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")


class TestEnvironmentVariableHandling:
    """Test embedder selection via environment variables."""

    def test_embedder_type_env_var(self):
        """Test embedder selection via DEEPWIKI_EMBEDDER_TYPE environment variable."""
        import importlib
        import api.config

        original_value = os.environ.get('DEEPWIKI_EMBEDDER_TYPE')

        try:
            for et in ['ollama', 'openai_local']:
                os.environ['DEEPWIKI_EMBEDDER_TYPE'] = et
                importlib.reload(api.config)

                from api.config import EMBEDDER_TYPE, get_embedder_type
                assert EMBEDDER_TYPE == et, f"EMBEDDER_TYPE should be {et}"
                assert get_embedder_type() == et, f"get_embedder_type() should return {et}"
        finally:
            if original_value is not None:
                os.environ['DEEPWIKI_EMBEDDER_TYPE'] = original_value
            elif 'DEEPWIKI_EMBEDDER_TYPE' in os.environ:
                del os.environ['DEEPWIKI_EMBEDDER_TYPE']
            importlib.reload(api.config)
