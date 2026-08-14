from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

import api.formats.markitdown as md_mod
from api.formats.markitdown import convert_to_markdown


# --------------------------------------------------------------------------- #
# api.formats.markitdown
# --------------------------------------------------------------------------- #
class TestPlaceholder:
    def test_placeholder_format(self):
        from api.formats.markitdown import _placeholder

        result = _placeholder("report.docx", "not installed")
        assert "report.docx" in result
        assert "not installed" in result
        assert result.startswith("<!--")

    def test_placeholder_none_filename(self):
        from api.formats.markitdown import _placeholder

        result = _placeholder(None, "error")
        assert "unknown" in result


class TestConvertUnavailable:
    def test_returns_placeholder_when_not_installed(self, monkeypatch):
        # Force _MARKITDOWN_AVAILABLE to False and _MARKITDOWN to None
        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", False)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", None)
        # Also patch _load_markitdown to return None
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: None)
        result = convert_to_markdown(b"some bytes", filename="test.pdf")
        assert "conversion unavailable" in result
        assert "test.pdf" in result


class TestConvertPath:
    def test_convert_file_path(self, monkeypatch, tmp_path):
        # Set up fake markitdown
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = "# Markdown from file"
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        test_file = tmp_path / "test.docx"
        test_file.write_bytes(b"fake docx content")

        result = convert_to_markdown(str(test_file))
        assert result == "# Markdown from file"
        fake_converter.convert.assert_called_once_with(str(test_file))

    def test_convert_pathlike(self, monkeypatch, tmp_path):
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = "# PathLike content"
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"fake pdf")

        result = convert_to_markdown(test_file)
        assert result == "# PathLike content"


class TestConvertBytes:
    def test_convert_bytes_with_filename(self, monkeypatch):
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = "# Bytes content"
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        result = convert_to_markdown(b"<html>test</html>", filename="page.html")
        assert result == "# Bytes content"
        # Verify convert was called with a BytesIO stream
        call_arg = fake_converter.convert.call_args[0][0]
        assert hasattr(call_arg, "read")
        assert call_arg.name == "page.html"

    def test_convert_bytes_without_filename(self, monkeypatch):
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = "# No filename"
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        result = convert_to_markdown(b"some data")
        assert result == "# No filename"


class TestConvertStream:
    def test_convert_file_like_object(self, monkeypatch):
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = "# Stream content"
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        stream = io.BytesIO(b"stream data")
        result = convert_to_markdown(stream, filename="stream.html")
        assert result == "# Stream content"

    def test_convert_stream_sets_name(self, monkeypatch):
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = "# Named stream"
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        stream = io.BytesIO(b"data")
        # Don't set .name on the stream; the function should set it from filename
        convert_to_markdown(stream, filename="from_filename.html")
        assert stream.name == "from_filename.html"


class TestConvertUnsupported:
    def test_unsupported_input_type_returns_placeholder(self, monkeypatch):
        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        fake_converter_class = MagicMock()
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        result = convert_to_markdown(12345)  # type: ignore
        assert "unsupported input type" in result


class TestConvertError:
    def test_conversion_exception_returns_placeholder(self, monkeypatch, tmp_path):
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_converter.convert.side_effect = RuntimeError("parse error")
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        test_file = tmp_path / "bad.docx"
        test_file.write_bytes(b"bad data")

        result = convert_to_markdown(str(test_file))
        assert "conversion error" in result
        assert "parse error" in result


class TestConvertTextContentFallback:
    def test_falls_back_to_markdown_attr(self, monkeypatch, tmp_path):
        """When text_content is None, falls back to .markdown or str()."""
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = None
        fake_result.markdown = "# From markdown attr"
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        test_file = tmp_path / "test.html"
        test_file.write_bytes(b"<html></html>")

        result = convert_to_markdown(str(test_file))
        assert result == "# From markdown attr"

    def test_falls_back_to_str(self, monkeypatch, tmp_path):
        """When both text_content and markdown are None, falls back to str()."""
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = None
        # Remove markdown attr so it falls through to str()
        del fake_result.markdown
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        test_file = tmp_path / "test.docx"
        test_file.write_bytes(b"data")

        result = convert_to_markdown(str(test_file))
        assert isinstance(result, str)
        assert len(result) > 0  # str(MagicMock) is non-empty

    def test_empty_text_returns_empty_string(self, monkeypatch, tmp_path):
        fake_converter_class = MagicMock()
        fake_converter = MagicMock()
        fake_result = MagicMock()
        fake_result.text_content = ""
        fake_converter.convert.return_value = fake_result
        fake_converter_class.return_value = fake_converter

        monkeypatch.setattr(md_mod, "_MARKITDOWN_AVAILABLE", True)
        monkeypatch.setattr(md_mod, "_MARKITDOWN", fake_converter_class)
        monkeypatch.setattr(md_mod, "_load_markitdown", lambda: fake_converter_class)

        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        result = convert_to_markdown(str(test_file))
        assert result == ""


# --------------------------------------------------------------------------- #
# api.tools.embedder
# --------------------------------------------------------------------------- #
class TestGetEmbedder:
    def test_returns_embedder_instance(self):
        from api.tools.embedder import get_embedder

        embedder = get_embedder(base_url="http://localhost:11434/v1", api_key="not-needed")
        assert embedder is not None
        # adal.Embedder has a model_client and model_kwargs
        assert hasattr(embedder, "model_client")
        assert hasattr(embedder, "model_kwargs")

    def test_uses_custom_base_url(self):
        from api.tools.embedder import get_embedder

        embedder = get_embedder(base_url="http://custom:9999/v1", api_key="not-needed")
        assert embedder.model_client.base_url == "http://custom:9999/v1"

    def test_uses_custom_api_key(self):
        from api.tools.embedder import get_embedder

        embedder = get_embedder(base_url="http://localhost:11434/v1", api_key="sk-test-key")
        assert embedder.model_client._api_key == "sk-test-key"

    def test_defaults_to_config_base_url(self, monkeypatch):
        from api.tools.embedder import get_embedder
        from api.config import configs

        # Ensure the embedder config exists
        assert "embedder_openai_local" in configs
        embedder = get_embedder()
        assert embedder is not None
        # Default base_url comes from LOCAL_OPENAI_BASE_URL env or "http://localhost:8080/v1"
        assert embedder.model_client.base_url is not None

    def test_raises_when_no_config(self, monkeypatch):
        import api.tools.embedder as emb_mod
        from api.tools.embedder import get_embedder

        # get_embedder() reads ``configs`` from api.tools.embedder's own
        # namespace (bound at its import time via `from api.config import configs`).
        # Other test files call importlib.reload(api.config), which rebinds
        # api.config.configs to a NEW dict — so the embedder module keeps the
        # OLD dict reference. We must therefore mutate the dict the embedder
        # module actually sees (emb_mod.configs), and use monkeypatch.delitem
        # for automatic teardown regardless of test ordering.
        monkeypatch.delitem(emb_mod.configs, "embedder_openai_local", raising=False)
        with pytest.raises(ValueError, match="No embedder configuration"):
            get_embedder()

    def test_sets_batch_size_attribute(self):
        from api.tools.embedder import get_embedder

        embedder = get_embedder(base_url="http://localhost:11434/v1", api_key="not-needed")
        # The config embedder.json has batch_size=100
        assert hasattr(embedder, "batch_size")
        assert embedder.batch_size == 100

    def test_admin_config_overrides_model(self, monkeypatch):
        from api.tools.embedder import get_embedder

        # Mock get_model_for_task to return a custom embedder model
        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "custom-emb-model", "base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        embedder = get_embedder()
        assert embedder.model_kwargs["model"] == "custom-emb-model"

    def test_admin_config_provides_base_url_and_key(self, monkeypatch):
        from api.tools.embedder import get_embedder

        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: {"model": "emb", "base_url": "http://admin-emb:5555/v1", "api_key": "emb-key-123"},
        )
        embedder = get_embedder()
        assert embedder.model_client.base_url == "http://admin-emb:5555/v1"
        assert embedder.model_client._api_key == "emb-key-123"
