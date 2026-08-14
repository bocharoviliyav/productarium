from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from adalflow.core.types import ModelType


@pytest.fixture
def _patch_openai_init(monkeypatch):
    """Patch OpenAI/AsyncOpenAI constructors so no real HTTP client is built."""
    import api.clients.openai_client as mod

    mock_sync = MagicMock()
    mock_async = MagicMock()
    monkeypatch.setattr(mod, "OpenAI", mock_sync)
    monkeypatch.setattr(mod, "AsyncOpenAI", mock_async)
    return mock_sync, mock_async


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_default_base_url(self, _patch_openai_init, monkeypatch):
        monkeypatch.delenv("LOCAL_OPENAI_BASE_URL", raising=False)
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed")
        assert c.base_url == "http://localhost:8080/v1"

    def test_custom_base_url(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://my-host:1234/v1")
        assert c.base_url == "http://my-host:1234/v1"

    def test_base_url_from_env(self, _patch_openai_init, monkeypatch):
        monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://from-env:9999/v1")
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient()
        assert c.base_url == "http://from-env:9999/v1"

    def test_real_api_key_flag_true(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="sk-real-key-12345", base_url="http://localhost:8080/v1")
        assert c._real_api_key is True

    def test_real_api_key_flag_false_for_placeholder(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        assert c._real_api_key is False

    def test_custom_env_names(self, _patch_openai_init, monkeypatch):
        monkeypatch.setenv("MY_BASE_URL", "http://custom:1111/v1")
        monkeypatch.setenv("MY_API_KEY", "not-needed")
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(env_base_url_name="MY_BASE_URL", env_api_key_name="MY_API_KEY")
        assert c.base_url == "http://custom:1111/v1"

    def test_remote_endpoint_with_real_key(self, _patch_openai_init, monkeypatch):
        # A real key on a remote endpoint should construct successfully.
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="sk-real-key", base_url="https://api.openai.com/v1")
        assert c._real_api_key is True


# --------------------------------------------------------------------------- #
# _is_local_endpoint / _is_no_auth_placeholder / _resolve_api_key
# --------------------------------------------------------------------------- #
class TestLocalEndpointDetection:
    def test_localhost_is_local(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        assert c._is_local_endpoint() is True

    def test_remote_is_not_local(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="https://api.openai.com/v1")
        assert c._is_local_endpoint() is False

    def test_docker_internal_is_local(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://host.docker.internal:8080/v1")
        assert c._is_local_endpoint() is True


class TestPlaceholderDetection:
    def test_none_is_placeholder(self):
        from api.clients.openai_client import OpenAIClient

        assert OpenAIClient._is_no_auth_placeholder(None) is True

    def test_empty_is_placeholder(self):
        from api.clients.openai_client import OpenAIClient

        assert OpenAIClient._is_no_auth_placeholder("") is True

    def test_not_needed_is_placeholder(self):
        from api.clients.openai_client import OpenAIClient

        assert OpenAIClient._is_no_auth_placeholder("not-needed") is True

    def test_not_needed_underscore_is_placeholder(self):
        from api.clients.openai_client import OpenAIClient

        assert OpenAIClient._is_no_auth_placeholder("not_needed") is True

    def test_real_key_is_not_placeholder(self):
        from api.clients.openai_client import OpenAIClient

        assert OpenAIClient._is_no_auth_placeholder("sk-real-key") is False


class TestResolveApiKey:
    def test_none_returns_none(self):
        from api.clients.openai_client import OpenAIClient

        assert OpenAIClient._resolve_api_key(None) is None

    def test_placeholder_returns_none(self):
        from api.clients.openai_client import OpenAIClient

        assert OpenAIClient._resolve_api_key("not-needed") is None

    def test_real_key_returned(self):
        from api.clients.openai_client import OpenAIClient

        assert OpenAIClient._resolve_api_key("sk-real") == "sk-real"


# --------------------------------------------------------------------------- #
# convert_inputs_to_api_kwargs
# --------------------------------------------------------------------------- #
class TestConvertInputsEmbedder:
    def test_string_input_converted_to_list(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        kwargs = c.convert_inputs_to_api_kwargs(
            "hello", model_kwargs={"model": "emb"}, model_type=ModelType.EMBEDDER
        )
        assert kwargs["input"] == ["hello"]
        assert kwargs["model"] == "emb"

    def test_list_input_kept(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        kwargs = c.convert_inputs_to_api_kwargs(
            ["a", "b"], model_kwargs={"model": "emb"}, model_type=ModelType.EMBEDDER
        )
        assert kwargs["input"] == ["a", "b"]

    def test_non_sequence_raises(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        with pytest.raises(TypeError, match="must be a sequence"):
            c.convert_inputs_to_api_kwargs(
                123, model_kwargs={}, model_type=ModelType.EMBEDDER
            )


class TestConvertInputsLLM:
    def test_text_input_creates_messages(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        kwargs = c.convert_inputs_to_api_kwargs(
            "Hello", model_kwargs={"model": "gpt"}, model_type=ModelType.LLM
        )
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_messages_input_type_with_tags(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(
            api_key="not-needed",
            base_url="http://localhost:8080/v1",
            input_type="messages",
        )
        prompt = "<START_OF_SYSTEM_PROMPT>You are helpful<END_OF_SYSTEM_PROMPT><START_OF_USER_PROMPT>Hi<END_OF_USER_PROMPT>"
        kwargs = c.convert_inputs_to_api_kwargs(
            prompt, model_kwargs={"model": "gpt"}, model_type=ModelType.LLM
        )
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][0]["content"] == "You are helpful"
        assert kwargs["messages"][1]["role"] == "user"
        assert kwargs["messages"][1]["content"] == "Hi"

    def test_unsupported_model_type_raises(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        with pytest.raises(ValueError, match="not supported"):
            c.convert_inputs_to_api_kwargs(
                "x", model_kwargs={}, model_type=ModelType.UNDEFINED
            )


class TestConvertInputsImageGeneration:
    def test_image_generation_requires_model(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        with pytest.raises(ValueError, match="model must be specified"):
            c.convert_inputs_to_api_kwargs(
                "a cat", model_kwargs={}, model_type=ModelType.IMAGE_GENERATION
            )

    def test_image_generation_defaults(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        kwargs = c.convert_inputs_to_api_kwargs(
            "a cat", model_kwargs={"model": "dall-e-3"}, model_type=ModelType.IMAGE_GENERATION
        )
        assert kwargs["prompt"] == "a cat"
        assert kwargs["model"] == "dall-e-3"
        assert kwargs["size"] == "1024x1024"
        assert kwargs["quality"] == "standard"
        assert kwargs["n"] == 1
        assert kwargs["response_format"] == "url"


# --------------------------------------------------------------------------- #
# parse_chat_completion / track_completion_usage
# --------------------------------------------------------------------------- #
class TestParseChatCompletion:
    def test_parses_content(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Hello world"
        completion.usage = None
        result = c.parse_chat_completion(completion)
        assert result.raw_response == "Hello world"
        assert result.error is None

    def test_parser_exception_returns_error(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        c.chat_completion_parser = lambda x: (_ for _ in ()).throw(ValueError("bad"))
        completion = MagicMock()
        result = c.parse_chat_completion(completion)
        assert result.data is None
        assert "bad" in result.error

    def test_track_usage_none(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        completion = MagicMock()
        completion.usage = None
        usage = c.track_completion_usage(completion)
        assert usage.completion_tokens is None
        assert usage.prompt_tokens is None

    def test_track_usage_with_values(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        completion = MagicMock()
        completion.usage = MagicMock(
            completion_tokens=10, prompt_tokens=20, total_tokens=30
        )
        usage = c.track_completion_usage(completion)
        assert usage.completion_tokens == 10
        assert usage.prompt_tokens == 20
        assert usage.total_tokens == 30


# --------------------------------------------------------------------------- #
# parse_embedding_response
# --------------------------------------------------------------------------- #
class TestParseEmbeddingResponse:
    def test_success(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        response = MagicMock()
        # parse_embedding_response is from adalflow; just ensure no exception
        result = c.parse_embedding_response(response)
        # It returns EmbedderOutput; data may be empty from mock but no error
        assert result is not None


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
class TestHelperFunctions:
    def test_get_first_message_content(self):
        from api.clients.openai_client import get_first_message_content

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "hello"
        assert get_first_message_content(completion) == "hello"

    def test_estimate_token_count(self):
        from api.clients.openai_client import estimate_token_count

        assert estimate_token_count("hello world foo") == 3
        assert estimate_token_count("") == 0

    def test_parse_stream_response(self):
        from api.clients.openai_client import parse_stream_response

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "chunk text"
        assert parse_stream_response(chunk) == "chunk text"

    def test_get_all_messages_content(self):
        from api.clients.openai_client import get_all_messages_content

        completion = MagicMock()
        c1 = MagicMock()
        c1.message.content = "msg1"
        c2 = MagicMock()
        c2.message.content = "msg2"
        completion.choices = [c1, c2]
        assert get_all_messages_content(completion) == ["msg1", "msg2"]

    def test_handle_streaming_response(self):
        from api.clients.openai_client import handle_streaming_response

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "a"
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "b"
        gen = iter([chunk1, chunk2])
        result = list(handle_streaming_response(gen))
        assert result == ["a", "b"]


# --------------------------------------------------------------------------- #
# _encode_image / _prepare_image_content
# --------------------------------------------------------------------------- #
class TestImageHelpers:
    def test_encode_image_file_not_found(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        with pytest.raises(ValueError, match="Image file not found"):
            c._encode_image("/nonexistent/path/img.png")

    def test_prepare_image_content_url(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        result = c._prepare_image_content("https://example.com/img.png", "high")
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/img.png"
        assert result["image_url"]["detail"] == "high"

    def test_prepare_image_content_dict_passthrough(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        custom = {"type": "custom", "data": "x"}
        assert c._prepare_image_content(custom) == custom


# --------------------------------------------------------------------------- #
# call (sync)
# --------------------------------------------------------------------------- #
class TestCall:
    def test_call_embedder(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        c.sync_client.embeddings.create = MagicMock(return_value="emb_result")
        result = c.call({"input": ["test"], "model": "emb"}, model_type=ModelType.EMBEDDER)
        assert result == "emb_result"

    def test_call_llm_streaming(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        # When stream=True, call() returns the raw stream iterator directly.
        chunk = MagicMock()
        chunk.id = "chatcmpl-1"
        chunk.model = "gpt"
        chunk.created = 123
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hello "
        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-1"
        chunk2.model = "gpt"
        chunk2.created = 123
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "world"
        stream_iter = iter([chunk, chunk2])
        c.sync_client.chat.completions.create = MagicMock(return_value=stream_iter)
        result = c.call({"messages": [{"role": "user", "content": "hi"}], "stream": True, "model": "gpt"}, model_type=ModelType.LLM)
        # streaming call returns the raw stream (caller iterates it)
        assert result is stream_iter

    def test_call_llm_non_streaming_accumulated(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        chunk = MagicMock()
        chunk.id = "chatcmpl-1"
        chunk.model = "gpt"
        chunk.created = 123
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Accumulated"
        c.sync_client.chat.completions.create = MagicMock(return_value=iter([chunk]))
        result = c.call({"messages": [{"role": "user", "content": "hi"}], "model": "gpt"}, model_type=ModelType.LLM)
        assert result.choices[0].message.content == "Accumulated"

    def test_call_unsupported_type_raises(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        with pytest.raises(ValueError, match="not supported"):
            c.call({}, model_type=ModelType.UNDEFINED)


# --------------------------------------------------------------------------- #
# acall (async)
# --------------------------------------------------------------------------- #
class TestAcall:
    @pytest.mark.asyncio
    async def test_acall_embedder(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        c.async_client = MagicMock()
        c.async_client.embeddings.create = AsyncMock(return_value="emb_result")
        result = await c.acall({"input": ["test"], "model": "emb"}, model_type=ModelType.EMBEDDER)
        assert result == "emb_result"

    @pytest.mark.asyncio
    async def test_acall_llm(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        c.async_client = MagicMock()
        c.async_client.chat.completions.create = AsyncMock(return_value="chat_result")
        result = await c.acall(
            {"messages": [{"role": "user", "content": "hi"}], "model": "gpt"},
            model_type=ModelType.LLM,
        )
        assert result == "chat_result"

    @pytest.mark.asyncio
    async def test_acall_unsupported_raises(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        with pytest.raises(ValueError, match="not supported"):
            await c.acall({}, model_type=ModelType.UNDEFINED)


# --------------------------------------------------------------------------- #
# parse_image_generation_response
# --------------------------------------------------------------------------- #
class TestParseImageGeneration:
    def test_single_image_unwrapped(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        img = MagicMock()
        img.url = "http://example.com/img.png"
        img.b64_json = None
        result = c.parse_image_generation_response([img])
        assert result.data == "http://example.com/img.png"

    def test_multiple_images(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        img1 = MagicMock()
        img1.url = "http://example.com/1.png"
        img1.b64_json = None
        img2 = MagicMock()
        img2.url = None
        img2.b64_json = "base64data"
        result = c.parse_image_generation_response([img1, img2])
        assert result.data == ["http://example.com/1.png", "base64data"]

    def test_error_returns_none(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        result = c.parse_image_generation_response("not-a-list")  # type: ignore
        assert result.data is None
        assert result.error is not None


# --------------------------------------------------------------------------- #
# get_probabilities
# --------------------------------------------------------------------------- #
class TestGetProbabilities:
    def test_get_probabilities(self, _patch_openai_init):
        from api.clients.openai_client import get_probabilities

        t1 = MagicMock(token="hello", logprob=-0.5)
        t2 = MagicMock(token="world", logprob=-0.3)
        choice = MagicMock()
        choice.logprobs.content = [t1, t2]
        completion = MagicMock()
        completion.choices = [choice]

        result = get_probabilities(completion)
        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0].token == "hello"
        assert result[0][0].logprob == -0.5


# --------------------------------------------------------------------------- #
# _drop_auth_header
# --------------------------------------------------------------------------- #
class TestDropAuthHeader:
    def test_drops_auth_headers(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        request = MagicMock()
        request.headers = {"authorization": "Bearer x", "Authorization": "Bearer y", "other": "keep"}
        OpenAIClient._drop_auth_header(request)
        assert "authorization" not in request.headers
        assert "Authorization" not in request.headers
        assert "other" in request.headers


# --------------------------------------------------------------------------- #
# track_completion_usage + parse_chat_completion exceptions
# --------------------------------------------------------------------------- #
class TestUsageExceptions:
    def test_track_usage_attr_error(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        class _BadUsage:
            @property
            def completion_tokens(self):
                raise RuntimeError("attr failed")

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        completion = MagicMock()
        completion.usage = _BadUsage()
        usage = c.track_completion_usage(completion)
        assert usage.completion_tokens is None

    def test_parse_chat_completion_usage_error(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        c.track_completion_usage = MagicMock(side_effect=RuntimeError("unexpected"))
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "hello"
        result = c.parse_chat_completion(completion)
        assert result.error is not None
        assert "unexpected" in result.error


# --------------------------------------------------------------------------- #
# parse_embedding_response exception
# --------------------------------------------------------------------------- #
class TestParseEmbeddingException:
    def test_parse_embedding_exception(self, _patch_openai_init, monkeypatch):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        monkeypatch.setattr(
            "api.clients.openai_client.parse_embedding_response",
            lambda r: (_ for _ in ()).throw(ValueError("parse failed")),
        )
        result = c.parse_embedding_response(MagicMock())
        assert result.data == []
        assert "parse failed" in result.error


# --------------------------------------------------------------------------- #
# Messages input_type: no-match, with images
# --------------------------------------------------------------------------- #
class TestMessagesInputTypeImages:
    def test_no_match_prints_message(self, _patch_openai_init, capsys):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1", input_type="messages")
        kwargs = c.convert_inputs_to_api_kwargs(
            "plain text without tags", model_kwargs={"model": "gpt"}, model_type=ModelType.LLM
        )
        assert kwargs["messages"] == [{"role": "user", "content": "plain text without tags"}]
        assert "No match found" in capsys.readouterr().out

    def test_messages_with_images_url(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1", input_type="messages")
        prompt = "<START_OF_SYSTEM_PROMPT>Sys<END_OF_SYSTEM_PROMPT><START_OF_USER_PROMPT>Describe<END_OF_USER_PROMPT>"
        kwargs = c.convert_inputs_to_api_kwargs(
            prompt, model_kwargs={"model": "gpt-4o", "images": "https://example.com/img.png"}, model_type=ModelType.LLM
        )
        assert kwargs["messages"][0]["role"] == "system"
        user_msg = kwargs["messages"][1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0] == {"type": "text", "text": "Describe"}
        assert user_msg["content"][1]["type"] == "image_url"

    def test_messages_with_images_list(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1", input_type="messages")
        prompt = "<START_OF_SYSTEM_PROMPT>Sys<END_OF_SYSTEM_PROMPT><START_OF_USER_PROMPT>Describe<END_OF_USER_PROMPT>"
        kwargs = c.convert_inputs_to_api_kwargs(
            prompt,
            model_kwargs={"model": "gpt-4o", "images": ["https://example.com/1.png", "https://example.com/2.png"]},
            model_type=ModelType.LLM,
        )
        user_msg = kwargs["messages"][1]
        assert len(user_msg["content"]) == 3

    def test_text_input_with_images_url(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        kwargs = c.convert_inputs_to_api_kwargs(
            "describe this", model_kwargs={"model": "gpt-4o", "images": "https://example.com/img.png"}, model_type=ModelType.LLM
        )
        user_msg = kwargs["messages"][0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0] == {"type": "text", "text": "describe this"}
        assert user_msg["content"][1]["type"] == "image_url"

    def test_text_input_with_images_list(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        kwargs = c.convert_inputs_to_api_kwargs(
            "describe", model_kwargs={"model": "gpt-4o", "images": ["https://example.com/1.png", "https://example.com/2.png"]}, model_type=ModelType.LLM
        )
        user_msg = kwargs["messages"][0]
        assert len(user_msg["content"]) == 3


# --------------------------------------------------------------------------- #
# Image generation call/acall
# --------------------------------------------------------------------------- #
class TestCallImageGeneration:
    def test_call_generate(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(url="http://ex.com/img.png")]
        c.sync_client.images.generate = MagicMock(return_value=mock_resp)
        result = c.call({"prompt": "a cat", "model": "dall-e-3"}, model_type=ModelType.IMAGE_GENERATION)
        assert result == mock_resp.data

    def test_call_edit(self, _patch_openai_init):
        # Source dispatch: image + mask -> images.edit
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock()]
        c.sync_client.images.edit = MagicMock(return_value=mock_resp)
        result = c.call({"prompt": "edit", "model": "dall-e-2", "image": "b64", "mask": "b64m"}, model_type=ModelType.IMAGE_GENERATION)
        assert result == mock_resp.data

    def test_call_variation(self, _patch_openai_init):
        # Source dispatch: image WITHOUT mask -> images.create_variation
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock()]
        c.sync_client.images.create_variation = MagicMock(return_value=mock_resp)
        result = c.call({"prompt": "var", "model": "dall-e-2", "image": "b64"}, model_type=ModelType.IMAGE_GENERATION)
        assert result == mock_resp.data


class TestAcallImageGeneration:
    @pytest.mark.asyncio
    async def test_acall_generate(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        c.async_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock()]
        c.async_client.images.generate = AsyncMock(return_value=mock_resp)
        result = await c.acall({"prompt": "a cat", "model": "dall-e-3"}, model_type=ModelType.IMAGE_GENERATION)
        assert result == mock_resp.data

    @pytest.mark.asyncio
    async def test_acall_edit(self, _patch_openai_init):
        # Source dispatch: image + mask -> images.edit (async)
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        c.async_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock()]
        c.async_client.images.edit = AsyncMock(return_value=mock_resp)
        result = await c.acall({"prompt": "edit", "model": "dall-e-2", "image": "b64", "mask": "b64m"}, model_type=ModelType.IMAGE_GENERATION)
        assert result == mock_resp.data

    @pytest.mark.asyncio
    async def test_acall_variation(self, _patch_openai_init):
        # Source dispatch: image WITHOUT mask -> images.create_variation (async)
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        c.async_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock()]
        c.async_client.images.create_variation = AsyncMock(return_value=mock_resp)
        result = await c.acall({"prompt": "var", "model": "dall-e-2", "image": "b64"}, model_type=ModelType.IMAGE_GENERATION)
        assert result == mock_resp.data


# --------------------------------------------------------------------------- #
# to_dict / from_dict
# --------------------------------------------------------------------------- #
class TestSerialization:
    def test_to_dict_excludes_clients(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        d = c.to_dict()
        assert "sync_client" not in d
        assert "async_client" not in d

    def test_from_dict_recreates_clients(self, _patch_openai_init):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        d = c.to_dict()
        c2 = OpenAIClient.from_dict(d)
        assert c2.sync_client is not None


# --------------------------------------------------------------------------- #
# _encode_image / _prepare_image_content with real files
# --------------------------------------------------------------------------- #
class TestImageEncoding:
    def test_encode_image_success(self, _patch_openai_init, tmp_path):
        import base64
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG data")
        assert c._encode_image(str(img)) == base64.b64encode(b"\x89PNG data").decode()

    def test_encode_image_permission_error(self, _patch_openai_init, monkeypatch):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("denied")))
        with pytest.raises(ValueError, match="Permission denied"):
            c._encode_image("/some/path.png")

    def test_encode_image_other_error(self, _patch_openai_init, monkeypatch):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("io error")))
        with pytest.raises(ValueError, match="Error encoding image"):
            c._encode_image("/some/path.png")

    def test_prepare_image_content_local_file(self, _patch_openai_init, tmp_path):
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        img = tmp_path / "local.png"
        img.write_bytes(b"\x89PNG data")
        result = c._prepare_image_content(str(img), "high")
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert result["image_url"]["detail"] == "high"


# --------------------------------------------------------------------------- #
# Image gen convert_inputs with file paths
# --------------------------------------------------------------------------- #
class TestImageGenFilePaths:
    def test_image_gen_with_image_file(self, _patch_openai_init, tmp_path):
        import base64
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        img = tmp_path / "src.png"
        img.write_bytes(b"\x89PNG src")
        kwargs = c.convert_inputs_to_api_kwargs(
            "edit this", model_kwargs={"model": "dall-e-2", "image": str(img)}, model_type=ModelType.IMAGE_GENERATION
        )
        assert kwargs["image"] == base64.b64encode(b"\x89PNG src").decode()

    def test_image_gen_with_mask_file(self, _patch_openai_init, tmp_path):
        import base64
        from api.clients.openai_client import OpenAIClient

        c = OpenAIClient(api_key="not-needed", base_url="http://localhost:8080/v1")
        mask = tmp_path / "mask.png"
        mask.write_bytes(b"\x89PNG mask")
        kwargs = c.convert_inputs_to_api_kwargs(
            "edit this", model_kwargs={"model": "dall-e-2", "mask": str(mask)}, model_type=ModelType.IMAGE_GENERATION
        )
        assert kwargs["mask"] == base64.b64encode(b"\x89PNG mask").decode()
