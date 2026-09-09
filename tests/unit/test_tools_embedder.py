from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.tools.embedder import get_embedder


class TestGetEmbedder:
    def test_sends_raw_strings_not_token_ids(self):
        # Local GGUF servers (LM Studio / llama.cpp / Unsloth Studio)
        # tokenize server-side and validate ids against their own vocab.
        # langchain's default client-side tiktoken pass
        # (check_embedding_ctx_length=True) sends cl100k token-ID arrays;
        # ids beyond the server's vocab trigger 400 "Prompt contains
        # invalid tokens" for almost any real text. The factory must send
        # plain strings.
        emb = get_embedder()
        assert emb.check_embedding_ctx_length is False

    def test_explicit_args_accepted(self):
        # Explicit base_url/api_key args are accepted and do not flip the
        # raw-strings behaviour.
        emb = get_embedder(base_url="http://embedder.local:9999/v1", api_key="k")
        assert emb.check_embedding_ctx_length is False
