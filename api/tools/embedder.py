import adalflow as adal

from api.config import configs


def get_embedder(base_url: str = None, api_key: str = None) -> adal.Embedder:
    """Get embedder based on configuration or parameters.

    Every supported local server (LM Studio, llama.cpp, vLLM, ...)
    exposes an OpenAI-compatible /v1/embeddings endpoint, so a single
    OpenAIClient-based embedder covers all cases.

    Args:

        base_url: Custom base URL for the embedder provider.
        api_key: Custom API key for the embedder provider.

    Returns:
        adal.Embedder: Configured embedder instance
    """
    # Thread admin-configured embedder model/base_url/api_key into the single
    # OpenAI-compatible embedder config.
    try:
        from api.config.settings import get_model_for_task
        emb_cfg = get_model_for_task("embedder") or {}
        if emb_cfg.get("model") and "embedder_openai_local" in configs:
            configs["embedder_openai_local"]["model_kwargs"]["model"] = emb_cfg["model"]
        if not base_url and emb_cfg.get("base_url"):
            base_url = emb_cfg["base_url"]
        if not api_key and emb_cfg.get("api_key"):
            api_key = emb_cfg["api_key"]
    except Exception:
        pass

    embedder_config = configs.get("embedder_openai_local")
    if not embedder_config:
        raise ValueError("No embedder configuration found. Please check your embedder.json config.")

    # --- Initialize Embedder ---
    model_client_class = embedder_config["model_client"]

    # Initialize model client with custom parameters if provided
    client_kwargs = {}
    if "initialize_kwargs" in embedder_config:
        client_kwargs.update(embedder_config["initialize_kwargs"])

    # Every supported server uses the OpenAI-compatible base_url=/api_key= shape.
    # SSL verify is wired via ssl_config.
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key

    model_client = model_client_class(**client_kwargs)

    # Create embedder with basic parameters
    embedder_kwargs = {"model_client": model_client, "model_kwargs": embedder_config["model_kwargs"]}

    embedder = adal.Embedder(**embedder_kwargs)

    # Set batch_size as an attribute if available (not a constructor parameter)
    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]
    return embedder
