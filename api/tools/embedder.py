import adalflow as adal

from api.config import configs, get_embedder_type


def get_embedder(is_local_ollama: bool = False, embedder_type: str = None, base_url: str = None, api_key: str = None) -> adal.Embedder:
    """Get embedder based on configuration or parameters.
    
    Args:
        is_local_ollama: Legacy parameter for Ollama embedder (kept for compatibility)
        embedder_type: Direct specification of embedder type ('ollama' or 'openai_local')
        base_url: Custom base URL for the embedder provider
        api_key: Custom API key for the embedder provider
    
    Returns:
        adal.Embedder: Configured embedder instance
    """
    # Determine which embedder config to use (local providers only)
    try:
        from api.settings_store import get_model_for_task
        emb_cfg = get_model_for_task("embedder") or {}
        if emb_cfg.get("model"):
            if "embedder_openai_local" in configs:
                configs["embedder_openai_local"]["model_kwargs"]["model"] = emb_cfg["model"]
            if "embedder_ollama" in configs:
                configs["embedder_ollama"]["model_kwargs"]["model"] = emb_cfg["model"]
        if not base_url and emb_cfg.get("base_url"):
            base_url = emb_cfg["base_url"]
        if not api_key and emb_cfg.get("api_key"):
            api_key = emb_cfg["api_key"]
    except Exception:
        pass

    if embedder_type:
        if embedder_type == 'openai_local':
            embedder_config = configs.get("embedder_openai_local", configs.get("embedder_ollama"))
        else:  # default to ollama
            embedder_config = configs.get("embedder_ollama")
    elif is_local_ollama:
        embedder_config = configs.get("embedder_ollama")
    else:
        # Auto-detect based on current configuration
        current_type = get_embedder_type()
        if current_type == 'openai_local':
            embedder_config = configs.get("embedder_openai_local", configs.get("embedder_ollama"))
        else:
            # Default to ollama for fully local operation
            embedder_config = configs.get("embedder_ollama")

    if not embedder_config:
        raise ValueError("No embedder configuration found. Please check your embedder.json config.")

    # --- Initialize Embedder ---
    model_client_class = embedder_config["model_client"]
    
    # Initialize model client with custom parameters if provided
    client_kwargs = {}
    if "initialize_kwargs" in embedder_config:
        client_kwargs.update(embedder_config["initialize_kwargs"])
    
    # Override with provided base_url and api_key
    # Determine actual type being used for parameter mapping
    if embedder_type:
        actual_type = 'openai_local' if embedder_type == 'openai_local' else 'ollama'
    elif is_local_ollama:
        actual_type = 'ollama'
    else:
        actual_type = get_embedder_type()
    
    if actual_type == 'ollama' and base_url:
        client_kwargs["host"] = base_url
    elif actual_type in ('openai_local', 'openai', 'openai_compatible') or 'openai' in str(actual_type):
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
