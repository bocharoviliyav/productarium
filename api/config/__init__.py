import os
import json
import logging
import re
from pathlib import Path
from typing import List, Union, Dict, Any

logger = logging.getLogger(__name__)

from api.clients.openai_client import OpenAIClient


# Local OpenAI-compatible API settings (e.g., LM Studio, llama.cpp server, vLLM,
# text-generation-webui). Default is LM Studio's OpenAI-compatible endpoint.
LOCAL_OPENAI_BASE_URL = os.environ.get('LOCAL_OPENAI_BASE_URL', 'http://localhost:1234/v1')
LOCAL_OPENAI_API_KEY = os.environ.get('LOCAL_OPENAI_API_KEY') or 'not-needed'  # Many local servers don't need a key; empty/missing falls back to 'not-needed'

# Enterprise Git settings (optional - for self-hosted GitHub/GitLab instances)
# These can be overridden via UI for each request
GITHUB_ENTERPRISE_URL = os.environ.get('GITHUB_ENTERPRISE_URL', '')  # e.g., https://github.company.com
GITLAB_SELF_HOSTED_URL = os.environ.get('GITLAB_SELF_HOSTED_URL', '')  # e.g., https://gitlab.company.com

# Wiki authentication settings
raw_auth_mode = os.environ.get('DEEPWIKI_AUTH_MODE', 'False')
WIKI_AUTH_MODE = raw_auth_mode.lower() in ['true', '1', 't']
WIKI_AUTH_CODE = os.environ.get('DEEPWIKI_AUTH_CODE', '')

# Get configuration directory from environment variable, or use default if not set
CONFIG_DIR = os.environ.get('DEEPWIKI_CONFIG_DIR', None)

def replace_env_placeholders(config: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    Recursively replace placeholders like "${ENV_VAR}" in string values
    within a nested configuration structure (dicts, lists, strings)
    with environment variable values. Logs a warning if a placeholder is not found.
    """
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replacer(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            logger.warning(
                f"Environment variable placeholder '{original_placeholder}' was not found in the environment. "
                f"The placeholder string will be used as is."
            )
            return original_placeholder
        return env_var_value

    if isinstance(config, dict):
        return {k: replace_env_placeholders(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_placeholders(item) for item in config]
    elif isinstance(config, str):
        return pattern.sub(replacer, config)
    else:
        # Handles numbers, booleans, None, etc.
        return config

# Load JSON configuration file
def load_json_config(filename):
    try:
        # If environment variable is set, use the directory specified by it
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            # Otherwise use default directory
            config_path = Path(__file__).parent / filename

        logger.info(f"Loading configuration from {config_path}")

        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} does not exist")
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config = replace_env_placeholders(config)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration file {filename}: {str(e)}")
        return {}

# Load generator model configuration
def load_generator_config():
    generator_config = load_json_config("generator.json")

    # Resolve the model client class for the openai_local provider. Every
    # supported local server (LM Studio, llama.cpp, vLLM, ...)
    # exposes the OpenAI-compatible /v1 API, so OpenAIClient covers all.
    if "providers" in generator_config:
        for provider_id, provider_config in generator_config["providers"].items():
            provider_config["model_client"] = OpenAIClient

    return generator_config

# Load embedder configuration
def load_embedder_config():
    embedder_config = load_json_config("embedder.json")

    # Resolve the embedder client class (OpenAIClient for all local servers).
    for key in ["embedder_openai_local"]:
        if key in embedder_config:
            embedder_config[key]["model_client"] = OpenAIClient

    return embedder_config

def get_embedder_config():
    """
    Get the current embedder configuration.

    Every supported local server (LM Studio, llama.cpp, vLLM, ...)
    exposes an OpenAI-compatible /v1/embeddings endpoint, so the single
    ``embedder_openai_local`` (OpenAIClient) config covers all cases.

    Returns:
        dict: The embedder configuration with model_client resolved
    """
    return configs.get("embedder_openai_local", {})

# Load repository and file filters configuration
def load_repo_config():
    return load_json_config("repo.json")

# Load language configuration
def load_lang_config():
    default_config = {
        "supported_languages": {
            "en": "English",
            "ru": "Русский (Russian)"
        },
        "default": "ru"
    }

    loaded_config = load_json_config("lang.json") # Let load_json_config handle path and loading

    if not loaded_config:
        return default_config

    if "supported_languages" not in loaded_config or "default" not in loaded_config:
        logger.warning("Language configuration file 'lang.json' is malformed. Using default language configuration.")
        return default_config

    return loaded_config

# Default excluded directories and files
DEFAULT_EXCLUDED_DIRS: List[str] = [
    # Virtual environments and package managers
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    # Version control
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    # Cache and compiled files
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
    # Build and distribution
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    # Documentation
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    # IDE specific
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    # Logs and temporary files
    "./logs/", "./log/", "./tmp/", "./temp/",
]

DEFAULT_EXCLUDED_FILES: List[str] = [
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk", ".env",
    ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv", ".gitignore",
    ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    "pyproject.toml", "tsconfig.json", "webpack.config.js", "babel.config.js",
    "rollup.config.js", "jest.config.js", "karma.conf.js", "vite.config.js",
    "next.config.js", "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
    "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib", "*.o",
    "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class", "*.pyc", "*.pyd",
    "*.pyo", "__pycache__", "*.a", "*.lib", "*.lo", "*.la", "*.slo", "*.dSYM",
    "*.egg", "*.egg-info", "*.dist-info", "*.eggs", "node_modules",
    "bower_components", "jspm_packages", "lib-cov", "coverage", "htmlcov",
    ".nyc_output", ".tox", "dist", "build", "bld", "out", "bin", "target",
    "packages/*/dist", "packages/*/build", ".output"
]

# Initialize empty configuration
configs = {}

# Load all configuration files
generator_config = load_generator_config()
embedder_config = load_embedder_config()
repo_config = load_repo_config()
lang_config = load_lang_config()

# Update configuration
if generator_config:
    configs["default_provider"] = generator_config.get("default_provider", "openai_local")
    configs["providers"] = generator_config.get("providers", {})

# Update embedder configuration
if embedder_config:
    for key in ["embedder_openai_local", "retriever", "text_splitter"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]

# Update repository configuration
if repo_config:
    for key in ["file_filters", "repository"]:
        if key in repo_config:
            configs[key] = repo_config[key]

# Update language configuration
if lang_config:
    configs["lang_config"] = lang_config


def get_model_config(model=None):
    """
    Get configuration for the model.

    Every supported local server (LM Studio, llama.cpp, vLLM, ...)
    exposes an OpenAI-compatible /v1 API, so the single ``openai_local``
    provider (OpenAIClient) is always used.

    Parameters:
        model (str): Model name, or None to use default model

    Returns:
        dict: Configuration containing model_client, model and other parameters
    """
    # Get provider configuration
    if "providers" not in configs:
        raise ValueError("Provider configuration not loaded")

    provider_config = configs["providers"].get("openai_local")
    if not provider_config:
        raise ValueError("Configuration for provider 'openai_local' not found")

    model_client = provider_config.get("model_client")
    if not model_client:
        raise ValueError("Model client not specified for provider 'openai_local'")

    # If model not provided, use default model for the provider
    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError("No default model specified for provider 'openai_local'")

    # Get model parameters (if present)
    model_params = {}
    if model in provider_config.get("models", {}):
        model_params = provider_config["models"][model]
    else:
        default_model = provider_config.get("default_model")
        model_params = provider_config["models"].get(default_model, {})

    # Every supported server uses the flat OpenAI-compatible parameter shape.
    return {
        "model_client": model_client,
        "model_kwargs": {"model": model, **model_params},
    }


def fetch_openai_local_models(base_url: str = None):
    """
    Fetch available models from a local OpenAI-compatible API.

    Every supported local server (LM Studio, llama.cpp, vLLM, ...)
    exposes the OpenAI-compatible ``/v1/models`` endpoint, so this is the
    single model-listing path.

    Args:
        base_url: Custom base URL for the API (defaults to LOCAL_OPENAI_BASE_URL)

    Returns:
        list: List of available model names
    """
    import requests
    from api.config.ssl import requests_verify

    url = base_url or LOCAL_OPENAI_BASE_URL
    if not url:
        return []

    # Ensure URL has correct format
    if not url.endswith("/v1"):
        url = url.rstrip("/") + "/v1"

    try:
        from api.config.timeout import resolve_model_list_timeout
        response = requests.get(f"{url}/models", timeout=resolve_model_list_timeout(), verify=requests_verify())
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("data", []):
                models.append({
                    "id": model.get("id", ""),
                    "object": model.get("object", ""),
                    "created": model.get("created", 0),
                })
            return models
        else:
            logger.warning(f"Failed to fetch OpenAI local models: {response.status_code}")
            return []
    except Exception as e:
        logger.warning(f"Error fetching OpenAI local models: {str(e)}")
        return []


def get_available_models():
    """
    Get available models from the configured OpenAI-compatible API.

    Returns:
        dict: Dictionary with provider IDs as keys and model lists as values
    """
    result = {}

    openai_models = fetch_openai_local_models()
    if openai_models:
        result["openai_local"] = openai_models

    return result
