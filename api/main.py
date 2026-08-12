import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Bootstrap a stable SETTINGS_SECRET_KEY (persisted to ~/.adalflow when the
# env var is unset) BEFORE any module reads the settings store. This keeps
# encrypted admin settings (models.*/api_key, confluence.token) and JWT
# session signing stable across restarts instead of using a new ephemeral
# per-process key on each boot. Non-fatal; see api/settings_store.py.
from api.settings_store import bootstrap_secret_key
bootstrap_secret_key()

# Apply SSL/TLS configuration (corporate CA bundle / skip-verify) BEFORE any
# HTTP client (requests, httpx, openai SDK, cognee) is constructed, so the
# default-trust-store consumers honor SSL_CERT_FILE for an enterprise AI
# gateway. See api/ssl_config.py.
from api.ssl_config import apply_ssl_env
apply_ssl_env()

from api.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Configure watchfiles logger to show file paths
watchfiles_logger = logging.getLogger("watchfiles.main")
watchfiles_logger.setLevel(logging.DEBUG)  # Enable DEBUG to see file paths

# Add the current directory to the path so we can import the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply watchfiles monkey patch BEFORE uvicorn import
is_development = os.environ.get("NODE_ENV") != "production"
if is_development:
    import watchfiles
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(current_dir, "logs")
    
    original_watch = watchfiles.watch
    def patched_watch(*args, **kwargs):
        # Only watch the api directory but exclude logs subdirectory
        # Instead of watching the entire api directory, watch specific subdirectories
        api_subdirs = []
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and item != "logs":
                api_subdirs.append(item_path)
            elif os.path.isfile(item_path) and item.endswith(".py"):
                api_subdirs.append(item_path)
        
        return original_watch(*api_subdirs, **kwargs)
    watchfiles.watch = patched_watch

import uvicorn

# Log configuration info (no cloud API keys required)
from api.config import OLLAMA_HOST, LOCAL_OPENAI_BASE_URL
logger.info(f"DeepWiki Local Mode - Ollama host: {OLLAMA_HOST}")
logger.info(f"Local OpenAI API endpoint: {LOCAL_OPENAI_BASE_URL}")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8001))

    # Import the app here to ensure environment variables are set first
    from api.api import app

    logger.info(f"Starting Streaming API on port {port}")

    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=is_development,
        reload_excludes=["**/logs/*", "**/__pycache__/*", "**/*.pyc"] if is_development else None,
    )
