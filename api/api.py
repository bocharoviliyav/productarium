import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import BackgroundTasks
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Import configuration functions
from api.config import get_available_models

# Database (SQLAlchemy) imports for Product/Artifact persistence
from sqlalchemy.orm import Session, selectinload
from api.models import ProductORM, ArtifactORM, LEGACY_ARTIFACT_TYPE_MAP
from api.db import get_db, init_db, SessionLocal
# Shared Pydantic models (contract J) — Product/Artifact live in api.schemas so
# Wave 2 routers can import them without a circular dependency on api.api.
from api.schemas import Product, Artifact
# Auth foundation: bootstrap admin (called in startup) + dependency callables.
# get_current_user is imported here as the injection point; existing endpoints
# are NOT gated yet (gating is a later step).
from api.auth.bootstrap import bootstrap_admin
from api.auth.deps import get_current_user  # noqa: F401  (injection point)
active_generations: Dict[str, Dict[str, Any]] = {}
app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper function to get adalflow root path
def get_adalflow_default_root_path():
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Pydantic Models ---
class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]

class ProcessedProjectEntry(BaseModel):
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str # Renamed from type to repo_type for clarity with existing models
    submittedAt: int # Timestamp
    language: str # Extracted from filename

class RepoInfo(BaseModel):
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """
    Model for the wiki sections.
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None

class WikiCacheData(BaseModel):
    """
    Model for the data to be stored in the wiki cache.
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  #compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class WikiCacheRequest(BaseModel):
    """
    Model for the request body when saving wiki cache.
    """
    repo: RepoInfo
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str

class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

# --- Model Configuration Models ---
class Model(BaseModel):
    """
    Model for LLM model configuration
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")

class Provider(BaseModel):
    """
    Model for LLM provider configuration
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether this provider supports custom models")

class ModelConfig(BaseModel):
    """
    Model for the entire model configuration
    """
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")


class ProviderSettings(BaseModel):
    """
    Model for provider connection settings
    """
    provider: str = Field(..., description="Provider type: 'ollama' or 'openai_local'")
    base_url: str = Field(..., description="API base URL")
    api_key: Optional[str] = Field(None, description="API key (optional)")
    embedding_model: Optional[str] = Field(None, description="Model to use for embeddings")
    custom_headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")


class ProviderTestResult(BaseModel):
    """
    Model for provider connection test result
    """
    success: bool = Field(..., description="Whether connection was successful")
    message: str = Field(..., description="Result message")
    models: Optional[List[str]] = Field(None, description="List of available models if successful")


class ProviderSettingsRequest(BaseModel):
    """
    Model for saving provider settings
    """
    settings: Dict[str, ProviderSettings] = Field(..., description="Provider settings keyed by provider type")


class AuthorizationConfig(BaseModel):
    code: str = Field(..., description="Authorization code")

from api.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE

@app.get("/lang/config")
async def get_lang_config():
    return configs["lang_config"]

@app.get("/auth/status")
async def get_auth_status():
    """
    Check if authentication is required for the wiki.
    """
    return {"auth_required": WIKI_AUTH_MODE}

@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """
    Check authorization code.
    """
    return {"success": WIKI_AUTH_CODE == request.code}

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    Get available model providers and their models.

    This endpoint returns the configuration of available model providers and their
    respective models, combining static config with dynamically fetched models from
    Ollama and OpenAI-compatible APIs.

    Returns:
        ModelConfig: A configuration object containing providers and their models
    """
    try:
        logger.info("Fetching model configurations")

        # First, get dynamically available models from Ollama and OpenAI-local
        available_models = get_available_models()

        # Create providers from the config file
        providers = []
        default_provider = configs.get("default_provider", "ollama")

        # Add provider configuration based on config.py
        for provider_id, provider_config in configs["providers"].items():
            models = []

            # Get dynamically available models for this provider
            dynamic_models = available_models.get(provider_id, [])

            # Add models from dynamic fetch first (these are most up-to-date)
            for dyn_model in dynamic_models:
                model_id = dyn_model.get("name") or dyn_model.get("id", "")
                if model_id:
                    models.append(Model(id=model_id, name=model_id))

            # Add models from config (as fallback/additional options)
            for model_id in provider_config["models"].keys():
                # Check if this model is not already in the list
                if not any(m.id == model_id for m in models):
                    models.append(Model(id=model_id, name=model_id))

            # Add provider with its models
            providers.append(
                Provider(
                    id=provider_id,
                    name=f"{provider_id.capitalize()}",
                    supportsCustomModel=provider_config.get("supportsCustomModel", False),
                    models=models
                )
            )

        # If no models found from any provider, add fallback
        if not any(p.models for p in providers):
            logger.warning("No models found from dynamic fetch, using config fallback")
            for provider_id, provider_config in configs["providers"].items():
                models = []
                for model_id in provider_config["models"].keys():
                    models.append(Model(id=model_id, name=model_id))
                providers.append(
                    Provider(
                        id=provider_id,
                        name=f"{provider_id.capitalize()}",
                        supportsCustomModel=provider_config.get("supportsCustomModel", False),
                        models=models
                    )
                )

        # Create and return the full configuration
        config = ModelConfig(
            providers=providers,
            defaultProvider=default_provider
        )
        return config

    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        # Return some default configuration in case of error
        return ModelConfig(
            providers=[
                Provider(
                    id="openai_local",
                    name="OpenAI-compatible (LM Studio)",
                    supportsCustomModel=True,
                    models=[
                        Model(id="qwen/qwen3.6-27b", name="qwen/qwen3.6-27b"),
                        Model(id="google/gemma-4-e4b", name="google/gemma-4-e4b")
                    ]
                )
            ],
            defaultProvider="openai_local"
        )

@app.get("/models/available")
async def get_available_models_endpoint():
    """
    Dynamically fetch available models from Ollama and OpenAI-compatible APIs.

    This endpoint queries the actual running instances to get their model lists,
    rather than using a static configuration.

    Returns:
        JSON with available models from each provider
    """
    try:
        logger.info("Fetching available models from all providers")
        models = get_available_models()

        # If no models found, return empty lists with status
        if not models:
            return {
                "status": "no_models",
                "providers": {},
                "message": "No models found. Make sure Ollama or local OpenAI API is running."
            }

        return {
            "status": "success",
            "providers": models,
            "message": f"Found models from {len(models)} provider(s)"
        }

    except Exception as e:
        logger.error(f"Error fetching available models: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "providers": {}
        }


@app.post("/models/test-connection")
async def test_provider_connection(settings: ProviderSettings):
    """
    Test connection to a model provider (Ollama or OpenAI-compatible).

    Request body:
    {
        "provider": "ollama" | "openai_local",
        "base_url": "http://localhost:11434",
        "api_key": "optional-key",
        "custom_headers": {"Header": "value"}
    }

    Returns:
    {
        "success": true/false,
        "message": "Connection result message",
        "models": ["model1", "model2"] (if successful)
    }
    """
    import requests
    from api.ssl_config import requests_verify

    try:
        logger.info(f"Testing connection to {settings.provider} at {settings.base_url}")

        # Build headers
        headers = {}
        api_key = settings.api_key
        if api_key == "__USE_STORED_KEY__":
            # Try to get from PROVIDER_SETTINGS first
            stored = PROVIDER_SETTINGS.get(settings.provider, {})
            api_key = stored.get("api_key")

            # If not in PROVIDER_SETTINGS, try environment variables
            if not api_key:
                if settings.provider == "ollama":
                    api_key = os.environ.get("OLLAMA_HOST_API_KEY") or os.environ.get("OLLAMA_API_KEY")
                elif settings.provider == "openai_local":
                    api_key = os.environ.get("LOCAL_OPENAI_API_KEY")

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if settings.custom_headers:
            headers.update(settings.custom_headers)

        # Test connection based on provider type
        if settings.provider == "ollama":
            # Test Ollama API
            response = requests.get(
                f"{settings.base_url}/api/tags",
                headers=headers,
                timeout=10,
                verify=requests_verify(),
            )

            if response.status_code == 200:
                models_data = response.json()
                model_names = [m.get("name", "") for m in models_data.get("models", [])]
                logger.info(f"Successfully connected to Ollama, found {len(model_names)} models")
                return ProviderTestResult(
                    success=True,
                    message=f"Successfully connected to Ollama. Found {len(model_names)} model(s).",
                    models=model_names
                )
            else:
                error_msg = f"Ollama returned status {response.status_code}"
                logger.warning(error_msg)
                return ProviderTestResult(
                    success=False,
                    message=error_msg,
                    models=None
                )

        elif settings.provider == "openai_local":
            # Test OpenAI-compatible API
            response = requests.get(
                f"{settings.base_url}/models",
                headers=headers,
                timeout=10,
                verify=requests_verify(),
            )

            if response.status_code == 200:
                models_data = response.json()
                # OpenAI-compatible format: { "data": [{ "id": "model-name" }] }
                model_names = [m.get("id", "") for m in models_data.get("data", [])]
                logger.info(f"Successfully connected to OpenAI-compatible API, found {len(model_names)} models")
                return ProviderTestResult(
                    success=True,
                    message=f"Successfully connected to OpenAI-compatible API. Found {len(model_names)} model(s).",
                    models=model_names
                )
            else:
                error_msg = f"OpenAI-compatible API returned status {response.status_code}"
                logger.warning(error_msg)
                return ProviderTestResult(
                    success=False,
                    message=error_msg,
                    models=None
                )
        else:
            return ProviderTestResult(
                success=False,
                message=f"Unknown provider: {settings.provider}",
                models=None
            )

    except requests.exceptions.ConnectionError:
        error_msg = f"Cannot connect to {settings.base_url}. Make sure the server is running."
        logger.warning(error_msg)
        return ProviderTestResult(
            success=False,
            message=error_msg,
            models=None
        )
    except requests.exceptions.Timeout:
        error_msg = f"Connection to {settings.base_url} timed out."
        logger.warning(error_msg)
        return ProviderTestResult(
            success=False,
            message=error_msg,
            models=None
        )
    except Exception as e:
        error_msg = f"Error testing connection: {str(e)}"
        logger.error(error_msg)
        return ProviderTestResult(
            success=False,
            message=error_msg,
            models=None
        )


# In-memory storage for provider settings (in production, use a database or secure storage)
PROVIDER_SETTINGS: Dict[str, Dict[str, Any]] = {}


@app.post("/models/settings")
async def save_provider_settings(request: ProviderSettingsRequest):
    """
    Save provider connection settings to persistent DB settings store with HIGHEST PRECEDENCE.
    """
    global PROVIDER_SETTINGS

    try:
        from api.config_abstraction import save_task_config, sync_runtime_settings

        for provider_id, provider_settings in request.settings.items():
            PROVIDER_SETTINGS[provider_id] = {
                "provider": provider_settings.provider,
                "base_url": provider_settings.base_url,
                "api_key": provider_settings.api_key,
                "embedding_model": provider_settings.embedding_model,
                "custom_headers": provider_settings.custom_headers
            }

            key_val = provider_settings.api_key
            if key_val == "__USE_STORED_KEY__":
                key_val = None  # keep stored key

            # Persist unconditionally to DB settings store (SettingORM)
            for task in ("docgen", "expert", "summary", "cognee"):
                save_task_config(
                    task=task,
                    provider=provider_settings.provider,
                    model=provider_settings.model or "",
                    base_url=provider_settings.base_url,
                    api_key=key_val,
                )

            if provider_settings.embedding_model:
                save_task_config(
                    task="embedder",
                    provider=provider_settings.provider,
                    model=provider_settings.embedding_model,
                    base_url=provider_settings.base_url,
                    api_key=key_val,
                )

        # Instantly apply and synchronize across process & cognee
        sync_runtime_settings()
        logger.info(f"Saved and synchronized provider settings for: {list(request.settings.keys())}")

        return {
            "success": True,
            "message": "Settings saved and synchronized successfully"
        }

    except Exception as e:
        logger.error(f"Error saving provider settings: {str(e)}")
        return {
            "success": False,
            "message": f"Error saving settings: {str(e)}"
        }


@app.get("/models/settings")
async def get_provider_settings():
    """
    Get saved provider settings (without exposing API keys).

    Returns:
    {
        "ollama": {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "hasApiKey": true/false,
            "hasCustomHeaders": true/false
        },
        "openai_local": {
            "provider": "openai_local",
            "base_url": "http://localhost:8080/v1",
            "hasApiKey": true/false,
            "hasCustomHeaders": true/false
        }
    }
    """
    result = {}

    for provider_id, settings in PROVIDER_SETTINGS.items():
        result[provider_id] = {
            "provider": settings.get("provider", provider_id),
            "base_url": settings.get("base_url", ""),
            "embedding_model": settings.get("embedding_model"),
            "hasApiKey": bool(settings.get("api_key")),
            "hasCustomHeaders": bool(settings.get("custom_headers")),
        }

    # Include defaults if no settings saved
    if not result:
        result = {
            "ollama": {
                "provider": "ollama",
                "base_url": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                "hasApiKey": bool(os.environ.get("OLLAMA_API_KEY")),
                "hasCustomHeaders": False
            },
            "openai_local": {
                "provider": "openai_local",
                "base_url": os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1"),
                "hasApiKey": bool(os.environ.get("LOCAL_OPENAI_API_KEY")),
                "hasCustomHeaders": False
            }
        }

    return result


async def run_wiki_generation_task(
    gen_key: str,
    owner: str,
    repo: str,
    repo_type: str,
    repo_url: str,
    file_tree: str,
    readme: str,
    language: str,
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_model: Optional[str] = None
):
    """Background task to generate wiki using WikiGenerator."""
    from api.wiki_generator import WikiGenerator, create_wiki_section_context
    from api.rag import RAG

    try:
        logger.info(f"Starting background wiki generation for {owner}/{repo}")
        active_generations[gen_key]["status"] = "processing"
        active_generations[gen_key]["progress"] = 5

        # Create context
        context = create_wiki_section_context(
            repo_url=repo_url,
            file_tree=file_tree,
            readme=readme,
            language=language
        )
        active_generations[gen_key]["progress"] = 10

        # Initialize RAG as LLM generator
        rag = RAG(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model
        )

        # Patch RAG to be callable as expected by WikiGenerator
        def llm_call(prompt_kwargs):
            # Use rag.generator directly with the prepared prompt.
            # adalflow 1.x Generator.call() takes ``prompt_kwargs`` (the dict
            # that fills the ``{{input_str}}`` template placeholder), NOT a
            # bare ``input_str=`` kwarg -- passing that raises TypeError.
            prompt_text = prompt_kwargs.get("input_str", "")
            result = rag.generator(prompt_kwargs={"input_str": prompt_text})
            # Extract the answer from the result
            return result.answer if hasattr(result, 'answer') else str(result)

        generator = WikiGenerator(provider=provider, model=model, language=language)
        generator.set_context(context)

        total_sections = len(generator.SECTION_ORDER)
        completed_sections = 0

        def section_callback(section_type, success, content):
            nonlocal completed_sections
            completed_sections += 1
            progress = 10 + int((completed_sections / total_sections) * 85)
            active_generations[gen_key]["progress"] = progress
            logger.info(f"Progress for {gen_key}: {progress}%")

        # Generate all sections
        generated_sections = generator.generate_all_sections(
            llm_generator=llm_call,
            section_callback=section_callback
        )

        # Convert generated sections to WikiStructureModel format for cache
        pages = []
        generated_pages_dict = {}
        for section_val, content in generated_sections.items():
            page_id = f"page_{section_val}"
            wiki_page = WikiPage(
                id=page_id,
                title=section_val.capitalize(),
                content=content,
                filePaths=[],
                importance="high",
                relatedPages=[]
            )
            pages.append(wiki_page)
            generated_pages_dict[page_id] = wiki_page

        wiki_structure = WikiStructureModel(
            id="wiki",
            title=f"Wiki for {owner}/{repo}",
            description=f"Automatically generated documentation for {repo_url}",
            pages=pages
        )

        cache_data = {
            "repo": {"owner": owner, "repo": repo, "type": repo_type, "repoUrl": repo_url},
            "language": language,
            "comprehensive": True,
            "wiki_structure": wiki_structure.dict(),
            "generated_pages": {k: v.dict() for k, v in generated_pages_dict.items()},
            "provider": provider,
            "model": model,
            "updatedAt": datetime.now().isoformat()
        }

        await save_wiki_cache(owner, repo, repo_type, language, cache_data)
        logger.info(f"Background wiki generation completed for {gen_key}")

    except Exception as e:
        logger.error(f"Error in background wiki generation for {gen_key}: {str(e)}", exc_info=True)
        if gen_key in active_generations:
            active_generations[gen_key]["status"] = "error"
            active_generations[gen_key]["error"] = str(e)
    finally:
        # Keep in active_generations for a while so frontend can see it's done
        if gen_key in active_generations and active_generations[gen_key]["status"] == "processing":
            active_generations[gen_key]["status"] = "completed"
            active_generations[gen_key]["progress"] = 100

        # Clean up old tasks after some time (e.g., 5 minutes)
        async def cleanup():
            await asyncio.sleep(300)
            if gen_key in active_generations:
                del active_generations[gen_key]

        asyncio.create_task(cleanup())

@app.post("/api/wiki/generate/background")
async def start_background_generation(request: dict, background_tasks: BackgroundTasks):
    """Start wiki generation in the background using WikiGenerator."""
    repo_url = request.get("repo_url", "")
    repo_type = request.get("type", "github")
    owner = request.get("owner", "")
    repo = request.get("repo", "")
    language = request.get("language", "ru")

    if not owner or not repo:
        # Try to parse from URL if not provided
        from api.data_pipeline import DatabaseManager
        db_manager = DatabaseManager()
        repo_name = db_manager._extract_repo_name_from_url(repo_url)
        if "/" in repo_name:
            owner, repo = repo_name.split("/", 1)
        else:
            owner = "local"
            repo = repo_name

    gen_key = f"{repo_type}_{owner}_{repo}_{language}"

    if gen_key in active_generations and active_generations[gen_key]["status"] == "processing":
        return {"status": "already_running", "gen_key": gen_key}

    active_generations[gen_key] = {
        "owner": owner,
        "repo": repo,
        "repo_type": repo_type,
        "language": language,
        "status": "queued",
        "progress": 0,
        "startTime": int(datetime.now().timestamp() * 1000)
    }

    background_tasks.add_task(
        run_wiki_generation_task,
        gen_key=gen_key,
        owner=owner,
        repo=repo,
        repo_type=repo_type,
        repo_url=repo_url,
        file_tree=request.get("file_tree", ""),
        readme=request.get("readme", ""),
        language=language,
        provider=request.get("provider", os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")),
        model=request.get("model", os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b")),
        base_url=request.get("base_url"),
        api_key=request.get("api_key"),
        embedding_model=request.get("embedding_model")
    )

    return {"status": "started", "gen_key": gen_key}

@app.get("/api/wiki/active")
async def get_active_generations():
    """Get list of active background generations."""
    return active_generations

@app.post("/wiki/generate/sequential")
async def generate_wiki_sequential(request: dict):
    """
    Generate wiki documentation with sequential sections using Qwen3.5 model.

    This endpoint generates wiki sections in order:
    1. Overview (Общая информация)
    2. Architecture (Системная архитектура - C4)
    3. Functional (Функциональное описание)
    4. Technical (Технические детали)
    5. CI/CD
    6. LLD (Low Level Design)
    7. Data Model (Модель данных)

    Request body:
    {
        "repo_url": "...",
        "type": "github|gitlab|local",
        "file_tree": "...",
        "readme": "...",
        "language": "ru|en|...",
        "model": "qwen3.5:35b-a3b",
        "provider": "ollama|openai_local"
    }

    Returns:
    {
        "status": "success|error",
        "sections": {
            "overview": {...},
            "architecture": {...},
            ...
        }
    }
    """
    from api.wiki_prompt_utils import WikiSection, get_section_prompt, get_all_sections_order, get_section_title

    try:
        logger.info(f"Starting sequential wiki generation for {request.get('repo_url', 'unknown')}")

        # Extract request parameters
        repo_url = request.get("repo_url", "")
        repo_type = request.get("type", "github")
        file_tree = request.get("file_tree", "")
        readme = request.get("readme", "")
        language = request.get("language", "ru")

        # Analyze file structure to create context
        main_directories = []
        main_files = []
        config_files = []
        cicd_files = []
        docker_files = []

        # Parse file tree to find key files
        for line in file_tree.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Find main directories and files
            if '/' not in line and line not in ['.git', 'node_modules', '__pycache__']:
                if line.endswith(('.py', '.js', '.ts', '.tsx', '.java', '.go', '.rs')):
                    main_files.append(line)
                elif line in ['src', 'lib', 'app', 'pkg', 'core', 'internal', 'api', 'services', 'models', 'handlers']:
                    main_directories.append(line)
            # Find config files
            elif any(line.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']):
                config_files.append(line)
            # Find CI/CD files
            elif any(x in line for x in ['.github', '.gitlab-ci', 'Jenkinsfile', 'azure-pipelines']):
                cicd_files.append(line)
            # Find Docker files
            elif 'dockerfile' in line.lower() or 'docker-compose' in line.lower():
                docker_files.append(line)

        # Create context for prompts
        context = {
            "main_directories": main_directories[:10],
            "main_files": main_files[:20],
            "tech_stack": {},
            "config_files": config_files[:10],
            "cicd_files": cicd_files[:5],
            "docker_files": docker_files[:5],
            "modules": main_directories[:15],
            "api_endpoints": [],
            "primary_language": "unknown",
            "file_count": len(file_tree.split('\n'))
        }

        # Generate prompts for all sections
        results = {}
        sections_order = get_all_sections_order()

        for section in sections_order:
            section_name = section.value
            logger.info(f"Preparing section: {section_name}")

            # Get prompt for this section
            prompt = get_section_prompt(
                section=section,
                repo_url=repo_url,
                file_tree=file_tree[:5000],
                readme=readme[:3000],
                context=context,
                language=language
            )

            results[section_name] = {
                "prompt": prompt,
                "status": "ready_to_generate",
                "section_title": get_section_title(section, language)
            }

        return {
            "status": "success",
            "message": f"Wiki structure prepared with {len(results)} sections",
            "sections": results,
            "generation_order": [s.value for s in sections_order]
        }

    except Exception as e:
        logger.error(f"Error in sequential wiki generation: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    Export wiki content as Markdown or JSON.

    Args:
        request: The export request containing wiki pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # Generate Markdown content
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            # Generate JSON content
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # Create response with appropriate headers for file download
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="Path to local repository")):
    """Return the file tree and README content for a local repository."""
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "No path provided. Please provide a 'path' query parameter."}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found: {path}"}
        )

    try:
        logger.info(f"Processing local repository at: {path}")
        file_tree_lines = []
        readme_content = ""

        for root, dirs, files in os.walk(path):
            # Exclude hidden dirs/files and virtual envs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'node_modules' and d != '.venv']
            for file in files:
                if file.startswith('.') or file == '__init__.py' or file == '.DS_Store':
                    continue
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                file_tree_lines.append(rel_file)
                # Find README.md (case-insensitive)
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
                        readme_content = ""

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        return {"file_tree": file_tree_str, "readme": readme_content}
    except Exception as e:
        logger.error(f"Error processing local repository: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing local repository: {str(e)}"}
        )

def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate Markdown export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"



        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate JSON export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        JSON content as string
    """
    # Create a dictionary with metadata and pages
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # Convert to JSON string with pretty formatting
    return json.dumps(export_data, indent=2)

# Import the simplified chat implementation
from api.simple_chat import chat_completions_stream
from api.websocket_wiki import handle_websocket_chat, handle_websocket_wiki_generate

# Add the chat_completions_stream endpoint to the main app
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])

# Add the WebSocket endpoints
app.add_api_websocket_route("/ws/chat", handle_websocket_chat)
app.add_api_websocket_route("/ws/wiki/generate", handle_websocket_wiki_generate)

# --- Wiki Cache Helper Functions ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str) -> str:
    """Generates the file path for a given wiki cache."""
    filename = f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)

async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str) -> Optional[WikiCacheData]:
    """Reads wiki cache data from the file system."""
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
            return None
    return None

async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """Saves wiki cache data to the file system."""
    cache_path = get_wiki_cache_path(data.repo.owner, data.repo.repo, data.repo.type, data.language)
    logger.info(f"Attempting to save wiki cache. Path: {cache_path}")
    try:
        payload = WikiCacheData(
            wiki_structure=data.wiki_structure,
            generated_pages=data.generated_pages,
            repo=data.repo,
            provider=data.provider,
            model=data.model
        )
        # Log size of data to be cached for debugging (avoid logging full content if large)
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(f"Payload prepared for caching. Size: {payload_size} bytes.")
        except Exception as ser_e:
            logger.warning(f"Could not serialize payload for size logging: {ser_e}")


        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False

# --- Wiki Cache API Endpoints ---

@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content")
):
    """
    Retrieves cached wiki data (structure and generated pages) for a repository.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        language = configs["lang_config"]["default"]

    logger.info(f"Attempting to retrieve wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language)
    if cached_data:
        return cached_data
    else:
        # Return 200 with null body if not found, as frontend expects this behavior
        # Or, raise HTTPException(status_code=404, detail="Wiki cache not found") if preferred
        logger.info(f"Wiki cache not found for {owner}/{repo} ({repo_type}), lang: {language}")
        return None

@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    Stores generated wiki data (structure and pages) to the server-side cache.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]

    if not supported_langs.__contains__(request_data.language):
        request_data.language = configs["lang_config"]["default"]

    logger.info(f"Attempting to save wiki cache for {request_data.repo.owner}/{request_data.repo.repo} ({request_data.repo.type}), lang: {request_data.language}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save wiki cache")

@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content"),
    authorization_code: Optional[str] = Query(None, description="Authorization code")
):
    """
    Deletes a specific wiki cache from the file system.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        raise HTTPException(status_code=400, detail="Language is not supported")

    if WIKI_AUTH_MODE:
        logger.info("check the authorization code")
        if not authorization_code or WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="Authorization code is invalid")

    logger.info(f"Attempting to delete wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Successfully deleted wiki cache: {cache_path}")
            return {"message": f"Wiki cache for {owner}/{repo} ({language}) deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting wiki cache {cache_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete wiki cache: {str(e)}")
    else:
        logger.warning(f"Wiki cache not found, cannot delete: {cache_path}")
        raise HTTPException(status_code=404, detail="Wiki cache not found")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "productarium-api"
    }

@app.get("/")
async def root():
    """Root endpoint to check if the API is running and list available endpoints dynamically."""
    # Collect routes dynamically from the FastAPI app
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # Skip docs and static routes
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            # Group endpoints by first path segment
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(f"{method} {route.path}")

    # Optionally, sort endpoints for readability
    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": endpoints
    }

# --- Processed Projects Endpoint --- (New Endpoint)
@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    Lists all processed projects found in the wiki cache directory.
    Projects are identified by files named like: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json
    """
    project_entries: List[ProcessedProjectEntry] = []
    # WIKI_CACHE_DIR is already defined globally in the file

    try:
        if not os.path.exists(WIKI_CACHE_DIR):
            logger.info(f"Cache directory {WIKI_CACHE_DIR} not found. Returning empty list.")
            return []

        logger.info(f"Scanning for project cache files in: {WIKI_CACHE_DIR}")
        filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR) # Use asyncio.to_thread for os.listdir

        for filename in filenames:
            if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                file_path = os.path.join(WIKI_CACHE_DIR, filename)
                try:
                    stats = await asyncio.to_thread(os.stat, file_path) # Use asyncio.to_thread for os.stat
                    parts = filename.replace("deepwiki_cache_", "").replace(".json", "").split('_')

                    # Expecting repo_type_owner_repo_language
                    # Example: deepwiki_cache_github_acme_deepwiki-open_en.json
                    # parts = [github, acme, deepwiki-open, en]
                    if len(parts) >= 4:
                        repo_type = parts[0]
                        owner = parts[1]
                        language = parts[-1] # language is the last part
                        repo = "_".join(parts[2:-1]) # repo can contain underscores

                        project_entries.append(
                            ProcessedProjectEntry(
                                id=filename,
                                owner=owner,
                                repo=repo,
                                name=f"{owner}/{repo}",
                                repo_type=repo_type,
                                submittedAt=int(stats.st_mtime * 1000), # Convert to milliseconds
                                language=language
                            )
                        )
                    else:
                        logger.warning(f"Could not parse project details from filename: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue # Skip this file on error

        # Sort by most recent first
        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"Found {len(project_entries)} processed project entries.")
        return project_entries

    except Exception as e:
        logger.error(f"Error listing processed projects from {WIKI_CACHE_DIR}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list processed projects from server cache.")


# ============================================================================
# PRODUCTS & ARTIFACTS ARCHITECTURE OVERHAUL ENDPOINTS
# ============================================================================

# Product/Artifact Pydantic models are imported from api.schemas (contract J):
#   Product  — no `type`; +summary, +owner_id
#   Artifact — type enum codebase|spec|links|documentation|guides; +kind,
#              +verified, +verified_by, +verified_at, +source

class GenerateDocRequest(BaseModel):
    # Provider/model default to the OpenAI-compatible local server (LM Studio /
    # llama.cpp / vLLM). Override via env or per-request. Ollama is still
    # supported when explicitly requested, but is no longer the default because
    # the default local server in this deployment is LM Studio (:1234).
    provider: Optional[str] = os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")
    model: Optional[str] = os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b")
    language: Optional[str] = "ru"

class RLMRunRequest(BaseModel):
    query: str
    model: Optional[str] = None

class ArtifactDocUpdate(BaseModel):
    """Partial update of an artifact's documentation (WYSIWYG saves).

    Exactly one of the doc shapes should be provided:
      - ``pages``               → replace the whole pages dict wholesale
      - ``page_id`` + ``content`` → upsert a single page's content field
      - ``generated_docs``      → replace the top-level generated_docs blob
      - ``raw_content``         → replace the artifact's raw ``content``
                                  (spec/links authored directly, no generation)
    """
    generated_docs: Optional[str] = None
    page_id: Optional[str] = None
    content: Optional[str] = None
    pages: Optional[Dict[str, Any]] = None
    raw_content: Optional[str] = None

# --- Product/Artifact DB persistence helpers (SQLAlchemy) ---

def _normalize_artifact_type(a: Artifact):
    """Map a (possibly legacy) artifact type to the new (type, kind) pair.

    Legacy openapi/asyncapi -> spec (+kind), testcase -> documentation (+kind).
    New types are passed through; an explicit kind on the request wins.
    """
    if a.type in LEGACY_ARTIFACT_TYPE_MAP:
        new_type, default_kind = LEGACY_ARTIFACT_TYPE_MAP[a.type]
        return new_type, a.kind or default_kind
    return a.type, a.kind


def _artifact_orm_from_pydantic(a: Artifact) -> ArtifactORM:
    """Build a new (transient) ArtifactORM from a Pydantic Artifact."""
    norm_type, kind = _normalize_artifact_type(a)
    return ArtifactORM(
        id=a.id,
        name=a.name,
        type=norm_type,
        kind=kind,
        repo_url=a.repo_url,
        repo_type=a.repo_type,
        token=a.token,
        content=a.content,
        allure_url=a.allure_url,
        generated_docs=a.generated_docs,
        pages=a.pages,
        verified=a.verified,
        verified_by=a.verified_by,
        verified_at=a.verified_at,
        source=a.source or "manual",
    )


def _orm_to_product(p_orm: ProductORM) -> Product:
    """Convert a ProductORM (with loaded artifacts) to the Pydantic Product.

    Field names/shapes match the previous JSON-file schema exactly so the
    frontend and Phase B consumers stay compatible. created_at/updated_at are
    intentionally NOT exposed in the public response shape.
    """
    return Product(
        id=p_orm.id,
        name=p_orm.name,
        description=p_orm.description,
        summary=p_orm.summary,
        owner_id=p_orm.owner_id,
        artifacts=[
            Artifact(
                id=a.id,
                name=a.name,
                type=a.type,
                kind=a.kind,
                repo_url=a.repo_url,
                repo_type=a.repo_type,
                token=a.token,
                content=a.content,
                allure_url=a.allure_url,
                generated_docs=a.generated_docs,
                pages=a.pages,
                verified=a.verified,
                verified_by=a.verified_by,
                verified_at=a.verified_at,
                source=a.source,
            )
            for a in p_orm.artifacts
        ],
    )


def _load_product_orm(db: Session, product_id: str) -> Optional[ProductORM]:
    """Fetch a single ProductORM with its artifacts eagerly loaded."""
    return (
        db.query(ProductORM)
        .options(selectinload(ProductORM.artifacts))
        .filter(ProductORM.id == product_id)
        .first()
    )


def _upsert_product(db: Session, product: Product) -> ProductORM:
    """Insert or update a Product and fully replace its artifacts.

    Mirrors the previous JSON ``save_product`` overwrite semantics (full
    replace of the artifacts list) so POST/PUT stay drop-in compatible.
    Existing artifact rows are deleted and flushed before the new ones are
    inserted to avoid PK collisions within a single flush.
    """
    p_orm = db.get(ProductORM, product.id)
    if p_orm is None:
        p_orm = ProductORM(
            id=product.id,
            name=product.name,
            description=product.description,
            summary=product.summary,
            owner_id=product.owner_id,
        )
        db.add(p_orm)
    else:
        p_orm.name = product.name
        p_orm.description = product.description
        p_orm.summary = product.summary
        p_orm.owner_id = product.owner_id

    db.query(ArtifactORM).filter(
        ArtifactORM.product_id == product.id
    ).delete(synchronize_session=False)
    db.flush()
    for a in product.artifacts:
        new_a = _artifact_orm_from_pydantic(a)
        new_a.product_id = product.id
        db.add(new_a)

    db.commit()
    db.refresh(p_orm)
    return p_orm


@app.on_event("startup")
async def startup_event():
    # Initialize SQLAlchemy tables for Product/Artifact persistence.
    # init_db() is non-fatal: it logs a warning and returns False if the DB
    # is unreachable, so app startup is never blocked.
    init_db()

    # Bootstrap configuration abstraction layer (highest precedence to DB settings)
    try:
        from api.config_abstraction import bootstrap_config
        bootstrap_config()
    except Exception as e:
        logger.warning("bootstrap_config failed (non-fatal): %s", e)

    from api.cognee_manager import init_cognee
    await init_cognee()
    # One-shot bootstrap admin (non-fatal): creates an admin from
    # BOOTSTRAP_ADMIN_USERNAME/PASSWORD when no admin exists yet.
    try:
        bootstrap_admin()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("bootstrap_admin failed (non-fatal): %s", e)
    # Pre-warm RLM in a background thread so the first-run fast-rlm
    # npm/pyodide download happens at boot, not inside the first generate
    # request. Non-fatal if fast-rlm is unavailable.
    try:
        from api.rlm_runner import prewarm_rlm_background
        prewarm_rlm_background()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("RLM prewarm could not start (non-fatal): %s", e)


@app.get("/api/products", response_model=List[Product])
async def list_products(db: Session = Depends(get_db)):
    products = (
        db.query(ProductORM)
        .options(selectinload(ProductORM.artifacts))
        .all()
    )
    return [_orm_to_product(p) for p in products]

@app.post("/api/products", response_model=Product)
async def create_product(product: Product, db: Session = Depends(get_db)):
    p_orm = _upsert_product(db, product)
    return _orm_to_product(p_orm)

@app.get("/api/products/{product_id}", response_model=Product)
async def get_product(product_id: str, db: Session = Depends(get_db)):
    p_orm = _load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return _orm_to_product(p_orm)

@app.put("/api/products/{product_id}", response_model=Product)
async def update_product(product_id: str, product: Product, db: Session = Depends(get_db)):
    # Preserve previous overwrite semantics: the body Product is saved as-is.
    p_orm = _upsert_product(db, product)
    return _orm_to_product(p_orm)

@app.delete("/api/products/{product_id}")
async def delete_product(product_id: str, db: Session = Depends(get_db)):
    p_orm = db.get(ProductORM, product_id)
    if p_orm is not None:
        # Child artifacts are removed via FK ON DELETE CASCADE.
        db.delete(p_orm)
        db.commit()
    # Match previous behavior: always return success, even if missing.
    return {"message": "Product deleted successfully"}


@app.post("/api/products/{product_id}/artifacts", response_model=Product)
async def add_artifact(product_id: str, artifact: Artifact, db: Session = Depends(get_db)):
    p_orm = _load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")

    # Remove any existing artifact with the same id (dedupe), then append.
    existing = next((a for a in p_orm.artifacts if a.id == artifact.id), None)
    if existing is not None:
        p_orm.artifacts.remove(existing)  # cascade delete-orphan
        db.flush()  # flush DELETE before INSERT to avoid PK collision
    p_orm.artifacts.append(_artifact_orm_from_pydantic(artifact))
    db.commit()
    db.refresh(p_orm)
    return _orm_to_product(p_orm)

@app.delete("/api/products/{product_id}/artifacts/{artifact_id}", response_model=Product)
async def delete_artifact(product_id: str, artifact_id: str, db: Session = Depends(get_db)):
    p_orm = _load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")

    existing = next((a for a in p_orm.artifacts if a.id == artifact_id), None)
    if existing is not None:
        p_orm.artifacts.remove(existing)  # cascade delete-orphan
        db.commit()
    db.refresh(p_orm)
    return _orm_to_product(p_orm)


@app.put("/api/products/{product_id}/artifacts/{artifact_id}", response_model=Product)
async def update_artifact_docs(
    product_id: str,
    artifact_id: str,
    body: ArtifactDocUpdate,
    db: Session = Depends(get_db),
):
    """Edit an artifact's generated documentation (WYSIWYG editor saves).

    Supports three save shapes (see ``ArtifactDocUpdate``): whole ``pages``
    replace, single ``page_id``+``content`` upsert, or ``generated_docs`` blob
    replace. The edited text is re-indexed into the product's cognee dataset in
    the background (fire-and-forget, non-fatal) so expert Ask/summary stay in
    sync with user edits. Returns the refreshed product so the frontend can
    update its local state.
    """
    p_orm = _load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    artifact = next((a for a in p_orm.artifacts if a.id == artifact_id), None)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    indexed_text: Optional[str] = None

    if body.pages is not None:
        # Wholesale replace (e.g. a future full-document editor).
        artifact.pages = body.pages
        indexed_text = json.dumps(body.pages, ensure_ascii=False)
    elif body.page_id is not None and body.content is not None:
        # Upsert a single page's content. ``pages`` is a JSON column persisted
        # as a dict keyed by page_id; tolerate None / non-dict by rebuilding.
        current = artifact.pages if isinstance(artifact.pages, dict) else {}
        page = current.get(body.page_id)
        if page is None:
            current[body.page_id] = {
                "id": body.page_id,
                "title": body.page_id,
                "content": body.content,
                "filePaths": [],
                "importance": "medium",
                "relatedPages": [],
            }
        else:
            page["content"] = body.content
            current[body.page_id] = page
        artifact.pages = current
        indexed_text = body.content
    elif body.generated_docs is not None:
        artifact.generated_docs = body.generated_docs
        indexed_text = body.generated_docs
    elif body.raw_content is not None:
        # Spec / links artifacts are authored directly into ``content`` (no
        # generation step). Replacing it keeps the structured viewer in sync;
        # the new text is re-indexed so expert Ask recall stays current.
        artifact.content = body.raw_content
        indexed_text = body.raw_content
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide one of: pages, (page_id + content), generated_docs, or raw_content",
        )

    db.commit()
    db.refresh(p_orm)

    # Re-index the edited text into the per-product cognee dataset so the
    # expert agent / Ask recall user edits. Fire-and-forget; never fatal.
    if indexed_text and indexed_text.strip():
        try:
            from api.artifact_docgen import _index_in_background
            _index_in_background(indexed_text, f"prod_{product_id}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Cognee re-index failed for artifact %s: %s", artifact_id, e)

    return _orm_to_product(p_orm)


# --- Async artifact documentation generation (202 + poll) ---------------------
# Long-running artifact doc generation (git clone, file read, RLM bootstrap)
# is offloaded to a dedicated ThreadPoolExecutor. The POST returns 202 + job_id
# immediately so the Next.js proxy never holds a long connection (which caused
# ECONNRESET). Each worker thread runs its OWN event loop (the docgen pipeline
# is async) with its OWN SQLAlchemy session, so the main FastAPI event loop is
# never blocked and request-scoped sessions are not shared across threads.
_docgen_jobs: Dict[str, Dict[str, Any]] = {}
_DOCGEN_MAX_WORKERS = int(os.environ.get("DOCGEN_MAX_WORKERS", "2"))
_docgen_executor = ThreadPoolExecutor(
    max_workers=_DOCGEN_MAX_WORKERS, thread_name_prefix="docgen"
)


def _docgen_prune_old_jobs(max_age_seconds: int = 3600) -> None:
    """Drop finished jobs older than ``max_age_seconds`` to bound memory."""
    cutoff = time.time() - max_age_seconds
    stale = [
        jid for jid, j in _docgen_jobs.items()
        if j.get("finished_at") and j["finished_at"] < cutoff
    ]
    for jid in stale:
        _docgen_jobs.pop(jid, None)


async def _run_docgen_job_async(
    job_id: str,
    product_id: str,
    artifact_id: str,
    provider: str,
    model: Optional[str],
    language: str,
) -> None:
    """Async body of a docgen job: loads the artifact in a FRESH DB session
    (the request session is closed by now), generates docs, commits, and
    records the outcome in the job registry. Runs inside the worker thread's
    own event loop."""
    job = _docgen_jobs[job_id]
    job["status"] = "running"
    job["indexing_status"] = "idle"
    job["indexing_message"] = "Генерация документации..."
    job["started_at"] = time.time()
    db = SessionLocal()
    try:
        p_orm = (
            db.query(ProductORM)
            .options(selectinload(ProductORM.artifacts))
            .filter(ProductORM.id == product_id)
            .first()
        )
        if p_orm is None:
            raise ValueError("Product not found")
        artifact = next((a for a in p_orm.artifacts if a.id == artifact_id), None)
        if artifact is None:
            raise ValueError("Artifact not found")
        from api.artifact_docgen import generate_artifact_documentation
        # generate_artifact_documentation writes generated_docs/pages onto the
        # artifact ORM in-place; persist them in the same transaction.
        docs = await generate_artifact_documentation(
            artifact,
            p_orm,
            provider=provider or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"),
            model=model,
            language=language or "ru",
        )
        db.commit()
        job["status"] = "succeeded"
        job["indexing_status"] = "indexing"
        job["indexing_message"] = "Документы сгенерированы. Обновляется граф знаний (cognee)..."
        job["finished_at"] = time.time()
        job["docs_chars"] = len(docs or "")
        logger.info("Docgen job %s succeeded for artifact %s; cognee indexing in progress.", job_id, artifact_id)
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        job["status"] = "failed"
        job["indexing_status"] = "failed"
        job["indexing_message"] = f"Ошибка генерации документации: {e}"
        job["error"] = str(e)
        job["finished_at"] = time.time()
        logger.error("Docgen job %s failed: %s", job_id, e, exc_info=True)
    finally:
        try:
            db.close()
        except Exception:
            pass


def _run_docgen_job(
    job_id: str,
    product_id: str,
    artifact_id: str,
    provider: str,
    model: Optional[str],
    language: str,
) -> None:
    """Worker-thread entry point: runs the async job in a brand-new event loop
    so the heavy sync work (git clone, file read, RLM) never touches the main
    loop. Fire-and-forget background tasks scheduled during the job (cognee
    indexing via ``asyncio.create_task``) are drained best-effort (120s) before
    the loop is closed, so indexing is not cancelled by loop teardown."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    job = _docgen_jobs.get(job_id)
    try:
        loop.run_until_complete(
            _run_docgen_job_async(job_id, product_id, artifact_id, provider, model, language)
        )

        async def _drain() -> None:
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]
            if pending:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=120
                )

        try:
            loop.run_until_complete(_drain())
            if job and job.get("status") == "succeeded":
                job["indexing_status"] = "succeeded"
                job["indexing_message"] = "Документы сгенерированы и граф знаний успешно обновлён."
                logger.info("Cognee indexing completed for job %s.", job_id)
        except asyncio.TimeoutError:
            if job and job.get("status") == "succeeded":
                job["indexing_status"] = "failed"
                job["indexing_message"] = "Документы сгенерированы. Превышено время ожидания индексации графа знаний."
            logger.warning("Docgen background drain timed out for job %s; non-fatal.", job_id)
        except Exception as e:  # pragma: no cover - defensive
            if job and job.get("status") == "succeeded":
                job["indexing_status"] = "failed"
                job["indexing_message"] = f"Документы сгенерированы. Ошибка индексации графа знаний: {e}"
            logger.warning("Docgen background drain error for job %s: %s", job_id, e)
    finally:
        try:
            loop.close()
        except Exception:
            pass


@app.post("/api/products/{product_id}/artifacts/{artifact_id}/generate")
async def generate_artifact_docs(
    product_id: str,
    artifact_id: str,
    request_data: GenerateDocRequest,
    db: Session = Depends(get_db),
):
    """Start asynchronous artifact documentation generation.

    Returns ``202`` with ``{job_id, status: "queued"}`` immediately. The heavy
    work (git clone, file read, RLM) runs in a worker thread with its own event
    loop, so the main FastAPI loop is never blocked and the Next.js proxy never
    sees a long-held connection (which previously caused ECONNRESET). Poll
    ``GET .../generate/status?job_id=...`` for the result.
    """
    p_orm = _load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    artifact = next((a for a in p_orm.artifacts if a.id == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    _docgen_prune_old_jobs()
    job_id = uuid.uuid4().hex
    _docgen_jobs[job_id] = {
        "job_id": job_id,
        "product_id": product_id,
        "artifact_id": artifact_id,
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "docs_chars": None,
    }
    _docgen_executor.submit(
        _run_docgen_job,
        job_id,
        product_id,
        artifact_id,
        request_data.provider or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"),
        request_data.model,
        request_data.language or "ru",
    )
    logger.info("Submitted docgen job %s for artifact %s", job_id, artifact_id)
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued", "artifact_id": artifact_id},
    )


@app.get("/api/products/{product_id}/artifacts/{artifact_id}/generate/status")
async def get_docgen_status(
    product_id: str,
    artifact_id: str,
    job_id: str = Query(..., description="Docgen job id returned by the generate endpoint"),
):
    """Poll the status of an asynchronous docgen job and cognee indexing."""
    job = _docgen_jobs.get(job_id)
    if (
        job is None
        or job.get("product_id") != product_id
        or job.get("artifact_id") != artifact_id
    ):
        raise HTTPException(status_code=404, detail="Docgen job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "indexing_status": job.get("indexing_status", "idle"),
        "indexing_message": job.get("indexing_message", ""),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "docs_chars": job.get("docs_chars"),
    }


@app.post("/api/rlm/run")
async def run_rlm_endpoint(request_data: RLMRunRequest):
    from api.rlm_runner import run_rlm_task
    try:
        result = await run_rlm_task(request_data.query, request_data.model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Dynamic router loader (foundation + Wave 2 routers) -------------------
# Discovers api/routers/*.py modules and includes their `router` APIRouters,
# plus the foundation auth router (api/auth/router.py). Wave 2 agents drop in
# new api/routers/<name>.py files without editing this file.
from api.routers import include_all_routers  # noqa: E402

_router_includes = include_all_routers(app)
if _router_includes:
    logger.info("Included routers via dynamic loader: %s", _router_includes)
