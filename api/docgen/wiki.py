"""
Wiki Section Generator Module

This module handles sequential generation of wiki documentation sections
using the Qwen3.5-35b-a3b model.

Prompt BODIES are externalised in refs/prompts/*.md and loaded via the
SECTION_PROMPTS registry from api.prompts. This module keeps only the abstract
steps: the section order, the per-section variable mapping in _format_prompt,
and the dispatch from get_section_prompt_for_frontend to the loaded templates.

Section order:
1. Overview (Общая информация)
2. Architecture (Системная архитектура - C4)
3. Functional (Функциональное описание)
4. Technical (Технические детали)
5. CI/CD
6. LLD (Low Level Design)
7. Data Model (Модель данных)
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from api.prompts import (
    WIKI_OVERVIEW_PROMPT,
    WIKI_ARCHITECTURE_PROMPT,
    WIKI_FUNCTIONAL_PROMPT,
    WIKI_TECHNICAL_PROMPT,
    WIKI_CICD_PROMPT,
    WIKI_LLD_PROMPT,
    WIKI_DATAMODEL_PROMPT,
    WIKI_STRUCTURE_PROMPT,
)

# Configure logging
from api.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class WikiSectionType(Enum):
    """Types of wiki sections in sequential order"""
    OVERVIEW = "overview"
    ARCHITECTURE = "architecture"
    FUNCTIONAL = "functional"
    TECHNICAL = "technical"
    CICD = "cicd"
    LLD = "lld"
    DATAMODEL = "datamodel"


@dataclass
class WikiSectionContext:
    """Context data for generating a wiki section"""
    repo_url: str
    repo_name: str
    repo_type: str
    primary_language: str
    file_count: int
    main_directories: List[str]
    project_structure: str
    main_files: List[str]
    tech_stack: Dict[str, Any]
    config_files: List[str]
    cicd_files: List[str]
    docker_files: List[str]
    api_endpoints: List[Dict[str, str]]
    databases: List[str]
    entities: List[str]
    modules: List[str]
    previous_sections: Dict[str, str] = field(default_factory=dict)


class WikiGenerator:
    """
    Main class for generating wiki sections sequentially.
    
    Uses Qwen3.5-35b-a3b model to generate comprehensive documentation
    following the specified section order.
    """
    
    # Section order for sequential generation
    SECTION_ORDER = [
        WikiSectionType.OVERVIEW,
        WikiSectionType.ARCHITECTURE,
        WikiSectionType.FUNCTIONAL,
        WikiSectionType.TECHNICAL,
        WikiSectionType.CICD,
        WikiSectionType.LLD,
        WikiSectionType.DATAMODEL,
    ]
    
    # Section display names (Russian)
    SECTION_NAMES = {
        WikiSectionType.OVERVIEW: "Общая информация",
        WikiSectionType.ARCHITECTURE: "Системная архитектура",
        WikiSectionType.FUNCTIONAL: "Функциональное описание",
        WikiSectionType.TECHNICAL: "Технические детали",
        WikiSectionType.CICD: "CI/CD",
        WikiSectionType.LLD: "LLD (Low Level Design)",
        WikiSectionType.DATAMODEL: "Модель данных",
    }
    
    def __init__(
        self,
        provider: str = None,
        model: str = None,
        language: str = "ru"
    ):
        """
        Initialize the wiki generator.
        
        Args:
            provider: LLM provider ('openai_local' or 'ollama'); defaults to
                the DEEPWIKI_DEFAULT_PROVIDER env var ('openai_local').
            model: Model name to use; defaults to DEEPWIKI_DEFAULT_MODEL env
                var ('qwen/qwen3.6-27b').
            language: Output language code ('ru', 'en', etc.)
        """
        import os
        self.provider = provider or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")
        self.model = model or os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b")
        self.language = language
        self.generated_sections: Dict[str, str] = {}
        self.context: Optional[WikiSectionContext] = None
        
        logger.info(f"WikiGenerator initialized with model: {model}, provider: {provider}")
    
    def set_context(self, context: WikiSectionContext):
        """Set the context data for generation"""
        self.context = context
        logger.info(f"Context set for repo: {context.repo_name}")
    
    def _get_prompt_for_section(
        self, 
        section_type: WikiSectionType
    ) -> str:
        """Get the prompt template for a specific section type.

        Templates are loaded from refs/prompts/<section>.md at api.prompts
        import time (via load_prompt_file), so the returned string is the
        external template — never an inline body.
        """
        prompts_map = {
            WikiSectionType.OVERVIEW: WIKI_OVERVIEW_PROMPT,
            WikiSectionType.ARCHITECTURE: WIKI_ARCHITECTURE_PROMPT,
            WikiSectionType.FUNCTIONAL: WIKI_FUNCTIONAL_PROMPT,
            WikiSectionType.TECHNICAL: WIKI_TECHNICAL_PROMPT,
            WikiSectionType.CICD: WIKI_CICD_PROMPT,
            WikiSectionType.LLD: WIKI_LLD_PROMPT,
            WikiSectionType.DATAMODEL: WIKI_DATAMODEL_PROMPT,
        }
        return prompts_map.get(section_type, "")
    
    def _format_prompt(
        self, 
        section_type: WikiSectionType,
        context: WikiSectionContext
    ) -> str:
        """
        Format the prompt with context data for a specific section.
        
        Args:
            section_type: Type of section to generate
            context: Context data
            
        Returns:
            Formatted prompt string
        """
        prompt_template = self._get_prompt_for_section(section_type)
        
        # Format previous sections for context
        previous_content = ""
        if self.generated_sections:
            previous_content = "\n\n".join([
                f"## {self.SECTION_NAMES.get(key, key)}\n\n{content}"
                for key, content in self.generated_sections.items()
            ])
        
        # Common context variables
        common_vars = {
            "repo_url": context.repo_url,
            "repo_name": context.repo_name,
            "repo_type": context.repo_type,
            "previous_content": previous_content,
        }
        
        # Section-specific context variables (the abstract variable mapping)
        if section_type == WikiSectionType.OVERVIEW:
            prompt_vars = {
                **common_vars,
                "primary_language": context.primary_language,
                "file_count": context.file_count,
                "main_directories": ", ".join(context.main_directories[:10]),
            }
        elif section_type == WikiSectionType.ARCHITECTURE:
            prompt_vars = {
                **common_vars,
                "project_structure": context.project_structure,
                "main_files": ", ".join(context.main_files[:20]),
            }
        elif section_type == WikiSectionType.FUNCTIONAL:
            prompt_vars = {
                **common_vars,
                "app_type": "web_application",
                "main_modules": ", ".join(context.modules[:10]),
                "api_endpoints": json.dumps(context.api_endpoints, indent=2),
            }
        elif section_type == WikiSectionType.TECHNICAL:
            prompt_vars = {
                **common_vars,
                "tech_stack": json.dumps(context.tech_stack, indent=2),
                "config_files": ", ".join(context.config_files),
            }
        elif section_type == WikiSectionType.CICD:
            prompt_vars = {
                **common_vars,
                "cicd_files": ", ".join(context.cicd_files),
                "docker_files": ", ".join(context.docker_files),
                "config_files": ", ".join(context.config_files),
            }
        elif section_type == WikiSectionType.LLD:
            prompt_vars = {
                **common_vars,
                "components": json.dumps(context.modules[:20], indent=2),
                "api_endpoints": json.dumps(context.api_endpoints, indent=2),
                "modules": ", ".join(context.modules[:20]),
            }
        elif section_type == WikiSectionType.DATAMODEL:
            prompt_vars = {
                **common_vars,
                "databases": ", ".join(context.databases),
                "entities": ", ".join(context.entities),
                "db_config": "configuration in config files",
            }
        else:
            prompt_vars = common_vars
        
        # Fill in the template using safe string replacement.
        # NOTE: str.replace is used (matching websocket_wiki.py) rather than
        # str.format() because the refs/prompts/*.md templates contain literal
        # braces in Mermaid/JSON examples that must be preserved verbatim.
        # str.format() would raise on those unescaped braces; str.replace only
        # substitutes exact {var_name} placeholders and leaves the rest intact.
        formatted = prompt_template
        for var_name, var_value in prompt_vars.items():
            formatted = formatted.replace("{" + var_name + "}", str(var_value))
        # Append the unified verification guard (grounding/citation/
        # no-line-numbers/unverified-flag rules). Read fresh from api.prompts at
        # call time so a hot-reload via the admin panel takes effect without a
        # process restart. The guard is additive on top of each section's
        # existing <important> block.
        from api.prompts import VERIFICATION_GUARD as _guard
        if _guard:
            formatted = formatted + "\n\n" + _guard

        try:
            from api.utils import get_model_context_window, clamp_text_by_tokens
            ctx_win = get_model_context_window(provider=self.provider, model_name=self.model, task="docgen")
            formatted = clamp_text_by_tokens(formatted, max(1024, ctx_win - 2048))
        except Exception:
            pass

        return formatted
    
    def generate_section(
        self, 
        section_type: WikiSectionType,
        llm_generator: Any = None
    ) -> Tuple[bool, str]:
        """
        Generate a single wiki section.
        
        Args:
            section_type: Type of section to generate
            llm_generator: Optional LLM generator instance (adalflow Generator)
            
        Returns:
            Tuple of (success: bool, content: str)
        """
        if not self.context:
            logger.error("Context not set, cannot generate section")
            return False, "Error: Context not set"
        
        try:
            logger.info(f"Generating section: {section_type.value}")
            
            # Format the prompt
            prompt = self._format_prompt(section_type, self.context)
            
            # If no LLM generator provided, return the prompt for external processing
            if llm_generator is None:
                logger.warning("No LLM generator provided, returning formatted prompt")
                return True, prompt
            
            # Generate content using LLM
            response = llm_generator(prompt_kwargs={"input_str": prompt})
            
            # Store the generated content
            self.generated_sections[section_type.value] = response
            
            logger.info(f"Successfully generated section: {section_type.value}")
            return True, response
            
        except Exception as e:
            logger.error(f"Error generating section {section_type.value}: {str(e)}")
            return False, f"Error generating section: {str(e)}"
    
    def generate_all_sections(
        self, 
        llm_generator: Any = None,
        section_callback: Optional[callable] = None
    ) -> Dict[str, str]:
        """
        Generate all wiki sections sequentially.
        
        Args:
            llm_generator: Optional LLM generator instance
            section_callback: Optional callback function called after each section generation
            
        Returns:
            Dictionary of generated sections
        """
        if not self.context:
            logger.error("Context not set, cannot generate sections")
            return {}
        
        logger.info("Starting sequential wiki generation")
        
        for section_type in self.SECTION_ORDER:
            logger.info(f"Generating section {section_type.value} ({self.SECTION_NAMES[section_type]})")
            
            success, content = self.generate_section(section_type, llm_generator)
            
            if not success:
                logger.error(f"Failed to generate section: {section_type.value}")
            
            # Call callback if provided
            if section_callback:
                try:
                    section_callback(section_type, success, content)
                except Exception as e:
                    logger.error(f"Error in section callback: {str(e)}")
        
        logger.info("Completed sequential wiki generation")
        return self.generated_sections
    
    def get_section_prompt_for_frontend(
        self, 
        section_type: WikiSectionType,
        file_tree: str = "",
        readme: str = "",
        relevant_files: Optional[List[str]] = None
    ) -> str:
        """
        Generate a prompt for the frontend to use with WebSocket.
        
        This is used by the frontend to generate wiki content using the existing
        WebSocket infrastructure. The prompt body is loaded from the
        refs/prompts/<section>.md template and formatted with the context
        variables via _format_prompt.
        
        Args:
            section_type: Type of section to generate
            file_tree: Project file tree structure (kept for API compatibility;
                already available as context.project_structure)
            readme: README content (kept for API compatibility)
            relevant_files: List of relevant files for this section (kept for
                API compatibility)
            
        Returns:
            Formatted prompt string for the LLM
        """
        if not self.context:
            return "Error: Context not set"
        
        # Dispatch to the per-section builder. Each builder loads the
        # corresponding refs/prompts/<section>.md template (through
        # _format_prompt) and formats it with the context variables.
        section_prompts = {
            WikiSectionType.OVERVIEW: self._build_overview_prompt,
            WikiSectionType.ARCHITECTURE: self._build_architecture_prompt,
            WikiSectionType.FUNCTIONAL: self._build_functional_prompt,
            WikiSectionType.TECHNICAL: self._build_technical_prompt,
            WikiSectionType.CICD: self._build_cicd_prompt,
            WikiSectionType.LLD: self._build_lld_prompt,
            WikiSectionType.DATAMODEL: self._build_datamodel_prompt,
        }
        
        builder = section_prompts.get(section_type)
        if builder:
            return builder(file_tree, readme, relevant_files)
        
        return "Unknown section type"
    
    # ------------------------------------------------------------------------
    # Per-section prompt builders.
    #
    # These used to contain full inline f-string prompt bodies. They now simply
    # load the matching refs/prompts/<section>.md template (via _format_prompt,
    # which uses the SECTION_PROMPTS registry loaded in api.prompts) and format
    # it with the abstract per-section variable mapping. The file_tree / readme
    # / relevant_files parameters are retained for call-site compatibility.
    # ------------------------------------------------------------------------
    
    def _build_overview_prompt(
        self, 
        file_tree: str, 
        readme: str, 
        relevant_files: List[str]
    ) -> str:
        """Build prompt for Overview section from refs/prompts/overview.md."""
        return self._format_prompt(WikiSectionType.OVERVIEW, self.context)
    
    def _build_architecture_prompt(
        self, 
        file_tree: str, 
        readme: str, 
        relevant_files: List[str]
    ) -> str:
        """Build prompt for Architecture section from refs/prompts/architecture.md."""
        return self._format_prompt(WikiSectionType.ARCHITECTURE, self.context)
    
    def _build_functional_prompt(
        self, 
        file_tree: str, 
        readme: str, 
        relevant_files: List[str]
    ) -> str:
        """Build prompt for Functional section from refs/prompts/functional.md."""
        return self._format_prompt(WikiSectionType.FUNCTIONAL, self.context)
    
    def _build_technical_prompt(
        self, 
        file_tree: str, 
        readme: str, 
        relevant_files: List[str]
    ) -> str:
        """Build prompt for Technical section from refs/prompts/technical.md."""
        return self._format_prompt(WikiSectionType.TECHNICAL, self.context)
    
    def _build_cicd_prompt(
        self, 
        file_tree: str, 
        readme: str, 
        relevant_files: List[str]
    ) -> str:
        """Build prompt for CI/CD section from refs/prompts/cicd.md."""
        return self._format_prompt(WikiSectionType.CICD, self.context)
    
    def _build_lld_prompt(
        self, 
        file_tree: str, 
        readme: str, 
        relevant_files: List[str]
    ) -> str:
        """Build prompt for LLD section from refs/prompts/lld.md."""
        return self._format_prompt(WikiSectionType.LLD, self.context)
    
    def _build_datamodel_prompt(
        self, 
        file_tree: str, 
        readme: str, 
        relevant_files: List[str]
    ) -> str:
        """Build prompt for Data Model section from refs/prompts/datamodel.md."""
        return self._format_prompt(WikiSectionType.DATAMODEL, self.context)


def create_wiki_section_context(
    repo_url: str,
    repo_type: str,
    file_tree: str,
    readme: str,
    file_analysis: Dict[str, Any]
) -> WikiSectionContext:
    """
    Create WikiSectionContext from repository analysis data.
    
    Args:
        repo_url: Repository URL
        repo_type: Type (github, gitlab, etc.)
        file_tree: File tree structure
        readme: README content
        file_analysis: Analysis results from data_pipeline
        
    Returns:
        WikiSectionContext instance
    """
    # Extract relevant information from file analysis
    main_directories = file_analysis.get("main_directories", [])
    main_files = file_analysis.get("main_files", [])
    tech_stack = file_analysis.get("tech_stack", {})
    config_files = file_analysis.get("config_files", [])
    cicd_files = file_analysis.get("cicd_files", [])
    docker_files = file_analysis.get("docker_files", [])
    api_endpoints = file_analysis.get("api_endpoints", [])
    databases = file_analysis.get("databases", [])
    entities = file_analysis.get("entities", [])
    modules = file_analysis.get("modules", [])
    primary_language = file_analysis.get("primary_language", "unknown")
    file_count = file_analysis.get("file_count", 0)
    
    # Extract repo name from URL
    repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
    
    return WikiSectionContext(
        repo_url=repo_url,
        repo_name=repo_name,
        repo_type=repo_type,
        primary_language=primary_language,
        file_count=file_count,
        main_directories=main_directories,
        project_structure=file_tree,
        main_files=main_files,
        tech_stack=tech_stack,
        config_files=config_files,
        cicd_files=cicd_files,
        docker_files=docker_files,
        api_endpoints=api_endpoints,
        databases=databases,
        entities=entities,
        modules=modules,
    )


# Example usage
if __name__ == "__main__":
    # Test the wiki generator
    context = WikiSectionContext(
        repo_url="https://github.com/example/myapp",
        repo_name="myapp",
        repo_type="github",
        primary_language="Python",
        file_count=150,
        main_directories=["src", "tests", "docs", "config"],
        project_structure="project structure...",
        main_files=["main.py", "app.py", "config.py"],
        tech_stack={"language": "Python", "framework": "FastAPI"},
        config_files=["config.json", "settings.py"],
        cicd_files=[".github/workflows/ci.yml"],
        docker_files=["Dockerfile", "docker-compose.yml"],
        api_endpoints=[{"path": "/api/users", "method": "GET"}],
        databases=["PostgreSQL"],
        entities=["User", "Post", "Comment"],
        modules=["auth", "users", "posts"],
    )
    
    generator = WikiGenerator(provider="openai_local", model="qwen/qwen3.6-27b", language="ru")
    generator.set_context(context)
    
    print("WikiGenerator initialized successfully")
    print(f"Section order: {[s.value for s in generator.SECTION_ORDER]}")
    print(f"Section names: {generator.SECTION_NAMES}")
