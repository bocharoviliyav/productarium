import logging
import os
from typing import List, Optional, Dict, Any
from urllib.parse import unquote

from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from api.config import get_model_config, configs
from api.data_pipeline import count_tokens, get_file_content
from api.mermaid_verifier import run_repair_loop
from api.openai_client import OpenAIClient
from api.rag import RAG

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Models for the API
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    """
    Model for requesting a chat completion.
    """
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None, description="Optional path to a file in the repository to include in the prompt")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    type: Optional[str] = Field("github", description="Type of repository (e.g., 'github', 'gitlab')")

    # model parameters - local providers only
    provider: str = Field(
        os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"),
        description="Model provider ('openai_local' or 'ollama')",
    )
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    language: Optional[str] = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    base_url: Optional[str] = Field(None, description="Custom base URL for the model provider")
    api_key: Optional[str] = Field(None, description="Custom API key for the model provider")
    embedding_model: Optional[str] = Field(None, description="Custom model for embeddings")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")

async def handle_websocket_chat(websocket: WebSocket):
    """
    Handle WebSocket connection for chat completions.
    This replaces the HTTP streaming endpoint with a WebSocket connection.
    """
    await websocket.accept()

    try:
        # Receive and parse the request data
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

        # Check if request contains very large input
        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content, request.provider == "ollama")
                logger.info(f"Request size: {tokens} tokens")
                if tokens > 8000:
                    logger.warning(f"Request exceeds recommended token limit ({tokens} > 7500)")
                    input_too_large = True

        # Create a new RAG instance for this request
        try:
            request_rag = RAG(
                provider=request.provider,
                model=request.model,
                base_url=request.base_url,
                api_key=request.api_key,
                embedding_model=request.embedding_model
            )

            # Extract custom file filter parameters if provided
            excluded_dirs = None
            excluded_files = None
            included_dirs = None
            included_files = None

            if request.excluded_dirs:
                excluded_dirs = [unquote(dir_path) for dir_path in request.excluded_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom excluded directories: {excluded_dirs}")
            if request.excluded_files:
                excluded_files = [unquote(file_pattern) for file_pattern in request.excluded_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom excluded files: {excluded_files}")
            if request.included_dirs:
                included_dirs = [unquote(dir_path) for dir_path in request.included_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom included directories: {included_dirs}")
            if request.included_files:
                included_files = [unquote(file_pattern) for file_pattern in request.included_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom included files: {included_files}")

            request_rag.prepare_retriever(request.repo_url, request.type, request.token, excluded_dirs, excluded_files, included_dirs, included_files)
            logger.info(f"Retriever prepared for {request.repo_url}")
        except ValueError as e:
            if "No valid documents with embeddings found" in str(e):
                logger.error(f"No valid embeddings found: {str(e)}")
                await websocket.send_text("Error: No valid document embeddings found. This may be due to embedding size inconsistencies or API errors during document processing. Please try again or check your repository content.")
                await websocket.close()
                return
            else:
                logger.error(f"ValueError preparing retriever: {str(e)}")
                await websocket.send_text(f"Error preparing retriever: {str(e)}")
                await websocket.close()
                return
        except Exception as e:
            logger.error(f"Error preparing retriever: {str(e)}")
            # Check for specific embedding-related errors
            if "All embeddings should be of the same size" in str(e):
                await websocket.send_text("Error: Inconsistent embedding sizes detected. Some documents may have failed to embed properly. Please try again.")
            else:
                await websocket.send_text(f"Error preparing retriever: {str(e)}")
            await websocket.close()
            return

        # Validate request
        if not request.messages or len(request.messages) == 0:
            await websocket.send_text("Error: No messages provided")
            await websocket.close()
            return

        last_message = request.messages[-1]
        if last_message.role != "user":
            await websocket.send_text("Error: Last message must be from the user")
            await websocket.close()
            return

        # Process previous messages to build conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]

                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(
                        user_query=user_msg.content,
                        assistant_response=assistant_msg.content
                    )

        # Check if this is a Deep Research request
        is_deep_research = False
        research_iteration = 1

        # Process messages to detect Deep Research requests
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                # Only remove the tag from the last message
                if msg == request.messages[-1]:
                    # Remove the Deep Research tag
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        # Count research iterations if this is a Deep Research request
        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")

            # Check if this is a continuation request
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                # Find the original topic from the first user message
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break

                if original_topic:
                    # Replace the continuation message with the original topic
                    last_message.content = original_topic
                    logger.info(f"Using original topic for research: {original_topic}")

        # Get the query from the last message
        query = last_message.content

        # Only retrieve documents if input is not too large
        context_text = ""
        retrieved_documents = None

        if not input_too_large:
            try:
                # If filePath exists, modify the query for RAG to focus on the file
                rag_query = query
                if request.filePath:
                    # Use the file path to get relevant context about the file
                    rag_query = f"Contexts related to {request.filePath}"
                    logger.info(f"Modified RAG query to focus on file: {request.filePath}")

                # Try to perform RAG retrieval
                try:
                    # This will use the actual RAG implementation
                    retrieved_documents = request_rag(rag_query, language=request.language)

                    if retrieved_documents and retrieved_documents[0].documents:
                        # Format context for the prompt in a more structured way
                        documents = retrieved_documents[0].documents
                        logger.info(f"Retrieved {len(documents)} documents")

                        # Group documents by file path
                        docs_by_file = {}
                        for doc in documents:
                            file_path = doc.meta_data.get('file_path', 'unknown')
                            if file_path not in docs_by_file:
                                docs_by_file[file_path] = []
                            docs_by_file[file_path].append(doc)

                        # Format context text with file path grouping
                        context_parts = []
                        for file_path, docs in docs_by_file.items():
                            # Add file header with metadata
                            header = f"## File Path: {file_path}\n\n"
                            # Add document content
                            content = "\n\n".join([doc.text for doc in docs])

                            context_parts.append(f"{header}{content}")

                        # Join all parts with clear separation
                        context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                    else:
                        logger.warning("No documents retrieved from RAG")

                    # In addition, query Cognee if Cognee dataset is available for this repo
                    try:
                        import re
                        from api.cognee_manager import query_cognee
                        # Deriving a safe dataset name
                        safe_dataset_name = re.sub(r'[^a-zA-Z0-9_]', '_', request.repo_url)
                        cognee_context = await query_cognee(rag_query, dataset_name=safe_dataset_name)
                        if cognee_context:
                            logger.info(f"Retrieved Cognee knowledge graph context: {len(cognee_context)} chars")
                            context_text += f"\n\n## Knowledge Graph Context (Cognee):\n{cognee_context}"
                    except Exception as ce:
                        logger.warning(f"Cognee search skipped or failed: {ce}")

                except Exception as e:
                    logger.error(f"Error in RAG retrieval: {str(e)}")
                    # Continue without RAG if there's an error

            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                context_text = ""

        # Get repository information
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # Determine repository type
        repo_type = request.type

        # Get language information
        language_code = request.language or configs["lang_config"]["default"]
        supported_langs = configs["lang_config"]["supported_languages"]
        language_name = supported_langs.get(language_code, "English")

        # Create system prompt
        if is_deep_research:
            # Check if this is the first iteration
            is_first_iteration = research_iteration == 1

            # Check if this is the final iteration
            is_final_iteration = research_iteration >= 5

            if is_first_iteration:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are conducting a multi-turn Deep Research process to thoroughly investigate the specific topic in the user's query.
Your goal is to provide detailed, focused information EXCLUSIVELY about this topic.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- This is the first iteration of a multi-turn research process focused EXCLUSIVELY on the user's query
- Start your response with "## Research Plan"
- Outline your approach to investigating this specific topic
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- Clearly state the specific topic you're researching to maintain focus throughout all iterations
- Identify the key aspects you'll need to research
- Provide initial findings based on the information available
- End with "## Next Steps" indicating what you'll investigate in the next iteration
- Do NOT provide a final conclusion yet - this is just the beginning of the research
- Do NOT include general repository information unless directly relevant to the query
- Focus EXCLUSIVELY on the specific topic being researched - do not drift to related topics
- Your research MUST directly address the original question
- NEVER respond with just "Continue the research" as an answer - always provide substantive research findings
- Remember that this topic will be maintained across all research iterations
</guidelines>

<style>
- Be concise but thorough
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
</style>"""
            elif is_final_iteration:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are in the final iteration of a Deep Research process focused EXCLUSIVELY on the latest user query.
Your goal is to synthesize all previous findings and provide a comprehensive conclusion that directly addresses this specific topic and ONLY this topic.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- This is the final iteration of the research process
- CAREFULLY review the entire conversation history to understand all previous findings
- Synthesize ALL findings from previous iterations into a comprehensive conclusion
- Start with "## Final Conclusion"
- Your conclusion MUST directly address the original question
- Stay STRICTLY focused on the specific topic - do not drift to related topics
- Include specific code references and implementation details related to the topic
- Highlight the most important discoveries and insights about this specific functionality
- Provide a complete and definitive answer to the original question
- Do NOT include general repository information unless directly relevant to the query
- Focus exclusively on the specific topic being researched
- NEVER respond with "Continue the research" as an answer - always provide a complete conclusion
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- Ensure your conclusion builds on and references key findings from previous iterations
</guidelines>

<style>
- Be concise but thorough
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
- Structure your response with clear headings
- End with actionable insights or recommendations when appropriate
</style>"""
            else:
                system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are currently in iteration {research_iteration} of a Deep Research process focused EXCLUSIVELY on the latest user query.
Your goal is to build upon previous research iterations and go deeper into this specific topic without deviating from it.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- CAREFULLY review the conversation history to understand what has been researched so far
- Your response MUST build on previous research iterations - do not repeat information already covered
- Identify gaps or areas that need further exploration related to this specific topic
- Focus on one specific aspect that needs deeper investigation in this iteration
- Start your response with "## Research Update {research_iteration}"
- Clearly explain what you're investigating in this iteration
- Provide new insights that weren't covered in previous iterations
- If this is iteration 3, prepare for a final conclusion in the next iteration
- Do NOT include general repository information unless directly relevant to the query
- Focus EXCLUSIVELY on the specific topic being researched - do not drift to related topics
- If the topic is about a specific file or feature (like "Dockerfile"), focus ONLY on that file or feature
- NEVER respond with just "Continue the research" as an answer - always provide substantive research findings
- Your research MUST directly address the original question
- Maintain continuity with previous research iterations - this is a continuous investigation
</guidelines>

<style>
- Be concise but thorough
- Focus on providing new information, not repeating what's already been covered
- Use markdown formatting to improve readability
- Cite specific files and code sections when relevant
</style>"""
        else:
            system_prompt = f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You provide direct, concise, and accurate information about code repositories.
You NEVER start responses with markdown headers or code fences.
IMPORTANT:You MUST respond in {language_name} language.
</role>

<guidelines>
- Answer the user's question directly without ANY preamble or filler phrases
- DO NOT include any rationale, explanation, or extra comments.
- Strictly base answers ONLY on existing code or documents
- DO NOT speculate or invent citations.
- DO NOT start with preambles like "Okay, here's a breakdown" or "Here's an explanation"
- DO NOT start with markdown headers like "## Analysis of..." or any file path references
- DO NOT start with ```markdown code fences
- DO NOT end your response with ``` closing fences
- DO NOT start by repeating or acknowledging the question
- JUST START with the direct answer to the question

<example_of_what_not_to_do>
```markdown
## Analysis of `adalflow/adalflow/datasets/gsm8k.py`

This file contains...
```
</example_of_what_not_to_do>

- Format your response with proper markdown including headings, lists, and code blocks WITHIN your answer
- For code analysis, organize your response with clear sections
- Think step by step and structure your answer logically
- Start with the most relevant information that directly addresses the user's query
- Be precise and technical when discussing code
- Your response language should be in the same language as the user's query
</guidelines>

<style>
- Use concise, direct language
- Prioritize accuracy over verbosity
- When showing code, cite the file path; do NOT prefix code lines with line numbers (the UI renders them automatically)
- Use markdown formatting to improve readability
</style>"""

        # Append the unified verification guard (grounding/citation/no-line-
        # numbers/unverified-flag) on top of the chat/deep-research system prompt
        # built above (inline for the WS path). Read fresh so a hot-reload via
        # the admin panel takes effect without a process restart.
        try:
            from api.prompts import VERIFICATION_GUARD as _guard
        except Exception:  # pragma: no cover - import-safe
            _guard = ""
        if _guard:
            system_prompt = system_prompt + "\n\n" + _guard

        # Fetch file content if provided
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"Successfully retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")
                # Continue without file content if there's an error

        # Format conversation history
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"

        # Create the prompt with context
        prompt = f"/no_think {system_prompt}\n\n"

        if conversation_history:
            prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

        # Check if filePath is provided and fetch file content if it exists
        if file_content:
            # Add file content to the prompt after conversation history
            prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

        # Only include context if it's not empty
        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            # Add a note that we're skipping RAG due to size constraints or because it's the isolated API
            logger.info("No context available from RAG")
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        model_config = get_model_config(request.provider, request.model)["model_kwargs"]

        if request.provider == "ollama":
            prompt += " /no_think"

            host = request.base_url
            if not host:
                from api.api import PROVIDER_SETTINGS
                stored = PROVIDER_SETTINGS.get("ollama", {})
                host = stored.get("base_url") or os.environ.get("OLLAMA_HOST")

            model = OllamaClient(host=host) if host else OllamaClient()
            # Low temperature + seed for deterministic generation (matches
            # generator.json). Defaults 0.1/0.9 so a missing config still yields
            # grounded, reproducible answers.
            _ws_ollama_options = {
                "temperature": model_config.get("temperature", 0.1),
                "top_p": model_config.get("top_p", 0.9),
                "num_ctx": model_config.get("num_ctx", 32000)
            }
            if "seed" in model_config:
                _ws_ollama_options["seed"] = model_config["seed"]
            model_kwargs = {
                "model": model_config["model"],
                "stream": True,
                "options": _ws_ollama_options
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider in ("openai_local", "openai", "openai_compatible") or "openai" in str(request.provider):
            logger.info(f"Using OpenAI-compatible API with model: {request.model}")

            # Initialize OpenAI-compatible client
            api_key = request.api_key
            if not api_key:
                from api.config_abstraction import get_task_config
                cfg = get_task_config("docgen") or {}
                api_key = cfg.get("api_key") or os.environ.get("LOCAL_OPENAI_API_KEY")

            base_url = request.base_url
            if not base_url:
                from api.config_abstraction import get_task_config
                cfg = get_task_config("docgen") or {}
                base_url = cfg.get("base_url") or os.environ.get("LOCAL_OPENAI_BASE_URL")

            model = OpenAIClient(base_url=base_url, api_key=api_key)
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config.get("temperature", 0.1)
            }
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]
            if "seed" in model_config:
                model_kwargs["seed"] = model_config["seed"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        else:
            # Default to OpenAI-compatible client for unknown providers
            logger.warning(f"Unknown provider: {request.provider}, using OpenAIClient as fallback")
            api_key = request.api_key
            if not api_key:
                from api.api import PROVIDER_SETTINGS
                stored = PROVIDER_SETTINGS.get("openai_local", {})
                api_key = stored.get("api_key") or os.environ.get("LOCAL_OPENAI_API_KEY")

            model = OpenAIClient(base_url=request.base_url, api_key=api_key)
            model_kwargs = {
                "model": request.model or "default",
                "stream": True,
                "temperature": model_config.get("temperature", 0.1)
            }
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]
            if "seed" in model_config:
                model_kwargs["seed"] = model_config["seed"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )

        # Process the response based on the provider
        try:
            if request.provider == "ollama":
                # Get the response and handle it properly using the previously created api_kwargs
                response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                # Handle streaming response from Ollama
                async for chunk in response:
                    text = None
                    if isinstance(chunk, dict):
                        text = chunk.get("message", {}).get("content") if isinstance(chunk.get("message"), dict) else chunk.get("message")
                    else:
                        message = getattr(chunk, "message", None)
                        if message is not None:
                            if isinstance(message, dict):
                                text = message.get("content")
                            else:
                                text = getattr(message, "content", None)

                    if not text:
                        text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None)

                    if not text and hasattr(chunk, "__dict__"):
                        message = chunk.__dict__.get("message")
                        if isinstance(message, dict):
                            text = message.get("content")

                    if isinstance(text, str) and text and not text.startswith('model=') and not text.startswith('created_at='):
                        clean_text = text.replace('<think>', '').replace('</think>', '')
                        await websocket.send_text(clean_text)
                # Explicitly close the WebSocket connection after the response is complete
                await websocket.close()
            elif request.provider == "openai_local":
                # Handle streaming response from OpenAI-compatible API
                try:
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    async for chunk in response:
                        text = None
                        # Standard OpenAI streaming format: choices[0].delta.content
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content'):
                                text = delta.content
                        # Fallback for dict-like chunks
                        elif isinstance(chunk, dict):
                            choices = chunk.get("choices", [])
                            if choices and len(choices) > 0:
                                text = choices[0].get("delta", {}).get("content")
                            else:
                                text = chunk.get("message", {}).get("content") or chunk.get("content")

                        if text:
                            logger.debug(f"OpenAI Local chunk: {text}")
                            await websocket.send_text(text)
                    await websocket.close()
                except Exception as e_openai:
                    logger.error(f"Error in OpenAI Local streaming: {str(e_openai)}")
                    await websocket.send_text(f"Error: {str(e_openai)}")
                    await websocket.close()
            else:
                # OpenAI-compatible client (default provider)
                try:
                    logger.warning("Using OpenAI-compatible client as default provider")
                    model_kwargs = {
                        "model": request.model or "default",
                        "stream": True,
                        "temperature": model_config.get("temperature", 0.1)
                    }
                    if "top_p" in model_config:
                        model_kwargs["top_p"] = model_config["top_p"]
                    if "seed" in model_config:
                        model_kwargs["seed"] = model_config["seed"]

                    api_kwargs = model.convert_inputs_to_api_kwargs(
                        input=prompt,
                        model_kwargs=model_kwargs,
                        model_type=ModelType.LLM
                    )

                    response = await model.acall(
                        api_kwargs=api_kwargs, model_type=ModelType.LLM
                    )
                    # Handle streaming response
                    async for chunk in response:
                        text = None
                        # Standard OpenAI streaming format: choices[0].delta.content
                        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content'):
                                text = delta.content
                        # Fallback for dict-like chunks
                        elif isinstance(chunk, dict):
                            choices = chunk.get("choices", [])
                            if choices and len(choices) > 0:
                                text = choices[0].get("delta", {}).get("content")
                            else:
                                text = chunk.get("message", {}).get("content") or chunk.get("content")

                        if text:
                            logger.debug(f"Default provider chunk: {text}")
                            await websocket.send_text(text)
                    await websocket.close()
                except Exception as e_default:
                    logger.error(f"Error in default provider streaming: {str(e_default)}")
                    await websocket.send_text(f"Error: {str(e_default)}")
                    await websocket.close()

        except Exception as e_outer:
            logger.error(f"Error in streaming response: {str(e_outer)}")
            error_message = str(e_outer)

            # Check for token limit errors
            if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                # If we hit a token limit error, try again without context
                logger.warning("Token limit exceeded, retrying without context")
                try:
                    # Create a simplified prompt without context
                    simplified_prompt = f"/no_think {system_prompt}\n\n"
                    if conversation_history:
                        simplified_prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

                    # Include file content in the fallback prompt if it was retrieved
                    if request.filePath and file_content:
                        simplified_prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

                    simplified_prompt += "<note>Answering without retrieval augmentation due to input size constraints.</note>\n\n"
                    simplified_prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

                    if request.provider == "ollama":
                        simplified_prompt += " /no_think"

                        # Create new api_kwargs with the simplified prompt
                        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                            input=simplified_prompt,
                            model_kwargs=model_kwargs,
                            model_type=ModelType.LLM
                        )

                        # Get the response using the simplified prompt
                        fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                        # Handle streaming fallback_response from Ollama
                        async for chunk in fallback_response:
                            text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                            if text and not text.startswith('model=') and not text.startswith('created_at='):
                                text = text.replace('<think>', '').replace('</think>', '')
                                await websocket.send_text(text)
                    else:
                        # OpenAI-compatible fallback (default provider)
                        logger.warning("Using OpenAI-compatible fallback")
                        model_config = get_model_config(request.provider, request.model)

                        # Use OpenAIClient for fallback
                        api_key = request.api_key
                        if not api_key:
                            from api.api import PROVIDER_SETTINGS
                            stored = PROVIDER_SETTINGS.get("openai_local", {})
                            api_key = stored.get("api_key") or os.environ.get("LOCAL_OPENAI_API_KEY")

                        fallback_client = OpenAIClient(base_url=request.base_url, api_key=api_key)
                        _fb_mk = model_config.get("model_kwargs", {})
                        fallback_model_kwargs = {
                            "model": _fb_mk.get("model", request.model or "default"),
                            "stream": True,
                            "temperature": _fb_mk.get("temperature", 0.1),
                        }
                        if "top_p" in _fb_mk:
                            fallback_model_kwargs["top_p"] = _fb_mk["top_p"]
                        if "seed" in _fb_mk:
                            fallback_model_kwargs["seed"] = _fb_mk["seed"]

                        fallback_api_kwargs = fallback_client.convert_inputs_to_api_kwargs(
                            input=simplified_prompt,
                            model_kwargs=fallback_model_kwargs,
                            model_type=ModelType.LLM
                        )

                        try:
                            fallback_response = await fallback_client.acall(
                                api_kwargs=fallback_api_kwargs,
                                model_type=ModelType.LLM
                            )
                            async for chunk in fallback_response:
                                text = None
                                # Standard OpenAI streaming format: choices[0].delta.content
                                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                                    delta = chunk.choices[0].delta
                                    if hasattr(delta, 'content'):
                                        text = delta.content
                                # Fallback for dict-like chunks
                                elif isinstance(chunk, dict):
                                    choices = chunk.get("choices", [])
                                    if choices and len(choices) > 0:
                                        text = choices[0].get("delta", {}).get("content")
                                    else:
                                        text = chunk.get("message", {}).get("content") or chunk.get("content")

                                if text:
                                    logger.debug(f"Fallback chunk: {text}")
                                    await websocket.send_text(text)
                        except Exception as fallback_error:
                            logger.error(f"Fallback error: {fallback_error}")
                            await websocket.send_text(f"\nError during fallback: {str(fallback_error)}")
                except Exception as e2:
                    logger.error(f"Error in fallback streaming response: {str(e2)}")
                    await websocket.send_text(f"\nI apologize, but your request is too large for me to process. Please try a shorter query or break it into smaller parts.")
                    # Close the WebSocket connection after sending the error message
                    await websocket.close()
            else:
                # For other errors, return the error message
                await websocket.send_text(f"\nError: {error_message}")
                # Close the WebSocket connection after sending the error message
                await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        try:
            await websocket.send_text(f"Error: {str(e)}")
            await websocket.close()
        except:
            pass


# =============================================================================
# Wiki Generation WebSocket Handler (7-section pipeline)
# =============================================================================

import json as _json
import asyncio


def _make_ws_repair_llm(llm_client, llm_kwargs, provider: str):
    """Build an async ``(prompt) -> str`` callable for the mermaid repair loop.

    Reuses the same OllamaClient/OpenAIClient already built for wiki generation
    so repair requests hit the same local model. Non-streaming. Returns None if
    the client/kwargs are unusable (repairs are then skipped).
    """
    if llm_client is None or not llm_kwargs:
        return None

    async def _call(prompt: str) -> str:
        try:
            kwargs = dict(llm_kwargs)
            kwargs["stream"] = False
            api_kwargs = llm_client.convert_inputs_to_api_kwargs(
                input=prompt, model_kwargs=kwargs, model_type=ModelType.LLM
            )
            response = await asyncio.to_thread(
                llm_client.call, api_kwargs=api_kwargs, model_type=ModelType.LLM
            )
            return _extract_llm_content(response)
        except Exception as e:  # pragma: no cover - depends on live LLM
            logger.warning("Mermaid repair LLM call failed: %s", e)
            return ""

    return _call


def _extract_llm_content(response) -> str:
    """Extract text content from various LLM response formats.
    
    Handles:
    - adalflow GeneratorOutput (has .raw_response which may be a ChatCompletion)
    - OpenAI ChatCompletion objects (choices[0].message.content)
    - Ollama response objects (message.content)
    - Plain strings
    """
    if response is None:
        return ""
    
    # 1. If it's already a string, return it
    if isinstance(response, str):
        return response
    
    # 2. adalflow GeneratorOutput — unwrap .raw_response first
    raw = getattr(response, 'raw_response', None)
    if raw is not None:
        # raw_response might be a ChatCompletion or a string
        return _extract_llm_content(raw)
    
    # 3. OpenAI ChatCompletion format: choices[0].message.content
    choices = getattr(response, 'choices', None)
    if choices and len(choices) > 0:
        message = getattr(choices[0], 'message', None)
        if message is not None:
            msg_content = getattr(message, 'content', None)
            if msg_content:
                return str(msg_content)
    
    # 4. Ollama format: response.message.content
    message = getattr(response, 'message', None)
    if message is not None:
        if isinstance(message, dict):
            return message.get('content', '')
        msg_content = getattr(message, 'content', None)
        if msg_content:
            return str(msg_content)
    
    # 5. .data field (some adalflow versions)
    data = getattr(response, 'data', None)
    if data and isinstance(data, str):
        return data
    
    # 6. Fallback — but avoid returning repr of ChatCompletion objects
    result = str(response)
    # If it looks like a ChatCompletion repr, it's wrong — return empty
    if result.startswith('ChatCompletion(') or result.startswith('GeneratorOutput('):
        return ""
    return result


async def handle_websocket_wiki_generate(websocket: WebSocket):
    """
    WebSocket handler for generating a complete 7-section wiki.
    
    Protocol:
    1. Client sends JSON request with repo info and settings
    2. Server sends progress messages: {"type": "progress", "section_index": N, "total": 7, "section_id": "...", "title": "..."}
    3. Server sends each completed section: {"type": "section", "section_id": "...", "title": "...", "content": "..."}
    4. Server sends final structure: {"type": "complete", "wiki_structure": {...}, "generated_pages": {...}}
    5. On error: {"type": "error", "message": "..."}
    """
    await websocket.accept()
    
    try:
        # 1. Receive request
        request_data = await websocket.receive_json()
        
        repo_url = request_data.get("repo_url", "")
        repo_type = request_data.get("type", "github")
        token = request_data.get("token")
        provider = request_data.get("provider", os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"))
        model_name = request_data.get("model", "")
        language = request_data.get("language", "en")
        comprehensive = request_data.get("comprehensive", True)
        base_url = request_data.get("base_url")
        api_key = request_data.get("api_key")
        embedding_model = request_data.get("embedding_model")
        excluded_dirs_str = request_data.get("excluded_dirs")
        excluded_files_str = request_data.get("excluded_files")
        included_dirs_str = request_data.get("included_dirs")
        included_files_str = request_data.get("included_files")
        
        logger.info(f"Wiki generation request: {repo_url}, provider={provider}, model={model_name}, lang={language}, comprehensive={comprehensive}")
        
        # 2. Import prompts
        from api.prompts import WIKI_SECTIONS, SECTION_PROMPTS, wrap_prompt, get_section_title
        
        # 3. Prepare RAG (clone repo, create embeddings)
        await websocket.send_json({"type": "progress", "message": "Preparing repository and embeddings...", "section_index": 0, "total": len(WIKI_SECTIONS)})
        
        try:
            rag = RAG(
                provider=provider,
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                embedding_model=embedding_model
            )
            
            # Parse file filters
            excluded_dirs = [unquote(d) for d in excluded_dirs_str.split('\n') if d.strip()] if excluded_dirs_str else None
            excluded_files = [unquote(f) for f in excluded_files_str.split('\n') if f.strip()] if excluded_files_str else None
            included_dirs = [unquote(d) for d in included_dirs_str.split('\n') if d.strip()] if included_dirs_str else None
            included_files = [unquote(f) for f in included_files_str.split('\n') if f.strip()] if included_files_str else None
            
            rag.prepare_retriever(repo_url, repo_type, token, excluded_dirs, excluded_files, included_dirs, included_files)
            logger.info(f"RAG retriever prepared for {repo_url}")
        except Exception as e:
            logger.error(f"Error preparing retriever: {str(e)}")
            await websocket.send_json({"type": "error", "message": f"Error preparing repository: {str(e)}"})
            await websocket.close()
            return
        
        # 4. Get RAG context for the whole repo
        try:
            repo_context_docs = rag.retriever(input="project overview architecture structure main components", top_k=30)
            if repo_context_docs and len(repo_context_docs) > 0:
                # Extract text from retrieved documents
                context_texts = []
                for doc_list in repo_context_docs:
                    if isinstance(doc_list, list):
                        for doc in doc_list:
                            text = getattr(doc, 'text', None) or str(doc)
                            file_path = ""
                            if hasattr(doc, 'meta_data') and doc.meta_data:
                                file_path = doc.meta_data.get('file_path', '')
                            if text and len(text.strip()) > 0:
                                context_texts.append(f"File: {file_path}\n{text[:2000]}")
                    else:
                        text = getattr(doc_list, 'text', None) or str(doc_list)
                        if text:
                            context_texts.append(text[:2000])
                repo_context = "\n\n---\n\n".join(context_texts[:20])
            else:
                repo_context = ""
        except Exception as e:
            logger.warning(f"Could not retrieve RAG context: {e}")
            repo_context = ""
        
        # 5. Extract repo name from URL
        url_parts = repo_url.rstrip('/').split('/')
        repo_name = url_parts[-1] if len(url_parts) > 0 else "repo"
        owner = url_parts[-2] if len(url_parts) > 1 else "unknown"
        
        # 6. Get model config
        try:
            model_config = get_model_config(provider, model_name)
        except Exception as e:
            logger.error(f"Error getting model config: {e}")
            await websocket.send_json({"type": "error", "message": f"Model configuration error: {str(e)}"})
            await websocket.close()
            return
        
        # 7. Generate each section sequentially
        generated_pages = {}
        previous_content = ""
        
        for idx, section_def in enumerate(WIKI_SECTIONS):
            section_id = section_def["id"]
            section_title = get_section_title(section_id, language)
            
            logger.info(f"Generating section {idx+1}/{len(WIKI_SECTIONS)}: {section_id} ({section_title})")
            
            # Send progress
            await websocket.send_json({
                "type": "progress",
                "message": f"Generating section {idx+1}/{len(WIKI_SECTIONS)}: {section_title}...",
                "section_index": idx + 1,
                "total": len(WIKI_SECTIONS),
                "section_id": section_id,
                "title": section_title
            })
            
            # Build prompt from template
            prompt_template = SECTION_PROMPTS.get(section_id, "")
            if not prompt_template:
                logger.warning(f"No prompt template for section: {section_id}")
                generated_pages[section_id] = {"title": section_title, "content": f"No content generated for {section_title}"}
                continue
            
            # Fill template variables with safe string replacement (not .format())
            # Using str.replace avoids errors from unmatched braces in Mermaid/JSON examples
            template_vars = {
                "repo_url": repo_url,
                "repo_name": repo_name,
                "repo_type": repo_type,
                "primary_language": "auto-detect",
                "file_count": "N/A",
                "main_directories": "see context below",
                "overview_content": previous_content[:3000] if previous_content else "N/A",
                "project_structure": "see context below",
                "main_files": "see context below",
                "previous_content": previous_content[:5000] if previous_content else "N/A",
                "previous_sections": previous_content[:3000] if previous_content else "N/A",
                "app_type": "web_application",
                "main_modules": "see context below",
                "api_endpoints": "see context below",
                "tech_stack": "see context below",
                "config_files": "see context below",
                "cicd_files": "see context below",
                "docker_files": "see context below",
                "components": "see context below",
                "modules": "see context below",
                "databases": "see context below",
                "entities": "see context below",
                "db_config": "see context below",
            }
            prompt = prompt_template
            for var_name, var_value in template_vars.items():
                prompt = prompt.replace("{" + var_name + "}", str(var_value))
            
            # Wrap with language and detail level
            prompt = wrap_prompt(prompt, language=language, comprehensive=comprehensive)
            
            # Append the unified verification guard (grounding/citation/no-line-
            # numbers/unverified-flag). Read fresh from api.prompts so a hot-reload
            # via the admin panel takes effect without a process restart.
            try:
                from api.prompts import VERIFICATION_GUARD as _guard
            except Exception:  # pragma: no cover - import-safe
                _guard = ""
            if _guard:
                prompt = prompt + "\n\n" + _guard
            
            # Add RAG context
            prompt += f"\n\n<repository_context>\n{repo_context[:8000]}\n</repository_context>\n"
            
            # Add /no_think for Ollama
            if provider == "ollama":
                prompt = "/no_think " + prompt + " /no_think"
            
            # Call LLM
            try:
                mk = model_config["model_kwargs"]
                
                if provider == "ollama":
                    host = base_url
                    if not host:
                        from api.api import PROVIDER_SETTINGS
                        stored = PROVIDER_SETTINGS.get("ollama", {})
                        host = stored.get("base_url") or os.environ.get("OLLAMA_HOST")
                    llm_client = OllamaClient(host=host) if host else OllamaClient()
                    # Low temperature + seed for deterministic wiki generation
                    # (matches generator.json). Defaults 0.1/0.9.
                    _wiki_ollama_options = {
                        "temperature": mk.get("temperature", 0.1),
                        "top_p": mk.get("top_p", 0.9),
                        "num_ctx": mk.get("num_ctx", 32000)
                    }
                    if "seed" in mk:
                        _wiki_ollama_options["seed"] = mk["seed"]
                    llm_kwargs = {
                        "model": mk.get("model", model_name),
                        "stream": False,
                        "options": _wiki_ollama_options
                    }
                else:  # openai_local
                    llm_api_key = api_key
                    if not llm_api_key:
                        from api.api import PROVIDER_SETTINGS
                        stored = PROVIDER_SETTINGS.get("openai_local", {})
                        llm_api_key = stored.get("api_key") or os.environ.get("LOCAL_OPENAI_API_KEY")
                    llm_client = OpenAIClient(base_url=base_url, api_key=llm_api_key)
                    llm_kwargs = {
                        "model": mk.get("model", model_name),
                        "stream": False,
                        "temperature": mk.get("temperature", 0.1),
                    }
                    if "top_p" in mk:
                        llm_kwargs["top_p"] = mk["top_p"]
                    if "seed" in mk:
                        llm_kwargs["seed"] = mk["seed"]
                
                api_kwargs = llm_client.convert_inputs_to_api_kwargs(
                    input=prompt,
                    model_kwargs=llm_kwargs,
                    model_type=ModelType.LLM
                )
                
                # Run sync LLM call in thread pool to avoid blocking the event loop
                # (blocking would prevent WebSocket progress messages from being sent)
                response = await asyncio.to_thread(
                    llm_client.call, api_kwargs=api_kwargs, model_type=ModelType.LLM
                )
                
                # Extract text content from LLM response
                content = _extract_llm_content(response)
                if not content:
                    logger.warning(f"Empty content from LLM for section {section_id}, response type: {type(response)}")
                    content = f"Failed to extract content for section {section_id}"
                
                # Clean up think tags
                content = content.replace('<think>', '').replace('</think>', '').strip()
                # Strip inline line-number prefixes inside code blocks (the UI
                # renders its own line numbers; duplicated prefixes are ugly).
                try:
                    from api.artifact_docgen import _strip_inline_line_numbers
                    content = _strip_inline_line_numbers(content)
                except Exception:  # pragma: no cover - import-safe
                    pass
                
                logger.info(f"Section {section_id} generated: {len(content)} chars")
                
            except Exception as e:
                logger.error(f"Error generating section {section_id}: {str(e)}")
                content = f"Error generating this section: {str(e)}"
            
            # Validate + repair any mermaid diagrams in this section before
            # storing/sending it. The repair loop reuses the same LLM client and
            # is non-fatal: if the Node verifier is unavailable the content is
            # returned unchanged. Emits an additive "mermaid_repair" progress
            # message (old clients ignore unknown message types).
            try:
                _repair_llm = _make_ws_repair_llm(llm_client, llm_kwargs, provider)

                async def _mermaid_progress(mstats):
                    try:
                        await websocket.send_json({
                            "type": "mermaid_repair",
                            "section_id": section_id,
                            "fixed": mstats.get("fixed", 0),
                            "failed": mstats.get("failed", 0),
                            "verified": mstats.get("verified", 0),
                            "broken": mstats.get("broken", 0),
                        })
                    except Exception:
                        pass

                content, _mstats = await run_repair_loop(
                    content, _repair_llm, on_progress=_mermaid_progress
                )
            except Exception as e:  # pragma: no cover - verifier must never break gen
                logger.warning("Mermaid repair loop failed for section %s: %s", section_id, e)
            
            # Store
            generated_pages[section_id] = {
                "id": section_id,
                "title": section_title,
                "content": content,
                "filePaths": [],
                "importance": "high",
                "relatedPages": []
            }
            
            # Accumulate for subsequent sections
            previous_content += f"\n\n## {section_title}\n{content[:2000]}"
            
            # Send completed section
            await websocket.send_json({
                "type": "section",
                "section_id": section_id,
                "title": section_title,
                "content": content
            })
        
        # 8. Build final wiki structure
        pages_list = []
        pages_dict = {}
        sections_list = []
        root_sections = []
        
        for section_def in WIKI_SECTIONS:
            sid = section_def["id"]
            page_data = generated_pages.get(sid, {})
            page_id = sid
            title = page_data.get("title", get_section_title(sid, language))
            content = page_data.get("content", "")
            
            page_obj = {
                "id": page_id,
                "title": title,
                "content": content,
                "filePaths": [],
                "importance": "high",
                "relatedPages": []
            }
            pages_list.append(page_obj)
            pages_dict[page_id] = page_obj
            
            sections_list.append({
                "id": f"section-{sid}",
                "title": title,
                "pages": [page_id]
            })
            root_sections.append(f"section-{sid}")
        
        wiki_structure = {
            "id": "wiki",
            "title": f"Wiki: {owner}/{repo_name}",
            "description": f"Documentation for {repo_url}",
            "pages": pages_list,
            "sections": sections_list,
            "rootSections": root_sections
        }
        
        # 9. Send complete message
        await websocket.send_json({
            "type": "complete",
            "wiki_structure": wiki_structure,
            "generated_pages": pages_dict
        })
        
        logger.info(f"Wiki generation complete for {repo_url}: {len(generated_pages)} sections")
        await websocket.close()
        
    except WebSocketDisconnect:
        logger.info("Wiki generation WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in wiki generation WebSocket: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except:
            pass
