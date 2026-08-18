# DeepWiki Tests

This directory contains all tests for the DeepWiki / Productarium project, organized by type and scope into a unified, structured folder.

## Directory Structure

```
tests/
├── unit/                         # Unit tests - fast, isolated component testing
│   ├── test_all_embedders.py     # Embedder configuration & factory tests
│   ├── test_extract_repo_name.py # Repository URL parsing & name extraction tests
│   └── test_mermaid_verifier.py # Mermaid diagram verification & auto-fix tests
├── integration/                  # Integration tests - component & API testing
│   ├── test_admin_public.py      # Admin panel CRUD & public API token tests
│   ├── test_auth_flows.py        # Authentication setup, login, password reset & OIDC tests
│   ├── test_docgen_async.py      # Async artifact documentation generation tests
│   ├── test_expert_agent.py      # Expert Agent SSE chat & document generation tests
│   ├── test_foundation.py        # Database, settings store & provider config tests
│   ├── test_full_integration.py  # End-to-end integration tests
│   ├── test_integrations.py      # GitHub, GitLab, Confluence & MCP connector tests
│   └── test_knowledge_tree.py    # Knowledge tree CRUD & markitdown upload tests
├── run_tests.py                 # Unified test runner script
└── README.md
```

## Running Tests

### Standard Pytest Execution
```bash
pytest                            # Run all tests
pytest tests/unit/                # Run unit tests
pytest tests/integration/         # Run integration tests
pytest tests/unit/test_mermaid_verifier.py # Run a single test file
```

### Using Test Runner Script
```bash
python tests/run_tests.py               # Run all tests
python tests/run_tests.py --unit        # Run unit tests only
python tests/run_tests.py --integration # Run integration tests only
```

## Local Environment

All tests run locally using SQLite in-memory database and mocks/fakes for local LLMs (Local OpenAI-compatible API). No cloud API keys required.
