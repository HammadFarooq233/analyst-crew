# Phase 0 Issue 1: Scaffold — package, config, docker-compose

## Parent

`docs/prd-phase-0-walking-skeleton.md`

## What to build

Set up the project skeleton that every subsequent Phase 0 issue depends on: Python package structure with `uv`, configuration loading from TOML, environment variable management via `.env`, and Docker Compose for Postgres and Langfuse containers.

The package is `analyst_crew/` with three subdirectories: `state/`, `agents/`, `tools/`. Configuration lives in `config/` with a TOML file loaded by a config module. A `docker-compose.yml` at the project root defines Postgres (with persistent volume) and Langfuse (self-hosted, MIT license) containers. A `.env.example` documents all required environment variables (Gemini API key, Langfuse keys, Postgres connection string) without values.

No graph, no agents, no LLM — just the structure that everything else plugs into.

## Acceptance criteria

- [ ] `uv init` completed; `pyproject.toml` has project metadata and dependencies (langgraph, langchain-google-genai, langmem, langfuse, psycopg2-binary, pydantic, python-dotenv, deepeval, pytest)
- [ ] `analyst_crew/state/`, `analyst_crew/agents/`, `analyst_crew/tools/` exist with `__init__.py` files
- [ ] `config/` directory exists with a `settings.toml` and a config-loading Python module
- [ ] `docker-compose.yml` defines Postgres (port 5432, named volume) and Langfuse (port 3000) containers
- [ ] `docker compose up -d` starts both containers successfully
- [ ] `.env.example` exists with all required keys documented but empty
- [ ] `.env` is in `.gitignore`

## Blocked by

None — can start immediately.