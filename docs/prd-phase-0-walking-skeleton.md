# Phase 0 — Walking Skeleton

## Problem Statement

The Analyst Crew project has a fully specified architecture — 6 Agents, a Plan/Step/Handoff contract, a CrewState schema, per-agent memory access, and a two-level safety model — but none of it is wired together. Before building all 6 Agents and their tools, the Handoff contract between Planner and the rest of the Crew must be proven with a real end-to-end run touching every architectural layer. Without this, subsequent work (Fetcher, Analyst, Visualizer, Critic, Reporter) risks being built on top of a state/contract that doesn't survive contact with a real LLM.

## Solution

Build the thinnest end-to-end slice: one LangGraph graph with a single Planner node, wired to the real `CrewState` schema, a real Postgres checkpointer, a real Langfuse trace, one scoped `search_memory` tool, and one real LLM call through an OpenRouter free model. No mocking. The Planner receives a user question, queries memory for relevant facts/preferences, and produces a structured Plan — an ordered list of Steps, each naming a target Agent, what to produce, and what prior Steps it depends on. This Plan is the Handoff contract everything else will consume.

## User Stories

1. As a developer building on the Crew, I want a validated Plan output shape so that I know exactly what the Planner will send to downstream Agents in later Phases.
2. As a developer, I want the Planner to query memory during planning so that I can verify memory tool integration before Phase 3 deepens it.
3. As a developer, I want state persisted via a real Postgres checkpointer so that I can resume runs across process restarts without re-architecting later.
4. As a developer, I want every run traced in Langfuse so that I have observability from day one rather than retrofitting it.
5. As a developer evaluating the Planner, I want a DeepEval test case that asserts Plan structure correctness so that I can catch regressions when the Planner prompt changes.
6. As Priya, I want to ask a data question in plain English and receive a coherent analysis plan so that I can confirm the Crew understands my intent before any data is fetched.

## Implementation Decisions

### Project structure

- The project uses Python 3.12+ with `uv` for package management.
- A single package `analyst_crew/` holds all source code, organized by concern: `state/`, `agents/`, `tools/`, `orchestration/`.
- Configuration lives in a `config/` directory with TOML files, loaded via a dedicated config module.
- A `docker-compose.yml` defines Postgres and Langfuse containers.

### CrewState schema

The LangGraph state uses four top-level keys (from §13 of the vision doc):

```python
class CrewState(TypedDict):
    current_step: str
    steps_completed: Annotated[list[str], operator.add]
    artifacts: Annotated[dict[str, Any], merge_dicts]
    messages: Annotated[list, add_messages]
```

In Phase 0, `current_step` holds the Planner's current execution phase, `steps_completed` is empty after the Planner runs, `artifacts` stores the Plan under key `"plan"`, and `messages` holds the LLM conversation.

### Plan and Step shapes

The Planner produces a typed Plan containing an ordered list of Steps:

```python
class Step(BaseModel):
    step_id: str
    agent: Literal["Fetcher", "Analyst", "Visualizer", "Critic", "Reporter"]
    produces: str
    depends_on: list[str] = []

class Plan(BaseModel):
    steps: list[Step]
    user_question: str
```

Each Step's `depends_on` references `step_id` values of prior Steps. The Crew resolves these into explicit inputs during Handoff.

### Orchestration

- The graph is a LangGraph `StateGraph` with one node: `planner_node`.
- Graph entry is a raw user question string; the Planner node transforms it into a Plan stored in `state["artifacts"]["plan"]`.
- The `langgraph checkpointer` with `PostgresSaver` persists state after every node execution.
- The graph's `invoke()` accepts a `config` dict containing `thread_id` and `user_id` (session meta per §13 — not in state).

### LLM integration

- The Planner Agent uses OpenRouter's OpenAI-compatible API via `ChatOpenAI`, with the free model `google/gemma-4-26b-a4b-it:free` by default. The model can be changed in `config/settings.toml` as OpenRouter's free-model availability changes.
- The model key is loaded from environment (`.env`), never committed.
- The Planner's system prompt defines the Agent's role, the Plan/Step output format, and instructions to query memory before drafting a Plan.
- The LLM is called via LangChain's `ChatModel` interface, with structured output parsing into the `Plan` Pydantic model.

### Memory

- The `search_memory` tool is created via LangMem's `create_search_memory_tool`, backed by the same Postgres store as the checkpointer.
- Scoped to the Planner role: retrieves facts tagged for planning context (e.g., user preferences, prior analytical conventions).
- The Planner's system prompt instructs it to call this tool before generating the Plan, and to incorporate relevant facts into Plan decisions.

### Observability

- Langfuse is initialized via environment variables (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).
- Tracing uses LangChain's callback integration; no manual span creation needed in Phase 0.
- Each run is tagged with `session_id` (from `config["configurable"]["thread_id"]`) and `agent: "Planner"`.

### CLI entry point

- A `cli.py` at the project root accepts a question argument (`--question "..."` or stdin) and prints the resulting Plan as formatted text.
- Supports `--memory-seed` flag to pre-populate a memory fact for testing.
- Exits with the Plan displayed; no interactive loop yet.

## Testing Decisions

### What makes a good test

- Tests assert external behavior (state shape, Plan structure, trace existence), not internal agent reasoning or exact LLM text.
- No mocking of the LLM — Phase 0 validates real model behavior. Subsequent Phases can introduce mocked tests for speed.
- State assertions check keys, types, and required fields; never exact string matching on LLM outputs.

### Test seams

1. **Graph `.invoke()`** — primary seam: feed a user question, get back a `CrewState` with a populated Plan in `artifacts["plan"]`.
2. **Plan validation** — assert the Plan is a valid `Plan` model with ≥1 Step, each Step names a known Agent, no circular dependencies.
3. **Memory integration** — pre-seed a known fact, invoke the graph, assert the resulting Plan reflects the fact (e.g., a memory fact about "unusual = >15% swing" changes the Plan's methodology notes).
4. **Checkpointer persistence** — invoke with `thread_id`, then re-invoke with the same `thread_id` and a follow-up question; assert prior state was loaded.
5. **Langfuse trace** — after a run, query Langfuse API for the trace matching the run's `thread_id`; assert it exists with correct metadata.

### Test modules

- `tests/test_planner.py` — graph invoke + Plan validation + memory integration.
- `tests/test_checkpointer.py` — state persistence across invocations.
- `tests/test_tracing.py` — Langfuse trace existence.
- `tests/test_eval.py` — DeepEval test case with a golden question.

### No prior art

This is a greenfield project. Tests follow standard pytest conventions with `pytest-asyncio` for async graph invocations.

## Out of Scope

- Fetcher, Analyst, Visualizer, Critic, Reporter Agents — Phase 2+.
- Real tool execution (`execute_python`, `Postgres.query`, `FRED.fetch`, `CSV.read`, `chart`) — Phase 1+.
- Multi-step graph execution (the Planner is the only node).
- Human-in-the-loop approval gates — Phase 6.
- Memory write/update tools (only search/read is wired) — Phase 3.
- DeepEval CI pipeline — Phase 5 (test case is written, not wired to CI yet).
- Docker Compose for project deployment (only Postgres + Langfuse containers are defined).
- Any UI beyond the CLI.
- Structured logging beyond Langfuse traces — Phase 4.

## Further Notes

- The vision doc's §13 Architecture Decisions Log is the source of truth for design rationale. This PRD implements those decisions, not re-litigates them.
- Use the domain vocabulary from `CONTEXT.md` throughout: Crew, Agent, Planner, Plan, Step, Handoff, Tool, Memory, Report.
- OpenRouter free models have rate limits and availability can change. The CLI should surface limit errors clearly rather than silently retrying.
- If OpenRouter free-model availability becomes a blocker mid-development, the fallback is Ollama with a local model — the LLM provider is config-driven, not hardcoded.
- The `.env` file must never be committed. A `.env.example` file documents required variables without values.
