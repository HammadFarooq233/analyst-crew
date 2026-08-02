# Phase 0 Issue 2: CrewState + Plan/Step models + stub graph + CLI

## Parent

`docs/prd-phase-0-walking-skeleton.md`

## What to build

Define the data contracts and wire them into a minimal LangGraph graph with a CLI entry point. The Planner Agent is stubbed — it returns a hardcoded Plan — so this issue validates the shape of state, models, and orchestration without any LLM dependency.

The `CrewState` TypedDict has four top-level keys: `current_step` (str, overwrite), `steps_completed` (Annotated[list[str], operator.add], append), `artifacts` (Annotated[dict[str, Any], merge_dicts], keyed dict), `messages` (Annotated[list, add_messages]). The `merge_dicts` reducer is a custom reducer that shallow-merges new key-value pairs into the existing dict.

The `Plan` and `Step` Pydantic models define the Handoff contract. A `Step` has `step_id: str`, `agent: Literal["Fetcher", "Analyst", "Visualizer", "Critic", "Reporter"]`, `produces: str`, and `depends_on: list[str]`. A `Plan` has `steps: list[Step]` and `user_question: str`.

The LangGraph `StateGraph` has one node: `planner_node`. It reads `user_question` from state `messages` (first HumanMessage), produces a hardcoded `Plan` with 2–3 Steps, and stores it at `state["artifacts"]["plan"]`. The graph compiles and `.invoke()` succeeds.

The CLI (`cli.py` at project root) accepts `--question` (or reads from stdin), invokes the graph, and prints the Plan as formatted text. It loads config and environment variables but does not yet wire a real LLM, checkpointer, tracing, or memory.

## Acceptance criteria

- [ ] `CrewState` TypedDict with correct types and reducers (overwrite for `current_step`, `operator.add` for `steps_completed`, custom `merge_dicts` reducer for `artifacts`, `add_messages` for `messages`)
- [ ] `Step` and `Plan` Pydantic models with all fields, validators enforce `step_id` uniqueness and no self-referencing `depends_on`
- [ ] LangGraph `StateGraph` compiles with a `planner_node` and an edge from START → planner_node → END
- [ ] `graph.invoke()` with a user question returns a `CrewState` containing a valid `Plan` in `artifacts["plan"]`
- [ ] The stub Plan has ≥2 Steps, each naming a real Agent, with correct dependency ordering
- [ ] `cli.py` accepts `--question "Compare this month's sales to last month"` and prints the Plan
- [ ] `cli.py` accepts piped input: `echo "my question" | python cli.py`
- [ ] Test: `test_planner.py::test_state_shape` asserts state keys, types, and Plan validity against Pydantic model

## Blocked by

- Issue 001 (scaffold — needs package structure and dependencies)