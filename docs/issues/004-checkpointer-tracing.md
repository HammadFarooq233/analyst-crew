# Phase 0 Issue 4: Postgres checkpointer + Langfuse tracing

## Parent

`docs/prd-phase-0-walking-skeleton.md`

## What to build

Wire persistence and observability into the Planner graph. After this issue, every run's state survives a process restart (Postgres checkpointer) and every run produces a visible trace (Langfuse).

The checkpointer uses LangGraph's `PostgresSaver` connected to the Postgres container from Issue 001. The graph's `.invoke()` accepts a `config` dict with `configurable: {thread_id, user_id}`. Re-invoking with the same `thread_id` restores prior state — the checkpointer replays `steps_completed` and `artifacts` from the previous run.

Langfuse tracing uses LangChain's callback integration. The callback handler is initialized from environment variables (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` pointing at the Docker container). Traces are tagged with `session_id` (the `thread_id`) and `agent: "Planner"`. No manual span creation — the callback captures the full LLM invocation automatically.

Session metadata (`user_id`, `thread_id`) lives in `config["configurable"]` per the vision doc's §13 — not duplicated in `CrewState`. The CLI accepts `--thread-id` to resume a prior session.

## Acceptance criteria

- [ ] `PostgresSaver` is initialized and connected to the Docker Postgres container
- [ ] `graph.invoke(state, config)` with a `thread_id` persists state after the run
- [ ] Re-invoking with the same `thread_id` and a follow-up question shows prior `steps_completed` and `artifacts` loaded from the checkpointer
- [ ] `cli.py --thread-id <id> --question "..."` resumes or creates a thread
- [ ] Langfuse callback handler is initialized from `.env` variables
- [ ] After a run, the trace is visible in the Langfuse UI (http://localhost:3000) with correct `session_id` and `agent` tags
- [ ] Test: `test_checkpointer.py::test_state_persistence` creates a thread, runs, re-runs with same thread_id, asserts prior state is loaded
- [ ] Test: `test_tracing.py::test_trace_exists` runs the graph and queries Langfuse API for the trace matching the thread_id

## Blocked by

- Issue 003 (Planner with LLM — needs a real graph run to persist and trace)