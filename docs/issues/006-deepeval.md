# Phase 0 Issue 6: DeepEval golden test case

## Parent

`docs/prd-phase-0-walking-skeleton.md`

## What to build

Create a DeepEval test case that serves as the project's first eval gate. This validates the end-to-end Planner pipeline with a golden question and asserts structural properties of the Plan — not exact text matching.

The test case defines a fixed user question ("Compare this month's sales to last month and flag anything unusual") and invokes the full Planner graph (LLM + memory + checkpointer + tracing). DeepEval's `GEval` metric asserts:

1. The Plan contains ≥2 Steps.
2. Every Step's `agent` field is one of the five known Agent names (Fetcher, Analyst, Visualizer, Critic, Reporter).
3. No Step's `depends_on` references its own `step_id`.
4. The Plan's Steps are ordered such that dependencies precede dependents.
5. At least one Step references data retrieval (Fetcher agent) and at least one references analysis (Analyst agent).

The test uses a seeded memory fact to make the output deterministic enough for structural assertions. It is marked with `@pytest.mark.eval` so it can be run separately from fast unit tests.

## Acceptance criteria

- [ ] `tests/test_eval.py` contains a DeepEval test case with a fixed golden question
- [ ] The test uses `GEval` metric with the 5 structural assertions above
- [ ] The test seeds a memory fact before invoking the graph (to stabilize output)
- [ ] `python -m pytest tests/test_eval.py -m eval` passes
- [ ] `python -m pytest tests/ -m "not eval"` skips the eval test (fast unit test suite is unaffected)
- [ ] The test is documented with a comment explaining what it checks and why structural assertions (not exact text) were chosen

## Blocked by

- Issue 005 (memory — the eval seeds a memory fact for deterministic output)