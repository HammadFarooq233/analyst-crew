# Phase 0 Issue 5: Memory — search_memory tool for Planner

## Parent

`docs/prd-phase-0-walking-skeleton.md`

## What to build

Give the Planner Agent access to persistent memory via a scoped `search_memory` tool. This proves the memory integration pattern (per-agent, scoped by role, backed by Postgres) before Phase 3 deepens it.

The tool is created via LangMem's `create_search_memory_tool`, backed by the same Postgres store as the checkpointer. It is scoped to the Planner role — retrieves facts tagged for planning context (user preferences, prior analytical conventions, data source notes). The Planner's system prompt is updated to instruct it to call `search_memory` before drafting the Plan and to incorporate relevant facts into Plan decisions.

A helper function allows pre-seeding memory facts for testing: given a fact string, it writes it to the LangMem store under the Planner namespace. The CLI gains an optional `--memory-seed "Fact text"` flag.

The Planner's system prompt is updated to include: "Before drafting the Plan, call search_memory to retrieve any relevant facts, preferences, or conventions the user has established in prior sessions. Incorporate these into your Plan decisions."

## Acceptance criteria

- [ ] `search_memory` tool is created via `create_search_memory_tool` and registered on the Planner Agent
- [ ] The Planner's system prompt instructs it to call `search_memory` before drafting the Plan
- [ ] Pre-seeding a fact (e.g., "'unusual' means >15% swings by product category") and running the graph produces a Plan that references or reflects that fact
- [ ] `cli.py --memory-seed "'unusual' means >15% swings by product category" --question "Flag anything unusual in this month's sales"` produces a Plan that defines "unusual" as >15%
- [ ] Memory facts survive across sessions — restart the process, run without `--memory-seed`, and the Planner retrieves previously seeded facts
- [ ] Test: `test_memory.py::test_memory_influences_plan` seeds a fact, invokes the graph, asserts the Plan output references the fact
- [ ] Test: `test_memory.py::test_memory_persists_across_runs` seeds a fact in a first run, invokes in a second run without re-seeding, asserts the fact is still retrieved

## Blocked by

- Issue 004 (checkpointer — needs Postgres connection for the LangMem store)