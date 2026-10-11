# Phase 2 (Full orchestration) — acceptance status

Spec issue #14 (tickets #15–#21), design in `docs/adr/0003-visualizer-selects-critic-judges-code-owns-numbers.md`, building on ADR-0001 and ADR-0002. Recorded 2026-10-11, branch `phase-2-orchestration` (PR #22).
The Reporter slice evidence (`docs/phase-2-reporter-acceptance-status.md`) and the Phase 1 evidence (`docs/phase-1-acceptance-status.md`) remain valid, with one correction noted under Known constraints.

## Result

All six Agents (Planner, Fetcher, Analyst, Visualizer, Reporter, Critic) run end to end on the README scenario (`data/sales.csv`, as-of 2025-06-30) with Azure OpenAI (`LLM_PROVIDER=azure`), local Postgres and local Langfuse.

| Acceptance criterion | Evidence |
| --- | --- |
| Live README scenario produces `chart_mode: llm`, rendered PNGs and a recorded Critique with a `critic_outcome` | Fresh CLI run on 2026-10-11 with the final code: exit 0, `chart_mode: llm`, three PNG Chart Artifacts listed with titles and paths, Critique recorded as `clean` (no Reviewer Notes section). Ten consecutive full-Crew runs after the prompt fixes below: `chart_mode: llm` 10/10, `critic_outcome: clean` 10/10 |
| Forced Visualizer failure shows a marked fallback and the recorded failure on the Langfuse trace | `tests/test_visualizer_live.py` (forced-failure test): trace metadata carries `chart_mode: fallback` and `chart_failure`; the Report still ships with the fallback charts, visibly marked |
| Forced Critic failure ships the Report with the failure recorded on the trace | `tests/test_critic_live.py` (forced-failure test): the un-reviewed Report ships, trace metadata carries `critic_failure`, the run does not fail |
| Visualizer, Critic and revision spans each nest their model generation | `tests/test_visualizer_live.py` (`visualizer_node` span), `tests/test_critic_live.py` (`Critic` span and the `Reporter (revision)` span), each with a `GENERATION` descendant, against local Langfuse v2 |
| Seeded-flaw eval passes against the real Critic | `tests/test_critic_eval.py` (`-m eval`): the planted causal-overreach Claim is flagged as `causal_overreach` on `claim_1`; the control Report yields no causal-overreach issue. Passed on every live run, including the final full-tier run |
| Unchanged re-run reuses charts, Report and Critique without model calls | Live through the Postgres checkpointer: a second invocation on the same thread printed identical output with `chart_mode: llm` and no Reviewer Notes. Offline: resume tests for the Visualizer, Reporter and Critic keys, including cascade on changed source data and retry of recorded failures |
| Full suite result recorded | Offline: 293 passed. Opt-in tiers (`-m "llm or integration or eval"`): 21 passed, 1 skipped (see below) |

## What was and was not exercised live

- Live: Azure OpenAI for all LLM Agents; local Postgres (checkpointer plus seeded `sales_targets`, so the target comparison was `available`); local Langfuse v2.
- Not exercised: the FRED CPI path (`tests/test_fred.py` skipped, no `FRED_API_KEY`). In the live Report CPI appears as a `not_configured` Caveat; the available path is covered offline with a stub.
- The `revised` outcome was not seen in a live README run, because the calibrated Critic returns `clean` on a sound Report. The revision path is covered offline through the Crew harness (revised, unresolved after revision, failed revision) and its span nesting by the live `Reporter (revision)` test with a scripted Critique.
- The Critic's quality is measured only by the seeded-flaw eval and its control, as the spec intends. It has not been tested on Reports with subtle real flaws.
- Environment: Postgres and Langfuse ran as hand-started containers (`postgres:16-alpine`, `ghcr.io/langfuse/langfuse:2` with its own `langfuse` database), not through `docker-compose.yml`, because the `postgres:16` image pull stalled. Langfuse refuses to migrate into a database that already holds the app's tables, so it needs its own database whenever the app has already created its tables there.

## What live QA found and fixed before sign-off

- **Visualizer fell back in 3 of 4 live runs.** The model built `grouped_bar` series of unequal length and the single retry failed too. Fixed with a catalog-derived chart menu of equal-length, same-order id lists in the prompt, stricter `grouped_bar` rules, and validation feedback that names each series and its length. The model still supplies no values or labels. Afterwards: 20/20 Visualizer-only runs and 10/10 full-Crew runs were `llm`.
- **Critic returned `unresolved` on every live run**, flagging `unsupported_emphasis` on most Claims (for example "largest", which the cited numbers support). Fixed with a calibrated prompt (an empty issue list is the expected outcome for a sound Report; comparisons the Evidence Entries show are not issues) and a scratch `review` field the model writes before its issues, which is discarded and not stored. Afterwards: 10/10 `clean`, with the seeded-flaw eval still passing.
- **A Reviewer Note began with a bare number** ("1 only identifies…"). Not a substitution bug: the model opened its rationale with a placeholder. The prompt now requires the rationale to say what a value is.
- **Live Planner sometimes omitted the Visualizer Step**, failing the run. The validation error now names the missing Step and the repair loop allows two retries. The live memory-influences-plan test went from 5/8 to 9/10 passing; the remaining failure was a malformed-JSON parse error from the model.
- **`test_forced_reporter_failure_records_report_mode_and_failure_on_the_trace` failed** because its scripted Plan lacked the Visualizer and Critic Steps that a default Crew now requires. The test now restricts the Crew to Fetcher, Analyst and Reporter.

## Known constraints

- Correction to the Reporter slice document: resuming through the CLI against live Postgres still prints "Deserializing unregistered type…" warnings (Plan, BaselineFindings, ChartArtifact, Report, EvidenceCatalog, Critique). The allowlist in `analyst_crew/state/serde.py` works with the in-memory checkpointer (`tests/test_state_serde.py`) but is not taking effect on the `PostgresSaver` path in `analyst_crew/agents/graph.py`. The warnings are harmless today but become errors under `LANGGRAPH_STRICT_MSGPACK=true`.
- After a revision the Critic reviews the revised Report once (a second Critic call) to decide `revised` versus `unresolved`; ADR-0003 records this. A Fallback Report is reviewed but not revised.
- The CLI prints `chart_mode` but not the recorded `chart_failure`; the reason for a fallback is on the run metadata and the Langfuse trace.
- The Planner and Critic are free-form LLM output, so live runs are not fully deterministic. The Plan shape is validated and repaired, never trusted.
