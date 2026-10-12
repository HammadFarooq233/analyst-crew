# Phase 3 (Memory, deepened) — acceptance status

Spec issue #23 (tickets #27–#33, plus the Critic clean-Report eval #25), design in `docs/adr/0005-deterministic-role-scoped-memory-and-crew-level-memory-update.md`, building on ADR-0001 to ADR-0004. Recorded 2026-10-11, branch `ticket-33` (based on `phase-3-memory`). Format follows `docs/phase-2-acceptance-status.md`; the inference eval history is in `docs/phase-3-inference-eval.md`.

## Result

The falsifiable two-session scenario passes live with both controls, on Azure OpenAI (`LLM_PROVIDER=azure`), a real Postgres and a real Langfuse v2. One caveat is stated under "Live scenario runs" below: the first runs of the new test included one failure of the test itself and one failure caused by the live Reporter.

| Acceptance criterion | Evidence |
| --- | --- |
| Live scenario passes with both controls | `tests/test_memory_live.py` (`-m "llm and integration"`). Session 1: `remember("for me, unusual means a swing over 8%")` through the real model stores a typed preference in Postgres. Session 2, a new thread with no restated threshold: North (+12.0%, 6.0% of the prior total, pre-computed) is a Material Change, `materiality.source` is `memory` at 8%, and the Report has an Applied Memories section. The 15% / 5% defaults call nothing material on the same data (control zero, run first). Control one: after `forget`, a new thread shows no Material Change and the preference is not applied. Control two: a different user sees no Material Change and no Applied Memories section. Passed on 4 of the 4 runs of the final test (3 standalone, 1 inside the full opt-in tier run) |
| Applied Memories visible on the Report and on the Langfuse trace | Report: the same test asserts the rendered Report contains `## Applied Memories`, that the preference's memory id is the only applied *preference*, that `report_property_violations` is empty (every displayed number is an Evidence Entry value; the threshold is shown through the Tool's own entry) and that the revenue bridge still reconciles (gap 0). Trace: the test fetches the trace from Langfuse by id and asserts `applied_memory_ids` in the trace metadata contains the preference id; for the other user the key is absent |
| Inference eval passes against the real model | `tests/test_memory_inference_eval.py` (`-m eval`): all 5 cases passed in the final full-tier run (3 proposer calls per case; properties only). History of earlier prompt iterations is in `docs/phase-3-inference-eval.md` |
| Clean-Report Critic eval (#25) landed and passed | `tests/test_critic_eval.py::test_critic_returns_no_issues_on_a_clean_report` (`-m eval`): **3 of 3 runs returned no issues** (one test invocation, `CLEAN_RUNS = 3`). The seeded-flaw eval and its control passed in the same invocation. One invocation was recorded; this is a small sample, not a rate |
| Full suite result recorded | See "Suite results" |

## Live scenario runs

The scenario uses a four-row CSV written to a temporary directory (May: North 100, South 100; June: North 112, South 100) so the change between thresholds is known in advance: default thresholds (15% relative, 5% of the prior total) call nothing material, an 8% threshold calls North. The Planner, Fetcher, Analyst, Visualizer, Reporter and Critic are all real. Only the question is fixed ("Compare June 2025 sales to May 2025 and flag anything unusual").

Six runs of the test while it was being finished:

| Run | Outcome |
| --- | --- |
| 1 | Failed on the test's own assertion: it expected the preference to be the only applied memory, but a Past-run Summary written by the earlier control run (same user, same data hash) was also applied. That is correct product behaviour; the test now filters to the preference kind |
| 2 | Failed: the Reporter hit a terminal failure on the last control run (after forgetting). The run correctly shipped a Fallback Report, as designed, but the test asserts no terminal failure. The cause was not captured (the assertion message now prints the failure's agent and error). Not reproduced |
| 3, 4, 5 | Passed |
| 6 | Passed, inside the full opt-in tier run (below) |

So four of six runs passed; the two failures are explained above, and neither was a memory-behaviour failure. A live Reporter failure can fail this test because the test does not tolerate a Fallback Report.

## Suite results

All opt-in runs used a scratch Postgres database (`analyst_tier_runs`, with `sales_targets` seeded) so the development database was untouched; the live scenario test creates and drops its own database.

| Tier | Result |
| --- | --- |
| Offline (`-m "not llm and not integration and not eval"`) | 443 passed, 40 deselected (re-run after merging the tip of `phase-3-memory`, which was already included) |
| Opt-in, `-m "llm or integration or eval"` | 40 selected. The first invocation was stopped by me while `tests/test_critic_live.py` was running (I mistook a slow live test for a hang), so it contributed only the Critic eval results above (seeded-flaw eval, its control and the clean-Report eval, all passed). The second invocation (Critic eval file deselected, 37 selected): 35 passed, 1 skipped, 1 failed, in 288 s |

- **The one failure was a stale test, fixed here.** `tests/test_eval.py::test_golden_plan_structure` still called `PlannerAgent(store=...)`, which Phase 3 removed (the Planner is handed role-scoped Memory text). It now passes `memory_context`, and passed live afterwards on its own (1 passed). It had been broken since the Planner slice of Phase 3 because the default suite deselects it.
- **Skipped:** `tests/test_fred.py::test_retrieve_cpi_against_live_fred_api`, because no `FRED_API_KEY` is configured.
- Included and passing in that invocation: the Postgres checkpointer resume test, the live Critic and Reporter span-nesting tests, the forced-failure tests, the live Reporter and Visualizer tests, the Postgres store contract, `test_memory.py` (memory influences the plan; memory persists across runs), the inference eval (5 of 5 cases) and the new live scenario.

## What was and was not exercised live

- Live: Azure OpenAI for every LLM call in the scenario (including `remember`), local Postgres (checkpointer plus the Memory store), local Langfuse v2 reached over HTTP for the trace assertion.
- Inference was **not** part of the live scenario: the scenario does not pass an inference model, so no inferred memory or pending preference was exercised in the two-session run. Inference is covered live only by the eval (proposer properties) and offline by the Crew-harness tests.
- Free-text facts and Past-run Summaries were not asserted live. A Past-run Summary was applied incidentally in the scenario runs (see run 1) because earlier runs for the same user on the same data wrote one.
- A storage outage (the "defaults used" Caveat) was not forced live; it is covered offline.
- Resume with a changed or forgotten memory on the *same* thread was not exercised live; the scenario uses a new thread for each session, as the spec's scenario does. It is covered offline through the Crew harness.
- Not exercised: the FRED CPI path (no key); Langfuse spans for memory retrieval and write beyond the metadata markers (only `applied_memory_ids` was asserted on the trace).
- The grouping-dimension preference named in the spec was not built (see Known constraints).
- Environment: the Docker services were the ones already running locally on ports 5434 (Postgres) and 3000 (Langfuse). The model is non-deterministic, so repeated runs are samples, not guarantees.

## Known constraints

- The typed-preference vocabulary has two keys (relative and absolute Material Change thresholds). The "grouping dimension emphasised" preference in the spec has no analysis parameter to map onto and was left out; ADR-0005 records this.
- `tests/test_memory_live.py` fails if the live Reporter hits a terminal failure on any of its five runs (it asserts no terminal failure), which happened once in six runs.
- On the `PostgresSaver` path the serializer allowlist still does not take effect (harmless "Deserializing unregistered type" warnings), as recorded in `docs/phase-2-acceptance-status.md`.
- The model's output is free-form, so the Critic clean-Report result (3 of 3) and the inference eval (5 of 5) are small samples.

## Manual QA pass (2026-10-12)

Driven by hand through the CLI on the real model, against a scratch Postgres database (dropped afterwards).

Worked as specified:
- The default run flags only the enterprise segment at 15%. After `remember "unusual means a swing over 8%"`, a new thread reports the 8% threshold with source `memory` and lists it under Applied Memories. A second user still gets the default.
- Rerunning the same thread after `forget` returned to the default threshold. `forget` on an unknown id, an already forgotten id and another user's id exits 1 with a clear message. `list --history` shows the superseded and forgotten records.
- Out-of-range (5000%), vague and "ignore the Office category" sentences are stored as free-text facts and change no parameter. An empty sentence stores nothing.
- A threshold in the question overrides the stored preference and the Report says so. An inferred preference stays pending (and is printed after the run), `confirm` by another user fails, `confirm` by its owner applies it, and a later `remember` supersedes it.
- An unreachable Postgres makes `list` exit 1 with a one-line message.

Found and fixed:
- Every free-text fact produced an identical Applied Memories line, so the reader could not tell them apart. Facts with one origin now share one line that points to `list`.
- A stored fact reading "unusual means a swing over 5000%" led the live Reporter to write a Finding saying the 17% enterprise change was "below the unusual cutoff", contradicting the 8% threshold. The Reporter prompt now says facts never define a threshold or what counts as unusual; a live rerun no longer produced that Finding. This is a prompt guard, not a code guarantee.

Not covered or still open:
- A memory-only storage outage during a run was not reproduced live (a full Postgres outage still fails in the checkpointer, as before Phase 3). Offline tests cover the memory-only case.
- The question parser only reads thresholds phrased like "10% relative threshold"; "only care about swings over 12%" is ignored by it (the inference step still proposes a pending preference from it).
- `remember` without `POSTGRES_CONNECTION_STRING` in the shell still uses the value from `.env`, so it is not refused in that case.
