# Phase 2 slice 1 (Reporter) — acceptance status

Issue #13, design in `docs/adr/0002-reporter-evidence-catalog-and-claims.md`. Recorded 2026-09-20.
The Phase 1 live-acceptance evidence about Findings (`docs/phase-1-acceptance-status.md`) remains valid and unchanged.

## Result

| Acceptance criterion | Evidence |
| --- | --- |
| The documented README scenario produces an `llm`-mode Report with every number traceable to an Evidence Entry | `pytest -m llm tests/test_reporter_live.py` and `pytest -m eval tests/test_eval.py -k report` ran the scenario (`data/sales.csv`, as-of 2025-06-30) through the real Azure OpenAI model: `report_mode == "llm"`, zero property violations (no digit outside a placeholder, every limitation and the quarantine/duplicate counts in Caveats, every Material Change cited, every rendered number matches a catalog display) |
| Every Findings limitation is present | The same property check, plus offline tests for every mode (`llm`, `fallback`) and for absent, unavailable and not-configured optional sources |
| A forced Reporter failure produces a visibly marked Fallback Report with the failure recorded on the trace | `tests/test_reporter_live.py::test_forced_reporter_failure_records_report_mode_and_failure_on_the_trace` against the local Langfuse: trace metadata carries `report_mode: fallback` and the Reporter `terminal_failure` |
| The Reporter is its own span with the LLM generation nested under it | `tests/test_reporter_live.py::test_reporter_is_its_own_trace_span_with_the_llm_generation_nested_under_it` (Langfuse v2 API: a `reporter_node` observation with a `GENERATION` descendant; trace metadata has the run's `report_mode`) |
| Unchanged resume reuses the Report with no model call | Live through the Postgres checkpointer: second invocation on the same thread recorded `reused / unchanged` for all three Steps and an identical Report. Offline: `tests/test_reporter.py::TestResume` (also changed Findings invalidate it, and a failed Reporter is retried, not reused) |

Full suite on this date: `pytest` (all markers) — 178 passed, 1 skipped.

## What was and was not exercised live

- Live: Azure OpenAI (`LLM_PROVIDER=azure`) for Planner, Analyst and Reporter; local Postgres (checkpointer plus seeded `sales_targets`, so the target comparison was `available` and reported by region and category); local Langfuse.
- Not exercised live: the FRED CPI path, because no `FRED_API_KEY` is configured here. In the live Report CPI therefore appears as a `not_configured` Caveat and Methodology line. The available-CPI path (figures, methodology, non-causality caveat) is covered offline with a stubbed `retrieve_cpi`.

## Known constraints

- LangGraph logs a "Deserializing unregistered type" warning when it loads `Report` and `EvidenceCatalog` from a Postgres checkpoint, exactly as it already does for `Plan` and `BaselineFindings`. It becomes an error only under `LANGGRAPH_STRICT_MSGPACK=true`; registering the state types is a follow-up for all four together.
- The model, not the code, decides which numbers sit next to which names in prose. Code guarantees every number is a Tool value and every Claim cites real entries; whether a Claim's wording is a fair reading of its cited entries is the later Critic Agent's job.
- Signed display values (`-269,534.66`) are substituted verbatim, so the Reporter prompt tells the model to write "changed by {E10}" rather than "fell by {E10}".
