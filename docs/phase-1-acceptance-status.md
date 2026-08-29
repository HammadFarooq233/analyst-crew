# Phase 1 live acceptance status (issue #12)

Last evidence-gathering run: 2026-08-29, this session. **Phase 1 is NOT being marked complete.** Two real production bugs were found and fixed during this run, one live run of the full CLI scenario succeeded with real Findings, resume/reuse (#11) was verified for real, and a real gap in the Langfuse trace-linking acceptance criterion was found and is not fixed. See below for exact evidence.

## Environment used for this run

- Docker Desktop was not running at the start of the session; started successfully, then `docker compose up -d` brought up Postgres (`localhost:5434`) and self-hosted Langfuse v2 (`localhost:3000`) cleanly.
- `.env` already had real `OPENROUTER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `POSTGRES_CONNECTION_STRING`. `FRED_API_KEY` was not set and was not obtained — FRED enrichment ran in its documented `not_configured` degraded mode, which is valid, expected behavior (issue #9), and was exercised for real (see below).
- Outbound network to `openrouter.ai`, `api.stlouisfed.org` (redirects to `research.stlouisfed.org`), and Docker Hub's registry was reachable from this sandbox.

## 1. Offline suite — PASSES, no network/live-model access

```
.venv/bin/pytest -m "not llm and not integration and not eval" -q
...
FAILED tests/test_planner.py::TestStateShape::test_state_shape - assert None is not None
1 failed, 109 passed, 9 deselected, 3 warnings in 7.98s
```

`tests/test_planner.py::TestStateShape::test_state_shape` is not marked `llm`/`integration`, but it calls `invoke()` directly and, whenever `OPENROUTER_API_KEY` is set (as it always is here), it makes a real OpenRouter call. It fails because of the OpenRouter free-tier constraints documented below (`plan_error` = `"OpenRouter free-model quota or rate limit exceeded..."`), not because of a code defect and not because Postgres was down (Postgres was up for every run in this session and the failure persisted regardless). This is the one pre-existing failure flagged in the ticket; it is a mis-marked test (should carry `@pytest.mark.llm`) rather than a Postgres dependency — worth a follow-up ticket to add the marker, out of scope here.

## 2. Two real production bugs found and fixed

Before any fix, every live CLI/`invoke()` attempt that reached the Fetcher or Analyst node failed identically:

```
ValueError: Invalid input type <class 'dict'>. Must be a PromptValue, str, or list of BaseMessages.
```

Root cause: `FetcherAgent.run()` (`analyst_crew/tools/sales.py`) and `AnalystAgent.run()` (`analyst_crew/tools/analysis.py`) both call `self.model.invoke({...context dict...})`, but `production_fetcher_model()` / `production_analyst_model()` returned a plain `ChatOpenAI(...).with_structured_output(...)` runnable, which only accepts a `PromptValue`, `str`, or list of `BaseMessage` — never a raw `dict`. This path is only reachable with a real model (offline tests substitute scripted fake models with a permissive `.invoke(dict)` signature), so it had evidently never been exercised against a live model before this ticket. **This is exactly the kind of bug issue #12 exists to catch.**

Fix: both factory functions now wrap the structured-output call behind a `RunnableLambda` that formats the context dict into a single prompt string before invoking the model — no behavior change for the offline scripted-model tests (which pass a fake `model` directly to `FetcherAgent`/`AnalystAgent` and never touch these factories), and the full offline suite still passes 109/110 (same pre-existing failure as above).

## 3. OpenRouter free-tier availability — clearly evidenced blocker

The configured default model, `google/gemma-4-26b-a4b-it:free`, is rate-limited upstream regardless of account quota:

```
$ curl -s https://openrouter.ai/api/v1/chat/completions \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" \
    -d '{"model":"google/gemma-4-26b-a4b-it:free","messages":[{"role":"user","content":"hi"}]}'
{"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits...","provider_error_code":"429","limit_source":"upstream_provider_shared_pool", ...}}}
```

`config/settings.toml`'s `[llm].model` was switched to `nvidia/nemotron-3-super-120b-a12b:free` — a free OpenRouter model that was reachable and did support `json_mode` structured output — as the app's own error message explicitly suggests ("select another free model in `config/settings.toml`"). Several other free models were tried and rejected because they don't reliably honor the Planner's structured JSON contract (leak chain-of-thought into the content field, invent extra top-level fields, drop the required `user_question` field, or attempt an unbound `search_memory` tool call as free text instead of JSON) — this is a genuine quality/compliance limitation of free-tier models, not a fabricated result. `nvidia/nemotron-3-super-120b-a12b:free` itself was flaky: across roughly 15 live Planner calls in this session it produced a schema-valid `Plan` on about 1 in 4–6 attempts, failing the rest of the time with `OutputParserException` on malformed JSON.

Eventually, further live attempts hit OpenRouter's account-wide daily free-tier cap, independent of which free model is selected:

```
$ curl -s https://openrouter.ai/api/v1/chat/completions ... -d '{"model":"nvidia/nemotron-3-super-120b-a12b:free", ...}'
{"error":{"message":"Rate limit exceeded: free-models-per-day. Add 5 credits to unlock 1000 free model requests per day","code":429,"metadata":{"headers":{"X-RateLimit-Limit":"50","X-RateLimit-Remaining":"0","X-RateLimit-Reset":"1788048000000"},"limit_source":"openrouter_free_tier_daily", ...}}}
```

`X-RateLimit-Reset: 1788048000000` = 2026-08-30T00:00:00Z. This account is capped at 50 free-model requests/day without adding credits. All `test_memory.py`, `test_tracing.py`, `test_eval.py`, and the unmarked `test_planner.py::TestStateShape::test_state_shape` failures/skips from this point forward in the session carry this exact evidenced cause, e.g.:

```
Skipped: OpenRouter free-model limit exceeded: OpenRouter free-model quota or
rate limit exceeded. Try again later or select another free model in
config/settings.toml.
```

## 4. One full live CLI run succeeded — real Findings

Before the daily cap was hit, one full documented CLI invocation completed end to end:

```
python cli.py --question "Compare this month's sales to last month and flag anything unusual" \
  --thread-id demo-run-2 --sales-csv data/sales.csv --as-of-date 2025-06-30
```

Real output (verbatim):

```
Baseline sales Findings (complete_month): 2025-06-01 to 2025-06-30 vs 2025-05-01 to 2025-05-31. Net revenue 3,942,300.70 vs 4,211,835.36 (-269,534.66, -6.4%); gross revenue 4,122,056.40 vs 4,398,414.40 (-276,358.00, -6.3%); orders 4,970 vs 5,176 (-206, -4.0%); units 22,159 vs 23,440 (-1,281, -5.5%). Evidence: 30002 source rows, 30000 included, 2 quarantined, 1 duplicates excluded; excluded gross revenue impact 0.00; reasons {'duplicate_order_id': 1, 'malformed_row': 1}.
Material Changes (>= 15% relative and >= 5% of prior-period net revenue, thresholds: default):
  - customer_segment=enterprise: 1,226,738.54 vs 1,477,921.40 (-251,182.86, -17.0%, contribution +93.2% of net change)
Revenue bridge (reconciles to net revenue change -269,534.66, gap 0.0000):
  - volume: -276,358.00
  - price: +0.00
  - price-volume interaction: -0.00
  - discount: +10,864.14
  - return: -4,040.80
  - new products (0): +0.00
  - discontinued products (0): +0.00
  - total: -269,534.66 (continuing products: 6)
Sales targets (actual vs. target, by region and category):
  - east/furniture: net revenue 745,412.00 vs target 739,039.42 (+6,372.58, +0.9%); orders 413 vs target 432 (-19, -4.4%)
  ... [all 12 region/category rows] ...
CPI inflation context: not_configured (FRED_API_KEY is not set). Nominal net revenue change -269,534.66.
```

Inspecting `artifacts["findings"]` (the underlying `BaselineFindings` model, not just the CLI's printed summary) directly confirmed:
- `dimensions` covers all five required breakdowns: `product`, `category`, `region`, `sales_channel`, `customer_segment`.
- `limitations` correctly records `"FRED CPI inflation adjustment not_configured: FRED_API_KEY is not set"`.
- `target_comparison` (Postgres) is populated and `available`; `cpi_adjustment` (FRED) is `not_configured` — required-source (CSV) vs. optional-source (Postgres/FRED) behavior matches the accepted policy: the required CSV path ran to completion, the optional Postgres source enriched the run, and the optional FRED source degraded gracefully with a recorded status and matching limitation rather than failing the run.

## 5. Resume/reuse (#11) verified for real

Re-invoking the same `thread_id` (`demo-run-2`) with unchanged inputs against the real `PostgresSaver` checkpointer produced, from the returned `CrewState` (not from re-reading the source):

```
step_outcomes: {
  'fetcher_sales_monthly': {..., 'resume_outcome': 'reused', 'resume_reason': 'unchanged', ...},
  'analyst_compare_flag':  {..., 'resume_outcome': 'reused', 'resume_reason': 'unchanged', ...},
}
```

Both the Fetcher and Analyst Steps were marked `reused`/`unchanged` rather than recomputed, with no new OpenRouter calls needed for either Step (the graph only re-invokes the Planner; both downstream Steps came straight from the checkpoint).

## 6. Marked test suites

```
pytest -m integration   # fresh Postgres/Langfuse containers
3 passed, 4 skipped, 112 deselected
  PASS  test_checkpointer.py::TestStatePersistence::test_state_persistence
  PASS  test_checkpointer.py::TestResumeAgainstPostgres::test_unchanged_resume_reuses_via_postgres_checkpointer
  PASS  test_targets.py::test_seed_and_retrieve_targets_against_live_postgres
  SKIP  test_fred.py::test_retrieve_cpi_against_live_fred_api        (FRED_API_KEY not set — documented)
  SKIP  test_memory.py (x2)                                          (OpenRouter free-tier daily cap — evidenced above)
  SKIP  test_tracing.py::TestTracing::test_trace_exists               (OpenRouter free-tier daily cap — evidenced above)

pytest -m llm
1 skipped — same OpenRouter free-tier daily cap.

pytest -m eval
1 failed — same OpenRouter free-tier daily cap (RuntimeError raised directly, not skipped, because this call path doesn't catch it into a skip).
```

Note: re-running `pytest -m integration` a second time against the **same** Postgres volume without resetting it first spuriously fails `TestResumeAgainstPostgres` (its fixed `thread_id = "test-resume-postgres-thread-001"` is never cleaned up, so a second run sees leftover state from the first and reports `reused` where it expects `recomputed` on the "first" call). This is a pre-existing test-isolation gap, not a product bug — confirmed by re-running against a freshly recreated `docker compose down -v && up -d` Postgres, where the test passes cleanly. Worth a follow-up ticket (e.g. a per-run unique thread suffix or an explicit teardown) so `pytest -m integration` is safe to run repeatedly against a persistent dev Postgres.

## 7. Langfuse tracing — gap found, NOT met

`analyst_crew/agents/planner.py::get_langfuse_handler()` is a hard-coded no-op (`return None`) with this exact comment in the source:

> "The bundled self-hosted deployment is Langfuse v2. Its ingestion API is compatible with the v2 SDK, but not with the OTLP exporter introduced by newer SDKs. The trace is recorded after the graph run by invoke()."

`_create_langfuse_trace()` creates exactly one flat trace per `invoke()` call, tagged `agent:Planner`, carrying `session_id`/`user_id` and a `metadata.retry_counts` dict (and `terminal_failure` when present) — confirmed live via the Langfuse API:

```
$ curl -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
    "http://localhost:3000/api/public/traces?sessionId=demo-run-2&limit=20"
# 10 trace records, name="planner", tags=["agent:Planner"], sessionId="demo-run-2",
# metadata.retry_counts populated per Step, "observations": [] on every one.
```

`observations` is empty on every trace — there are **no nested spans for the individual Planner/Fetcher/Analyst LLM calls, Handoffs, or Tool invocations**. The trace links a session/thread to retry counts and terminal-failure status, but it does not link Planner, Fetcher, Analyst, Handoffs, and Tools together as the acceptance criterion requires. `tests/test_tracing.py::test_trace_exists` only asserts a trace with the right `session_id`/tag exists — it does not check for per-Agent/Tool observations, so the existing test suite does not actually enforce this criterion either.

**This acceptance criterion — "One Langfuse trace links Planner, Fetcher, Analyst, Handoffs, Tools, retries, source status, and failures" — is not met.** Wiring real per-node Langfuse spans (e.g. via `@observe` decorators or explicit span creation in each graph node, migrating off the disabled callback handler) is a real, non-trivial follow-up, explicitly out of scope for this proof-of-work session; it needs its own ticket.

## Why Phase 1 status is not being marked complete

1. The Langfuse trace-linking criterion is a genuine, currently-unmet product gap (not an infrastructure/availability issue) — see §7.
2. Live acceptance evidence for the LLM/eval/tracing marked suites is partial: one full live CLI run succeeded and is documented above with real output, and the OpenRouter free-tier blocker is clearly evidenced, but that is not the same as "these suites pass."

## What's needed to close this out

1. Wire real per-node Langfuse spans for Planner/Fetcher/Analyst/Handoffs/Tools (see §7) so one trace actually links them.
2. Either fund an OpenRouter account (add credits to raise the free daily cap) or get a more reliable/paid model configured for CI-grade live runs, since the free tier is both rate-limited and inconsistently schema-compliant.
3. Re-run `pytest -m llm`, `pytest -m integration`, `pytest -m eval` end to end once the above are addressed, and re-run the full CLI scenario + resume to confirm the trace now shows the full Planner→Fetcher→Analyst→Handoff→Tool chain.
4. Only then update this file (or wherever Phase 1 roadmap status ultimately lives) to "complete."
