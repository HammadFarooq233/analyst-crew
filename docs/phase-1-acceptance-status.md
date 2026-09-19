# Phase 1 live acceptance status (issue #12)

## Update 2026-09-19 — live acceptance PASSED on Azure OpenAI (`gpt-5.4-mini`)

The OpenRouter free-tier blockers below were bypassed by adding a provider switch (`LLM_PROVIDER=azure|openrouter`, `analyst_crew/llm.py`) and running against an Azure OpenAI / Microsoft Foundry deployment. Evidence from this run:

- Offline suite: 113 passed. Marked suites: `llm` 2/2, `integration` 7 passed + 1 skipped (`FRED_API_KEY` unset — optional source, documented), `eval` 1/1.
- README CLI scenario (`--as-of-date 2025-06-30`) produced real Findings identical to the earlier verified run (net revenue 3,942,300.70 vs 4,211,835.36, bridge gap 0.0000).
- Langfuse: one trace per run with real `AzureChatOpenAI` (`gpt-5.4-mini`) GENERATION observations nested under `planner_node` / `fetcher_node` / `analyst_node`.
- Resume: second run on the same thread made **0 LLM calls** and reported both Steps `reused`/`unchanged`.

Two real bugs found and fixed while proving this:
1. `json_mode` on OpenAI-family endpoints requires the word "JSON" in the prompt; the Planner prompt now states it (rule 12).
2. **Resume never actually reused work with a live Planner.** Each invoke re-planned, and the LLM worded `step_id`/`objective` differently, changing every execution key. `planner_node` now reuses the checkpointed Plan when the question and capabilities are unchanged (`_reusable_prior_plan`, with regression tests). The earlier "reused" evidence only held because fakes/older models were stable.

Remaining: `FRED_API_KEY` is still unset (live FRED test skipped). README's old `--as-of-date 2025-07-01` example produced a degenerate one-day comparison because the data ends 2025-06-30; README now uses 2025-06-30. The history below is retained for context.

---

Last evidence-gathering run: 2026-08-29, this session (two passes). **Phase 1 is still NOT being marked complete**, but the Langfuse trace-linking gap flagged in the first pass has now been fixed and verified live against the real Langfuse server — see §7 (updated). Three real production bugs were found and fixed across both passes, one live run of the full CLI scenario succeeded with real Findings, resume/reuse (#11) was verified for real, and the Langfuse trace-linking mechanism was verified live through the real production graph/node code — but not yet with a genuine OpenRouter-backed run, because OpenRouter's free-tier daily quota was exhausted before the fix landed. See below for exact evidence and what's left for a human with fresh quota to confirm.

## Environment used for this run

- Docker Desktop was not running at the start of the session; started successfully, then `docker compose up -d` brought up Postgres (`localhost:5434`) and self-hosted Langfuse v2 (`localhost:3000`) cleanly.
- `.env` already had real `OPENROUTER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `POSTGRES_CONNECTION_STRING`. `FRED_API_KEY` was not set and was not obtained — FRED enrichment ran in its documented `not_configured` degraded mode, which is valid, expected behavior (issue #9), and was exercised for real (see below).
- Outbound network to `openrouter.ai`, `api.stlouisfed.org` (redirects to `research.stlouisfed.org`), and Docker Hub's registry was reachable from this sandbox.
- Mid-session, an earlier `docker compose down -v` (run to get a clean Postgres for isolated test-checkpointer debugging) also wiped Langfuse's own database, orphaning `.env`'s `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` against the fresh, empty Langfuse project (confirmed: `curl .../api/public/traces` → `"Invalid credentials"`). Fixed properly rather than just re-copying new keys into `.env`: `docker-compose.yml`'s `langfuse` service now sets Langfuse's own `LANGFUSE_INIT_*` bootstrap env vars (org/project/user, with the project's API key pair read from `.env` via compose's automatic `.env` substitution), so any future `docker compose down -v && up -d` re-provisions a project whose keys match `.env` automatically — verified live (see §7).

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

## 7. Langfuse tracing — FIXED and verified live (second pass)

### What was wrong

`get_langfuse_handler()` was a hard-coded no-op (`return None`), and `_create_langfuse_trace()` created one flat, disconnected trace per `invoke()` call after the graph already finished — confirmed live in the first pass via the Langfuse API (10 trace records, `name="planner"`, `"observations": []` on every one). No nested spans for Planner/Fetcher/Analyst LLM calls, Handoffs, or Tool invocations.

### Root cause of why it was stubbed out

The pinned Langfuse v2 SDK's LangChain integration (`langfuse.callback.CallbackHandler`) imports `langchain.callbacks.base.BaseCallbackHandler`, `langchain.schema.agent.{AgentAction,AgentFinish}`, and `langchain.schema.document.Document` — legacy import paths. This project's installed `langchain` is 1.3.14, which no longer exposes any of them at those paths (verified: `python -c "import langchain.callbacks.base"` → `ModuleNotFoundError: No module named 'langchain.callbacks'`). Those exact symbols do still exist, just only in `langchain_core` (already a direct dependency here), which `langfuse.callback` doesn't know to fall back to.

### The fix

- `analyst_crew/agents/planner.py::_ensure_langchain_callback_compat()` registers thin `sys.modules` shims for the missing legacy `langchain.callbacks.base` / `langchain.schema.agent` / `langchain.schema.document` paths, pointing them at their `langchain_core` equivalents, before `langfuse.callback` is imported. Verified directly: `from langfuse.callback import CallbackHandler` succeeds after the shim runs, and fails without it.
- `get_langfuse_handler()` now creates one real Langfuse v2 trace up front (`client.trace(name="crew_run", session_id=thread_id, tags=["agent:Planner"], ...)`) and a `CallbackHandler(stateful_client=trace)` bound to it, returned as `(client, trace, handler)`.
- `invoke()` threads that one `handler` through `config["callbacks"]` for the whole `graph.invoke()` call, exactly as before — but now the handler actually exists.
- `planner_node`/`fetcher_node`/`analyst_node` now pass their `config` (which carries `callbacks`) into `PlannerAgent.generate_plan(..., config=config)`, `FetcherAgent.run(..., config=config)`, `AnalystAgent.run(..., config=config)`, and the two production Tool-calling seams (`production_fetcher_model`/`production_analyst_model`, already reworked in the first pass) forward it to the underlying structured-output `.invoke(prompt, config=config)` call.
- A new `analyst_crew/tools/retry.py::invoke_with_optional_config(model, payload, config)` helper forwards `config` only when the target's `invoke` signature actually declares a `config` parameter (checked via `inspect.signature`) — real LangChain Runnables always do; the scripted fake models used throughout the offline test suite (`invoke(self, request)`, no `config`) don't, so they're untouched and the full offline suite still passes unchanged.
- `_finish_langfuse_trace()` (replacing `_create_langfuse_trace()`) updates that same trace's metadata after the graph run with retry counts, terminal-failure outcome, per-Step `resume_outcome`/`resume_reason`, and optional-source status (`target_comparison.source_status` / `cpi_adjustment.source_status` from the Findings) — see `_run_metadata()`.
- `docker-compose.yml`'s `langfuse` service now sets `LANGFUSE_INIT_*` env vars (org/project/user, project API keys pulled from `.env`'s own `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` via compose's automatic `.env` substitution) so a fresh `docker compose down -v && up -d` auto-provisions a project whose keys match `.env`, instead of leaving `.env`'s keys orphaned against an empty Langfuse instance (this happened to us mid-session from an earlier `down -v` and had to be fixed to keep testing).

### Live verification — mechanism confirmed real, but with local LLM stand-ins, not OpenRouter

OpenRouter's free-tier daily quota (50 req/day) was already exhausted before this fix was written (see §3; reset 2026-08-30T00:00:00Z), so the fix could not be verified with a genuine OpenRouter-backed Planner/Fetcher/Analyst run in this session. What **was** verified live, running the actual production `analyst_crew.agents.invoke()` → `build_graph()` → `planner_node`/`fetcher_node`/`analyst_node` code (real Postgres checkpointer, real CSV Fetcher Tool against `data/sales.csv`, real DuckDB Analyst Tool, real Postgres targets query, real Langfuse server) with only the three LLM calls substituted for a local, network-free `langchain_core.language_models.fake_chat_models.FakeListChatModel` (a genuine `BaseChatModel` that fires the same LangChain callback events a real `ChatOpenAI` call does — `planner_model`/`fetcher_model`/`analyst_model` configurable overrides, the same seam the offline test suite already uses):

```
$ curl -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
    "http://localhost:3000/api/public/traces/2d0834c3-66c5-4045-8656-15767fae6ad2"
{
  "name": "crew_run", "tags": ["agent:Planner"], "sessionId": "trace-wiring-check-795ca4fb",
  "metadata": {
    "retry_counts": {"fetch_sales": {...}, "analyze_sales": {...}},
    "source_status": {"cpi_fred": {"status": "not_configured", "detail": "FRED_API_KEY is not set"},
                       "sales_targets_postgres": {"status": "available", "detail": null}},
    "step_outcomes": {"fetch_sales": {"status": "completed", "resume_outcome": "recomputed", ...},
                       "analyze_sales": {"status": "completed", "resume_outcome": "recomputed", ...}}
  },
  "observations": [ ... 9 observations ... ]
}
```

The 9 observations on that one trace, with real parent/child links:

```
SPAN       name=LangGraph            parent=None                  <- the graph's own top-level run
SPAN       name=planner_node         parent=LangGraph
GENERATION name=FakeListChatModel    parent=planner_node
SPAN       name=next_node            parent=planner_node          <- LangGraph's own conditional-edge fn
SPAN       name=fetcher_node         parent=LangGraph
GENERATION name=FakeListChatModel    parent=fetcher_node
SPAN       name=after_fetcher        parent=fetcher_node
SPAN       name=analyst_node         parent=LangGraph
GENERATION name=FakeListChatModel    parent=analyst_node
```

This run also produced real Findings (`findings present: True`, `terminal_failure: None`) from real CSV/DuckDB/Postgres work, in the same `invoke()` call. A second run with a deliberately unreadable CSV path confirmed `terminal_failure` also lands correctly on the same trace's metadata:

```
$ curl -u "..." "http://localhost:3000/api/public/traces?sessionId=trace-failure-check-26b26403"
"metadata": {"retry_counts": {...}, "terminal_failure": {"agent": "Fetcher", "step_id": "fetch_sales",
  "status": "failed", "error": "CSV Data Source is unreadable: /nonexistent/sales.csv", ...}}
```

### What this proves and what it doesn't

**Proven, live, against the real self-hosted Langfuse server and the real production graph/node code:** one trace per Crew run; nested Planner/Fetcher/Analyst spans with nested LLM-generation observations under them; retry counts, per-Step resume outcomes, optional-source status, and terminal-failure all attached to that same trace's metadata; the compat shim actually makes the pinned Langfuse v2 SDK importable against this project's langchain 1.x.

**Not yet proven:** a genuine OpenRouter-backed live run through this exact new tracing code — the CLI run in §4 predates this fix and used the old flat/unlinked tracing. Also, Tool execution itself (`fetch_csv`, `run_sales_baseline`) is plain Python inside the `fetcher_node`/`analyst_node` spans, not a separately-named Tool-type observation — it's visible via that span's duration/input/output, not as its own node in the trace tree. `tests/test_tracing.py::test_trace_exists` still could not be run live this session (it calls `invoke()` for real, which needs a live Planner call — quota-blocked, evidenced skip below).

## Why Phase 1 status is still not being marked complete

1. The Langfuse trace-linking gap is fixed and verified live against the real production code, but only with local LLM stand-ins — not yet with a genuine OpenRouter-backed run, since quota ran out first. That is a real, reasonable, but real gap between "the mechanism is proven correct" and "the full live acceptance scenario has been observed end to end with this fix in place."
2. Live acceptance evidence for the LLM/eval/tracing marked suites is still partial for the same reason (§3, §6): one full live CLI run succeeded pre-fix, and the OpenRouter free-tier blocker is clearly evidenced, but `pytest -m llm`/`eval`/`test_tracing.py` still can't run live today.

## What's needed to close this out

1. Once OpenRouter quota resets (2026-08-30T00:00:00Z) or a paid/BYOK key is available, re-run the documented CLI command (README.md) once, then re-run it a second time with the same `--thread-id` to also re-confirm resume/reuse under the new tracing code.
2. Check the resulting trace via the Langfuse UI (`http://localhost:3000`) or API (`GET /api/public/traces?sessionId=<thread_id>`) and confirm it shows the same `planner_node`/`fetcher_node`/`analyst_node` span structure seen above, but with real `ChatOpenAI` `GENERATION` observations (model name, token usage, real prompts/completions) instead of `FakeListChatModel` ones.
3. Re-run `pytest -m llm`, `pytest -m integration`, `pytest -m eval` end to end once quota is available — expect `test_tracing.py::test_trace_exists` to now pass for real (it only checks tag/session_id, which this fix already satisfies structurally).
4. Only then update this file (or wherever Phase 1 roadmap status ultimately lives) to "complete."
