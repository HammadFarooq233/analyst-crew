# Analyst Crew

A multi-agent data analysis copilot. See `CONTEXT.md` for the shared vocabulary (Crew, Agent, Plan, Step, Handoff, Findings, ...) and `docs/adr/0001-phase-1-real-tool-use-boundaries.md` for the accepted Phase 1 scope and policies.

## Setup

```bash
uv sync   # or: pip install -e .
cp .env.example .env   # set LLM_PROVIDER and its credentials (OPENROUTER_API_KEY, or the AZURE_OPENAI_* vars), LANGFUSE_*, POSTGRES_CONNECTION_STRING; FRED_API_KEY is optional
docker compose up -d   # starts Postgres (targets + checkpointer) and Langfuse
```

Without `FRED_API_KEY` set, the optional FRED CPI enrichment runs in its documented `not_configured` degraded mode — this is expected, valid behavior, not a failure (see issue #9 / ADR-0001).

## 1. Generate deterministic sales data and seed Postgres targets

One command each, both deterministic and idempotent:

```bash
# Generates data/sales.csv — seeded, so re-running reproduces byte-identical output.
python scripts/generate_sales_data.py

# Seeds the sales_targets table in Postgres from POSTGRES_CONNECTION_STRING.
python scripts/seed_targets.py
```

## 2. Run the complete investigation through the CLI

```bash
python cli.py \
  --question "Compare this month's sales to last month and flag anything unusual" \
  --thread-id demo-run-1 \
  --sales-csv data/sales.csv \
  --as-of-date 2025-06-30
```

This one command executes the capability-aware Plan through all six Agents (Planner, Fetcher, Analyst, Visualizer, Reporter and Critic) against the required CSV Data Source, with the optional Postgres sales-targets comparison and optional FRED CPI inflation adjustment enriching the Findings whenever those sources are configured and available. It prints a written, cited **Report** (`docs/adr/0002-reporter-evidence-catalog-and-claims.md`):

- **Summary** — a short headline answer, then **Findings** ordered by importance: Material Changes with their size and share of the change, the revenue bridge, target variance by region and category (when Postgres targets are available) and inflation-adjusted context (when FRED is available).
- **Charts** — a Visualizer Step between the Analyst and the Reporter has an LLM choose what to plot (a closed set of marks and Evidence Entry ids; code supplies every value and title from the catalog, one validation retry) and renders content-addressed PNG Chart Artifacts, marked `chart_mode: llm`. If the model fails after its bounded retries the deterministic fallback chart set (revenue bridge waterfall, top drivers bar, and target variance when targets are available) is used instead, marked `chart_mode: fallback` with the failure recorded as `chart_failure` on the run metadata (`docs/adr/0003-visualizer-selects-critic-judges-code-owns-numbers.md`). A Crew without the Visualizer keeps the section empty with a "no charts" notice.
- **Caveats** — always complete and never model-written: every Findings limitation, every optional source that was not configured or unavailable, the CPI non-causality statement when CPI is shown, and the source/included/quarantined/duplicate row counts.
- **Methodology** — the window, Material Change thresholds and their source, the bridge method, the Dataset Artifact and its integrity hash, optional sources, and row counts.
- **Reviewer Notes** — only when the Critic left issues standing. The Critic (no Tools; it reads the draft Report and the Evidence Catalog) returns a Critique: issues in a closed set of categories (causal overreach, misread evidence, overstated driver, unsupported emphasis), each aimed at an existing Claim or the Summary. A non-empty Critique triggers at most one Reporter revision, validated by the same rules as the first draft; the Critic then reviews the revision once, and whatever still stands is shown here. `critic_outcome` is `clean`, `revised`, `unresolved` or `failed`. A Critic failure never fails the run: the un-reviewed Report ships with the failure recorded (`critic_failure`). Caveats and Methodology are code-assembled and never touched by revision (`docs/adr/0003-visualizer-selects-critic-judges-code-owns-numbers.md`).
- **Evidence** — the Evidence Entries the Summary and Findings cite (`[E12]` markers), each with the Findings field it came from.

Every number comes from an analysis Tool: the Reporter writes claims with `{E12}`-style placeholders, code rejects any digit outside a placeholder or any unknown evidence id (one retry with the errors fed back), and the renderer substitutes each entry's display value. If the Reporter still fails, or the provider fails after bounded retries, the run records a terminal failure (on the state and the Langfuse trace, with `report_mode: fallback`) and prints a deterministic **Fallback Report**, visibly marked as generated without the Reporter Agent, plus a concise failure line; the exit code stays 0 because the analysis itself succeeded. A required-Step failure (Fetcher or Analyst) prints the terminal failure and exits 1, as before.

Add `--report-file report.md` to also save the rendered Report as Markdown, with the chart PNGs written beside it so the image links resolve (no file is written otherwise). The terminal output lists chart titles and paths.

Re-running the same command with the same `--thread-id` and unchanged inputs resumes the persisted CrewState from the Postgres checkpointer and reuses the prior Fetcher/Analyst Steps, Dataset Artifacts and the Report instead of recomputing them or calling the model again (issues #11 and #13). Changed Findings regenerate the Report.

## 3. Memory: remember, list, forget, confirm

The Crew keeps a persistent, per-user Memory in Postgres (`POSTGRES_CONNECTION_STRING` is required; without it the memory commands refuse rather than pretend to store). Memory changes framing and the parameters the Tools compute with, never which data is included. Terms are defined in `CONTEXT.md`; the design is `docs/adr/0005-deterministic-role-scoped-memory-and-crew-level-memory-update.md`.

```bash
# State a preference. The model proposes a typed preference or a free-text fact; code validates it
# against the closed vocabulary. It applies from the next run, in any thread.
python cli.py remember "for me, unusual means a swing over 8%" --user-id priya

python cli.py list --user-id priya              # what the Crew will apply
python cli.py list --user-id priya --history    # include superseded and forgotten memories
python cli.py forget mem-1a2b3c4d --user-id priya
python cli.py confirm mem-5e6f7a8b --user-id priya   # apply a pending inferred preference
python cli.py reject  mem-5e6f7a8b --user-id priya   # discard it
```

- **Typed preferences** (closed vocabulary, range-checked): the Material Change relative threshold and the absolute-impact threshold. The latest statement wins and the older one stays in the history. A sentence that does not fit (for example "ignore the Office category") is kept as a free-text fact and never becomes a parameter or an exclusion.
- **Free-text facts** shape how the Planner and Reporter frame a Report and never change a number. **Past-run summaries** are written after a successful run, go to the Planner, and expire when the source data changes.
- **Inference**: after a run the Crew may propose memories from your own words and its run records (`--no-infer` turns this off). An inferred fact applies immediately and is marked inferred. An inferred preference waits as pending until you `confirm` it; the run output lists the pending ids.
- **Applied Memories**: every Report a memory touched has an Applied Memories section (kind and origin, never the stored text), absent when nothing applied. A threshold in your question overrides a stored preference for that run and says so. If storage is down, the run continues with defaults and a Caveat says so. The applied memory ids are also on the Langfuse trace metadata (`applied_memory_ids`).
- Changing or forgetting a memory invalidates exactly the Findings, Report and Critique that depended on it when you re-run a thread; an unchanged set reuses them.

## Tests

```bash
# Offline suite — no network or live-model access required.
pytest -m "not llm and not integration and not eval"

# Marked live suites — require Docker services up and real credentials in .env.
pytest -m llm
pytest -m integration
pytest -m eval
```

The opt-in Phase 3 acceptance test is `tests/test_memory_live.py` (`-m "llm and integration"`); it creates and drops its own scratch database, so it leaves the development database untouched.

See `docs/phase-1-acceptance-status.md` (Findings), `docs/phase-2-reporter-acceptance-status.md` (Reporter), `docs/phase-2-acceptance-status.md` (orchestration) and `docs/phase-3-acceptance-status.md` (Memory) for the live-acceptance evidence and known constraints.
