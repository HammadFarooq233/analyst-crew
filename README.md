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

This one command executes the capability-aware Plan through Planner, Fetcher, Analyst, and Reporter against the required CSV Data Source, with the optional Postgres sales-targets comparison and optional FRED CPI inflation adjustment enriching the Findings whenever those sources are configured and available. It prints a written, cited **Report** (`docs/adr/0002-reporter-evidence-catalog-and-claims.md`):

- **Summary** — a short headline answer, then **Findings** ordered by importance: Material Changes with their size and share of the change, the revenue bridge, target variance by region and category (when Postgres targets are available) and inflation-adjusted context (when FRED is available).
- **Charts** — a Visualizer Step between the Analyst and the Reporter has an LLM choose what to plot (a closed set of marks and Evidence Entry ids; code supplies every value and title from the catalog, one validation retry) and renders content-addressed PNG Chart Artifacts, marked `chart_mode: llm`. If the model fails after its bounded retries the deterministic fallback chart set (revenue bridge waterfall, top drivers bar, and target variance when targets are available) is used instead, marked `chart_mode: fallback` with the failure recorded as `chart_failure` on the run metadata (`docs/adr/0003-visualizer-selects-critic-judges-code-owns-numbers.md`). A Crew without the Visualizer keeps the reserved, empty section.
- **Caveats** — always complete and never model-written: every Findings limitation, every optional source that was not configured or unavailable, the CPI non-causality statement when CPI is shown, and the source/included/quarantined/duplicate row counts.
- **Methodology** — the window, Material Change thresholds and their source, the bridge method, the Dataset Artifact and its integrity hash, optional sources, and row counts.
- **Evidence** — the Evidence Entries the Summary and Findings cite (`[E12]` markers), each with the Findings field it came from.

Every number comes from an analysis Tool: the Reporter writes claims with `{E12}`-style placeholders, code rejects any digit outside a placeholder or any unknown evidence id (one retry with the errors fed back), and the renderer substitutes each entry's display value. If the Reporter still fails, or the provider fails after bounded retries, the run records a terminal failure (on the state and the Langfuse trace, with `report_mode: fallback`) and prints a deterministic **Fallback Report**, visibly marked as generated without the Reporter Agent, plus a concise failure line; the exit code stays 0 because the analysis itself succeeded. A required-Step failure (Fetcher or Analyst) prints the terminal failure and exits 1, as before.

Add `--report-file report.md` to also save the rendered Report as Markdown, with the chart PNGs written beside it so the image links resolve (no file is written otherwise). The terminal output lists chart titles and paths.

Re-running the same command with the same `--thread-id` and unchanged inputs resumes the persisted CrewState from the Postgres checkpointer and reuses the prior Fetcher/Analyst Steps, Dataset Artifacts and the Report instead of recomputing them or calling the model again (issues #11 and #13). Changed Findings regenerate the Report.

## Tests

```bash
# Offline suite — no network or live-model access required.
pytest -m "not llm and not integration and not eval"

# Marked live suites — require Docker services up and real credentials in .env.
pytest -m llm
pytest -m integration
pytest -m eval
```

See `docs/phase-1-acceptance-status.md` (Findings) and `docs/phase-2-reporter-acceptance-status.md` (Reporter) for the current live-acceptance evidence and known upstream constraints.
