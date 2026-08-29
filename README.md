# Analyst Crew

A multi-agent data analysis copilot. See `CONTEXT.md` for the shared vocabulary (Crew, Agent, Plan, Step, Handoff, Findings, ...) and `docs/adr/0001-phase-1-real-tool-use-boundaries.md` for the accepted Phase 1 scope and policies.

## Setup

```bash
uv sync   # or: pip install -e .
cp .env.example .env   # fill in OPENROUTER_API_KEY, LANGFUSE_*, POSTGRES_CONNECTION_STRING; FRED_API_KEY is optional
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
  --as-of-date 2025-07-01
```

This one command executes the capability-aware Plan through Planner, Fetcher, and Analyst against the required CSV Data Source, with the optional Postgres sales-targets comparison and optional FRED CPI inflation adjustment enriching the Findings whenever those sources are configured and available. The printed report covers overall performance, all required dimensional drivers (product, category, region, channel, customer segment), Material Changes, the full revenue bridge, target variance (when Postgres targets are configured), inflation-adjusted context (when FRED is configured), data quality and provenance, and limitations.

Re-running the same command with the same `--thread-id` and unchanged inputs resumes the persisted CrewState from the Postgres checkpointer and reuses the prior Fetcher/Analyst Steps and Dataset Artifacts instead of recomputing them (issue #11).

## Tests

```bash
# Offline suite — no network or live-model access required.
pytest -m "not llm and not integration and not eval"

# Marked live suites — require Docker services up and real credentials in .env.
pytest -m llm
pytest -m integration
pytest -m eval
```

See `docs/phase-1-acceptance-status.md` for the current live-acceptance evidence and known upstream constraints.
