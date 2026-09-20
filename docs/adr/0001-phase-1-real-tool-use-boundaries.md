# ADR-0001: Phase 1 real tool use boundaries

Status: Accepted

## Context

Phase 0 proved the Planner and the architectural layers around it. Phase 1 must demonstrate useful work rather than another minimal wiring exercise, while preserving the decision that arbitrary `execute_python` calls remain denied until human approval exists in Phase 6.

The Planner currently knows about every future Agent and always ends with a Reporter Step. That cannot produce a fully executable Plan while only the Fetcher and Analyst exist.

## Decision

Phase 1 will deliver one deep CSV sales investigation and smaller Postgres and FRED retrieval demonstrations.

The primary acceptance scenario compares sales performance across comparable monthly periods and identifies the products, categories, and regions that drove the change. It separates price and volume effects, accounts for discounts and returns, flags Material Changes, and reports data-quality limitations.

The Crew will execute a real Plan through an explicit Planner to Fetcher to Analyst flow. It will resolve Step dependencies into Handoffs and persist the resulting artifacts. The Planner will be capability-aware and may only assign Steps to Agents currently available to the Crew. In Phase 1, those downstream Agents are Fetcher and Analyst.

CSV Data Sources must be supplied explicitly. The Crew will profile the selected source and give its metadata to the Planner; the Planner must not invent sources. Sales CSVs use this canonical schema:

- `order_id`
- `order_date`
- `product_id`
- `product_name`
- `category`
- `region`
- `quantity`
- `unit_price`
- `discount_amount`
- `returned_quantity`

Derived revenue values are computed rather than accepted as duplicated source columns. Invalid or missing required fields produce clear validation errors.

When the latest month is incomplete, the default comparison is month-to-date against the equivalent number of days in the preceding month. Otherwise, the latest two complete calendar months are compared. Findings state the chosen window.

A Material Change defaults to both a change of at least 15% and an absolute revenue impact of at least 5% of the prior period's total revenue. Explicit user instructions may override these defaults.

The Analyst will use narrow, deterministic analysis Tools for Phase 1. General-purpose `execute_python` remains registered but denied until the human-in-the-loop safety phase.

DuckDB is the Phase 1 computation engine. Fetcher output is normalized into Parquet, and Analyst Tools execute application-owned SQL templates. Agents may select Tools and parameters through validated schemas, but may not supply executable SQL.

The existing Step contract will evolve to include `objective`, catalogued `source_ids`, `expected_output`, and `depends_on`. The Crew will construct a separate runtime Handoff containing the Step, resolved input artifact references, user question, and analysis parameters. Secrets and unresolved paths do not appear in either contract.

Fetched records will not be embedded in CrewState or checkpoints. The Fetcher produces a content-addressed Dataset Artifact containing its source identity, normalized file path, integrity hash, schema, row count, date range, quality summary, and creation time. The Analyst verifies the reference before using it.

Fetcher and Analyst are genuine LLM-driven Agents. The Fetcher chooses among its allowed CSV, Postgres, and FRED Tools; the Analyst chooses among narrow deterministic analysis Tools and returns validated structured Findings. Analytical results come from Tool output rather than model-authored arithmetic.

Testing is layered: offline deterministic tests cover Tools, policies, artifacts, Handoffs, calculations, and failure behavior; scripted-model tests cover Agent tool loops and graph transitions; marked live tests cover OpenRouter, Postgres, FRED, and the final acceptance run.

The primary demonstration uses deterministically generated synthetic data for a small multi-region home-and-office products retailer. It spans six months and roughly 25,000 to 50,000 orders, with controlled seasonal patterns, price and volume changes, discounts, returns, duplicates, invalid values, and known material drivers.

The canonical sales schema also includes `sales_channel` and `customer_segment`. Analysis covers product, category, region, channel, and customer-segment drivers.

Revenue is decomposed using gross revenue as quantity multiplied by unit price, return value as returned quantity multiplied by unit price, and net revenue as gross revenue less discount amount and return value. The bridge keeps volume, unit-price, interaction, discount, return, new-product, and discontinued-product effects separate.

Unreadable sources and missing required columns fail. Invalid individual records are quarantined, later duplicate order identifiers are excluded, and impossible commercial values are quarantined. Analysis stops when quarantined records exceed 1% of source rows. Findings disclose included, excluded, and duplicate counts and calculable revenue impact.

Postgres stores monthly targets at region and category grain with target net revenue and order count. FRED monthly CPI optionally supports an inflation-adjusted revenue comparison. CPI is contextual and must not be presented as causal evidence.

Phase 1 does not autonomously re-plan after failure. Transient Tool failures receive bounded retries, optional Data Sources degrade gracefully, required-source failures stop execution, and invalid Agent output may be retried once with validation feedback.

Checkpoint resume uses a Step execution key derived from the Step, analysis parameters, and input artifact hashes. Matching completed Steps are reused; changed sources or parameters invalidate downstream work; failed or interrupted Steps restart from their last safe boundary.

## Consequences

- Phase 1 tests the Plan and Handoff contracts through actual multi-Agent execution.
- The primary demonstration is realistic enough to exercise validation, aggregation, decomposition, anomaly detection, and evidence handling.
- Exact CSV columns keep correctness testable; flexible schema inference is deferred.
- The deterministic analysis boundary is auditable and does not weaken the established safety model.
- Reporter prose, visualization, critique, arbitrary analysis code, and remembered analytical preferences remain later-phase work.

## Amendment

ADR-0002 adds the Reporter as a third available Agent, so the Crew's default capability set is now Fetcher, Analyst and Reporter, and the sales Plan is Fetcher, Analyst, Reporter. The bounded-failure, Handoff and resume rules above apply to the Reporter Step unchanged; the Reporter's failure degrades to a Fallback Report instead of stopping the run.
