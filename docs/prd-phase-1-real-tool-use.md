# Phase 1 — Real Tool Use

## Problem Statement

Priya can currently ask the Crew a question and receive a structured Plan, but the Crew cannot execute that Plan or analyze real data. The Phase 0 walking skeleton proves the Planner, state, Memory, checkpointing, tracing, and evaluation layers only at minimum depth. Its Plan is descriptive rather than load-bearing, and no Fetcher or Analyst performs useful work.

The next increment must be meaningfully practical rather than another toy integration. Priya needs the Crew to investigate realistic retail sales, show what changed and why, distinguish commercial drivers, handle imperfect data honestly, enrich the analysis with internal targets and external economic data, and preserve evidence for later visualization, critique, and reporting. The implementation must also respect the existing safety decision that arbitrary code execution remains denied until human approval exists.

## Solution

Add genuine LLM-driven Fetcher and Analyst Agents and execute a capability-aware Plan through real Handoffs. The primary demonstration analyzes six months of deterministic synthetic order data for a multi-region home-and-office products retailer. It compares comparable monthly periods and explains net-revenue changes across products, categories, regions, sales channels, and customer segments. It separates price, volume, interaction, discount, return, new-product, and discontinued-product effects; flags Material Changes; and reports data-quality limitations.

The explicitly supplied sales CSV is required. Seeded Postgres sales targets and FRED CPI are optional enrichments. The Fetcher validates and normalizes allowed Data Sources into content-addressed Dataset Artifacts. The Analyst selects narrow deterministic DuckDB Tools and returns structured Findings with a concise CLI interpretation. Application-owned queries perform every calculation; models select Tools and interpret validated results but never invent numbers or executable SQL.

The Crew persists Plan execution, Handoffs, artifacts, outcomes, and diagnostics. It resumes compatible completed Steps, invalidates stale downstream work when sources or parameters change, degrades gracefully when optional Data Sources fail, and stops explicitly when required evidence is invalid.

## User Stories

1. As Priya, I want to supply a sales CSV explicitly, so that the Crew analyzes the intended file rather than guessing.
2. As Priya, I want to ask a natural-language business question, so that I do not need to write analysis code or SQL.
3. As Priya, I want the Crew to compare comparable monthly periods, so that a partial current month is not misleadingly compared with a full prior month.
4. As Priya, I want the comparison window stated explicitly, so that I understand the basis of every result.
5. As Priya, I want net and gross revenue calculated consistently, so that discounts and returns are not hidden.
6. As Priya, I want overall revenue and order changes, so that I can understand headline performance.
7. As Priya, I want product and category drivers, so that I can identify which merchandise changed performance.
8. As Priya, I want regional drivers, so that I can distinguish local performance differences.
9. As Priya, I want sales-channel drivers, so that I can see whether web, marketplace, or store mix changed.
10. As Priya, I want customer-segment drivers, so that I can see whether consumer, small-business, or enterprise demand changed.
11. As Priya, I want price and volume effects separated, so that I know whether revenue changed because of demand or pricing.
12. As Priya, I want price-volume interaction reported separately, so that attribution is mathematically honest.
13. As Priya, I want discount and return effects isolated, so that apparent growth is not confused with lower-quality sales.
14. As Priya, I want new and discontinued products handled separately, so that missing baselines do not create false comparisons.
15. As Priya, I want Material Changes to require both percentage and absolute impact, so that tiny categories do not dominate alerts.
16. As Priya, I want to override materiality thresholds in my question, so that the analysis can fit a specific decision.
17. As Priya, I want actual sales compared with internal targets, so that performance is evaluated against business expectations as well as history.
18. As Priya, I want an optional inflation-adjusted comparison, so that I can distinguish nominal growth from an approximation of real growth.
19. As Priya, I want economic context labeled as non-causal, so that contextual correlation is not presented as an explanation.
20. As Priya, I want isolated malformed records quarantined, so that one bad row does not discard an otherwise useful dataset.
21. As Priya, I want analysis stopped when data damage is material, so that graceful degradation does not produce untrustworthy conclusions.
22. As Priya, I want duplicate orders identified and excluded deterministically, so that revenue is not double-counted.
23. As Priya, I want included and excluded row counts and revenue impact disclosed, so that I can judge data quality.
24. As Priya, I want optional-source failures disclosed without losing the core CSV analysis, so that transient enrichment failures do not erase useful work.
25. As Priya, I want the Crew never to fill missing evidence from model knowledge, so that every number remains traceable to a Data Source.
26. As a developer, I want the Planner constrained to currently available Agents, so that every Step in a Plan is executable.
27. As a developer, I want Steps to identify objectives, catalogued sources, expected outputs, and dependencies, so that Plans are operational contracts rather than prose.
28. As a developer, I want the Crew to resolve dependencies into explicit Handoffs, so that each Agent receives only declared inputs.
29. As a developer, I want credentials and unresolved paths excluded from Plans and Handoffs, so that execution contracts do not leak secrets.
30. As a developer, I want Fetcher and Analyst to be genuine tool-using Agents, so that Phase 1 exercises multi-Agent tool selection.
31. As a developer, I want deterministic Tools to calculate metrics, so that LLM nondeterminism cannot change arithmetic truth.
32. As a developer, I want arbitrary Python and model-generated SQL denied, so that Phase 1 preserves the established safety boundary.
33. As a developer, I want fetched data stored as Dataset Artifacts rather than checkpoint payloads, so that realistic datasets do not bloat CrewState.
34. As a developer, I want artifact integrity hashes and provenance, so that downstream analysis can verify exactly what it consumed.
35. As a developer, I want Plan and Step outcomes checkpointed, so that interrupted work can resume.
36. As a developer, I want unchanged completed Steps reused on resume, so that external calls and computation are not repeated unnecessarily.
37. As a developer, I want changed source content or parameters to invalidate downstream work, so that stale Findings are never reused.
38. As a developer, I want bounded retries and explicit terminal failures, so that Agent execution cannot loop indefinitely.
39. As a developer, I want the entire run visible in one Langfuse trace, so that Plans, Handoffs, Tools, retries, and failures can be inspected together.
40. As a developer, I want a deterministic synthetic dataset with known drivers and defects, so that analytical correctness can be asserted without private data.
41. As a developer, I want offline correctness tests, so that ordinary development is not blocked by free-model quotas or network access.
42. As a developer, I want scripted-model Agent contract tests, so that tool loops and graph transitions can be tested deterministically.
43. As a developer, I want marked live acceptance tests, so that real OpenRouter, Postgres, FRED, and tracing integrations are still proven.
44. As a future Visualizer, Critic, or Reporter, I want structured Findings with provenance, so that later Agents can consume stable evidence rather than scrape console prose.

## Implementation Decisions

- The Crew will execute a real Planner → Fetcher → Analyst flow. It will not use a temporary hardcoded route.
- The Planner is capability-aware. Phase 1 exposes only Fetcher and Analyst as assignable downstream Agents; later phases expand the capability set.
- Step evolves from a descriptive shape to an executable contract containing a unique identifier, Agent, objective, catalogued Data Source identifiers, expected output, and prior-Step dependencies.
- Handoff is a separate runtime contract containing the Step, resolved input artifact references, the original user question, and explicit analysis parameters.
- The Crew resolves dependencies and Data Source identifiers. Agents do not receive secrets and may not invent files, tables, or FRED series.
- The CLI requires an explicit sales CSV. Postgres targets and FRED CPI are optional enrichments to the same investigation.
- The canonical sales schema contains order identifier, order date, product identifier, product name, category, region, sales channel, customer segment, quantity, unit price, discount amount, and returned quantity.
- Derived revenue is not trusted as duplicated input. Gross revenue is quantity multiplied by unit price; return value is returned quantity multiplied by unit price; net revenue is gross revenue less discount and return value.
- Complete latest months compare against the preceding complete month. An in-progress latest month compares month-to-date against the same number of days in the preceding month. Explicit user instructions may override the default.
- Material Change defaults to both at least 15% relative change and absolute impact of at least 5% of prior-period total revenue. Explicit question parameters may override those defaults. Persistent learned preferences remain Phase 3 work.
- Analysis covers overall, product, category, region, channel, and customer-segment performance.
- The revenue bridge keeps volume, unit-price, price-volume interaction, discount, return, new-product, and discontinued-product effects distinct.
- Findings is a structured artifact containing the comparison window, data-quality results, overall metrics, dimensional breakdowns, revenue bridge, target variance, optional inflation adjustment, Material Changes, methodology, limitations, and source references. The Analyst may add a concise validated interpretation for CLI usability, but not a polished Report.
- The repository provides a fixed-seed generator for roughly 25,000–50,000 orders spanning six months for a multi-region home-and-office products retailer. It includes known commercial drivers, seasonal and weekday variation, discounts, returns, duplicates, and invalid records.
- Missing required columns or an unreadable required CSV fails the source. Invalid rows are quarantined. Later duplicate order identifiers are excluded. Negative quantities or prices, returns exceeding quantity, and discounts exceeding gross revenue are invalid.
- Analysis stops when quarantined rows exceed 1% of source rows. Findings disclose included, excluded, and duplicate counts and calculable revenue impact.
- Postgres holds six months of deterministic sales targets at month, region, and category grain, including target net revenue and target order count.
- FRED monthly CPI supports an optional approximate inflation-adjusted revenue comparison. Findings must state that national CPI is context and not causal evidence for product-level sales movement.
- The Fetcher normalizes source data to Parquet and emits a content-addressed Dataset Artifact with source identity, durable local location, content hash, schema, profile, row count, date range, quality summary, and creation time.
- Dataset contents remain outside CrewState and Postgres checkpoints in a Git-ignored artifact area. State and Handoffs carry typed references and profiles.
- The Analyst verifies artifact integrity before computation.
- DuckDB is the deterministic computation engine. Analyst Tools execute application-owned SQL templates only; models cannot provide arbitrary SQL.
- Fetcher is an LLM-driven Agent allowed to select only its CSV, read-only Postgres, and FRED Tools. Analyst is an LLM-driven Agent allowed to select only approved deterministic analysis Tools. General-purpose `execute_python` remains denied.
- Required CSV failure stops execution. Optional Postgres or FRED failure records an unavailable or invalid status, adds a limitation, and permits core analysis to continue.
- Transient Tool failures receive small bounded retries. Validation and authentication failures are not blindly retried. Invalid Agent output may be retried once with schema feedback.
- Phase 1 performs no autonomous re-planning. Exhausted failures are explicit and retain the original Plan and diagnostics.
- Step execution keys derive from the Step definition, analysis parameters, and input artifact hashes. Resume reuses matching completed Steps, invalidates affected downstream results after changes, and restarts failed or interrupted Steps from safe boundaries.
- Langfuse traces the Planner, Fetcher, Analyst, resolved Handoffs, Tool calls, retries, source status, and failures under one session.
- The no-paid-services constraint remains binding. OpenRouter free models remain configurable, and live tests remain opt-in because availability and quotas fluctuate.

## Testing Decisions

- Good tests assert external contracts, numerical results, state transitions, provenance, policy outcomes, and failure behavior. They do not assert exact LLM prose, private reasoning, internal SQL text, or incidental module structure.
- The preferred highest seam is the existing top-level Crew invocation, exercised through the CLI for acceptance. A single request should produce an executed Plan and structured Findings in CrewState while persisting checkpoints and trace metadata.
- The top-level acceptance fixture uses the deterministic synthetic retailer, seeded Postgres targets, and a known CPI response or live FRED call depending on test tier. Expected driver directions, bridge reconciliation, target variances, quality exclusions, and Material Changes are asserted.
- Deterministic Tool tests cover CSV validation and quarantine, deduplication, normalization, Dataset Artifact integrity, comparable-period selection, dimensional aggregations, revenue reconciliation, bridge calculations, target comparison, CPI adjustment, and materiality.
- Contract tests cover Plan validation, capability-aware Agent selection, catalogued source references, dependency ordering, Handoff resolution, Findings validation, and secret exclusion.
- Execution tests cover required-source failure, optional-source degradation, transient retry limits, invalid Agent output retry, lack of autonomous re-planning, and explicit terminal diagnostics.
- Resume tests cover matching Step reuse, source-hash invalidation, parameter invalidation, downstream invalidation, and interrupted-Step recovery.
- Scripted chat-model tests verify Fetcher and Analyst tool-calling loops and graph transitions without claiming to test analytical quality.
- Marked live tests verify real OpenRouter tool selection, read-only Postgres retrieval, FRED CPI retrieval, Postgres checkpoint persistence, and one coherent Langfuse trace.
- Ordinary test runs remain offline and deterministic. Live LLM, integration, and eval tests are opt-in markers and are required for the final Phase 1 acceptance run.
- Existing Phase 0 graph invocation, Pydantic contract tests, checkpoint tests, Memory integration tests, Langfuse trace test, and structural DeepEval test provide prior art. They will be evolved rather than duplicated into unrelated seams.
- Exact-string assertions against Agent summaries are prohibited. Tests validate Findings values, cited evidence, required caveats, and schema properties.

## Out of Scope

- Visualizer, Critic, and Reporter Agents and their execution loops.
- Chart generation, polished narrative Reports, critique/revision passes, and final-report citations.
- Arbitrary Python execution or model-generated SQL.
- Human-in-the-loop approval and the Phase 6 risk-tier workflow.
- Automatic re-planning, open-ended recovery loops, and dynamic execution-budget policies.
- Flexible inference of arbitrary user CSV column names; Phase 1 requires the canonical schema.
- Automatic filesystem discovery of Data Sources.
- Data warehouses, remote object storage, streaming ingestion, very large dataset optimization, and a general semantic layer.
- Causal inference from CPI or other macroeconomic series.
- Memory write/update Tools and learned cross-session analytical preferences.
- Full golden-set CI, structured cost/latency dashboards, deployment packaging, and graphical interfaces.

## Further Notes

- The accepted architecture and vocabulary are recorded in the root domain glossary and ADR-0001. This spec refines the Phase 0 Handoff contract because Phase 1 is the first point where it must become executable.
- The primary live demonstration is: compare the latest comparable sales periods, explain commercial drivers, reconcile actuals to internal targets, optionally show inflation-adjusted change, disclose quality limitations, and resume without recomputing unchanged work.
- Phase 1 is complete only when the live CLI scenario executes a capability-aware Plan through both new Agents, produces verified structured Findings, persists and resumes correctly, passes offline and marked live suites, and exposes the complete run in Langfuse.
- FRED free-tier and series details should be verified against current official documentation during implementation. The Data Source identifier remains configuration rather than prompt text.
