# Analyst Crew

A multi-agent data analysis copilot — the system Priya talks to. Orchestrates, delegates, and reports.

## Language

**Crew**:
The entire running system. Receives a user question, plans work, delegates to Agents, and returns a report.
_Avoid_: The system, the app, the pipeline

**Agent**:
A specialized role within the Crew that does one class of work. Each has its own tools, prompt, and handoff contract.
_Avoid_: Subagent, worker, node

## Agents

**Planner**:
Decomposes the user's question into a structured plan. Decides which Agents to invoke, in what order, and what each should produce.
_Avoid_: Supervisor, orchestrator, router

**Fetcher**:
Retrieves data from sources (CSV, Postgres, FRED API). Returns raw data — no analysis, no interpretation.
_Avoid_: Data loader, connector, extractor

**Analyst**:
Performs computation on data — statistics, joins, comparisons, trend detection. Consumes data, produces findings.
_Avoid_: Calculator, processor, query runner

**Visualizer**:
Generates charts and tables from findings.
_Avoid_: Chart maker, plotter, graph builder

**Critic**:
Reviews a draft report for uncited claims, missing caveats, and logical gaps.
_Avoid_: Reviewer, verifier, checker

**Reporter**:
Writes the narrative report from findings and charts. Synthesizes results into prose, cites specific numbers, and addresses Critic feedback in a revision pass.
_Avoid_: Writer, summarizer, formatter

**Plan**:
An ordered sequence of Steps produced by the Planner. Each Step names a target Agent, what to produce, and what prior Steps it depends on.
_Avoid_: Workflow, pipeline, DAG

**Step**:
A single unit of work within a Plan: invoke one Agent to produce one output.
_Avoid_: Task, job, node

**Handoff**:
The Crew resolves a Step's dependencies into explicit input values and passes them to the target Agent. Agents are stateless — they only receive the inputs their Step declares.
_Avoid_: Context injection, state sharing

A Handoff contains the Step, resolved input artifact references, the user's question, and explicit analysis parameters. Secrets and unresolved source locations never belong in a Handoff.

## Tools

**Tool**:
A callable capability exposed to an Agent. The Agent's LLM decides whether to invoke it; the Crew or a sandbox executes it and returns the result as the next observation.
_Avoid_: Function, plugin, skill

**execute_python**:
Runs arbitrary Python/pandas code in a sandbox and returns the result or error. Used exclusively by the Analyst.
_Avoid_: Code runner, REPL

**Postgres.query**:
Executes a read-only SQL query against the self-hosted Postgres database. Used by the Fetcher and Critic.
_Avoid_: DB connector, SQL runner

**FRED.fetch**:
Calls the FRED (Federal Reserve Economic Data) API and returns the requested series or metadata.
_Avoid_: API connector

**CSV.read**:
Reads a CSV file from disk and returns its contents as structured data.
_Avoid_: File reader, data loader

**chart**:
Generates a chart image from a structured JSON spec (`mark`, `x`, `y`, `data`). Used exclusively by the Visualizer.

**Report**:
The final output returned to Priya. Contains a Summary (headline answer), Findings (cited claims), Charts (embedded from the Visualizer), Caveats (known limitations), and Methodology (sources and computations used).
_Avoid_: Brief, output, response

**Memory**:
Facts, preferences, and session artifacts that persist across user sessions. Each Agent accesses it independently via a scoped `search_memory` tool, not through a central Planner-mediated fetch.
_Avoid_: Context, history, long-term storage

**Data Source**:
An explicitly supplied location from which the Fetcher may retrieve data. The Crew receives a catalog describing available Data Sources; Agents must not invent file paths, database tables, or external series.
_Avoid_: Input, dataset location, connector

**Findings**:
The Analyst's structured output containing computed metrics, comparisons, material changes, anomalies, and data-quality limitations. Findings are evidence for a later Report; they are not themselves polished narrative prose.
_Avoid_: Report, answer, analysis text

**Evidence Catalog**:
A deterministic, code-built flattening of Findings into a numbered list of Evidence Entries. The Reporter selects from it; it never computes or copies values itself.
_Avoid_: Fact list, context dump

**Evidence Entry**:
One citable fact in the Evidence Catalog: a stable id, the raw value, its display rendering, and the Findings field it came from. It also carries a human-readable label for the Reporter and, for a driver or target row, a group key and a Material Change flag. Claims cite Evidence Entries by id.
_Avoid_: Reference, citation, fact

**Claim**:
A single item of a Report's Summary or Findings section, written by the Reporter and citing one or more Evidence Entries. Code-assembled Caveats and Methodology lines are not Claims. A Claim is only valid if every cited id exists in the Evidence Catalog.
_Avoid_: Statement, insight, finding (the singular is ambiguous with Findings)

**Fallback Report**:
A deterministic, template-only Report built from the Evidence Catalog when the Reporter fails after its bounded retry. It is visibly marked as generated without the Reporter Agent, and the Reporter's failure is still recorded.
_Avoid_: Degraded report, default report

**Dataset Artifact**:
A durable, content-addressed reference to normalized fetched data. It carries provenance, schema, profile, integrity hash, and storage location; full datasets do not live in CrewState.
_Avoid_: Dataset, raw records, state payload

**Material Change**:
A change worth surfacing under explicit percentage and absolute-impact thresholds. Phase 1 defaults to at least 15% change and an absolute revenue impact of at least 5% of prior-period total revenue; explicit user instructions may override those defaults.
_Avoid_: Significant change, important difference
