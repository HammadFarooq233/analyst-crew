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