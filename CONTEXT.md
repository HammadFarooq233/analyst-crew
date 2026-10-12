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
Reviews a draft Report for judgment problems: causal overreach, Claims whose wording misreads their cited Evidence Entries, and overstated drivers. Citation validity and Caveat completeness are guaranteed by code, not by the Critic. Has no Tools; it reads the Report and the Evidence Catalog only.
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
Executes a read-only SQL query against the self-hosted Postgres database. Used by the Fetcher.
_Avoid_: DB connector, SQL runner

**FRED.fetch**:
Calls the FRED (Federal Reserve Economic Data) API and returns the requested series or metadata.
_Avoid_: API connector

**CSV.read**:
Reads a CSV file from disk and returns its contents as structured data.
_Avoid_: File reader, data loader

**chart**:
Renders a Chart Spec to a Chart Artifact (PNG). Used exclusively by the Visualizer.

**Chart Spec**:
A structured description of one chart: mark type, title, and which Evidence Entries or Findings series to plot. The Visualizer's LLM chooses these; code fills in the data from the Evidence Catalog, so a chart can never contain a number that is not a Tool value.
_Avoid_: Chart config, plot definition

**Chart Artifact**:
A durable, content-addressed rendered image of one Chart Spec. The Report's Charts section holds references to Chart Artifacts, never prose.
_Avoid_: Chart image, figure

**Critique**:
The Critic's structured verdict on a draft Report: a list of issues, each pointing at a Claim. Triggers at most one Reporter revision pass. The Critic then reviews the revised Report once more (the re-review); issues it still flags are shown in the Report as unresolved Reviewer Notes. A rationale is a note, not Report body text: it may contain digits and is truncated if over-long.
_Avoid_: Review, feedback

**critic_outcome**:
How the Critic pass ended, recorded in run metadata. `clean`: the first Critique had no issues and the draft shipped. `revised`: the Critique had issues, one Reporter revision was made and the re-review found none. `unresolved`: issues remain after the re-review, the revision or re-review failed, or the draft was a Fallback Report (which is never revised); the remaining issues are Reviewer Notes. `failed`: the Critic itself failed, the un-reviewed Report shipped and the failure is recorded (`critic_failure`).

**chart_mode**:
How the Report's charts were chosen, recorded on the Report and in run metadata. `llm`: the Visualizer selected them. `fallback`: the Visualizer failed and code built the deterministic fallback chart set. Absent when the Report has no charts.

**Reviewer Notes**:
The Report section listing Critique issues the revision pass did not resolve. Absent when the Critique is clean.

**Report**:
The final output returned to Priya. Contains a Summary (headline answer), Findings (cited claims), Charts (embedded from the Visualizer), Caveats (known limitations), and Methodology (sources and computations used).
_Avoid_: Brief, output, response

**Memory**:
Facts, preferences, and run summaries that persist across sessions and belong to one user. Retrieval is deterministic and scoped by user, memory kind and Agent role; Agents do not decide whether to look. Memory may change framing and the parameters Tools compute with, never which data is included. There are three kinds: typed preference, free-text fact, and past-run summary.
_Avoid_: Context, history, long-term storage

**Typed Preference**:
A Memory with a meaning the Crew can act on, drawn from a closed vocabulary (for example the Material Change threshold). The Crew resolves it into the analysis parameters of a Handoff. A statement that does not fit the vocabulary is kept as a free-text fact and never becomes a parameter.
_Avoid_: Setting, config, rule

**Free-text Fact**:
A Memory kept as the user's own wording. It may shape how a Report is framed but never changes a number.
_Avoid_: Note, context

**Past-run Summary**:
A Memory describing what the Crew found in an earlier run, tied to the source data it described. It is stale once that source data changes.
_Avoid_: History, log

**Pending Preference**:
An inferred Typed Preference waiting for the user's confirmation. It changes no number until the user confirms it (`confirm`); rejecting it discards it. Superseded, forgotten and rejected Memories stay in the history but never reach a run.
_Avoid_: Draft, suggestion

**Memory Update**:
The Crew-level step that writes Memory, from an explicit user statement or from inference over the user's own words and the Crew's run records. It is not an Agent.
_Avoid_: Memory Agent, learning

**Applied Memories**:
The Report section listing every Memory that influenced the run, with its origin (stated, inferred, or run record). Assembled by code, absent when none were applied.
_Avoid_: Used context, personalisation

**Marketing Spend**:
An optional Data Source of weekly marketing spend, retrieved from Postgres. When absent or unavailable the Report is still produced and names the missing source in its Caveats.
_Avoid_: Ad data, campaign data

**Spend Relationship**:
The Analyst's description of how weekly revenue and Marketing Spend move together across a fixed set of lags, with the paired-week count behind each figure and the number of lags tried. It is correlation only; a Report never states it as a cause.
_Avoid_: Spend impact, attribution, effect

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
