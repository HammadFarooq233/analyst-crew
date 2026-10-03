# ADR-0003: Visualizer selects, Critic judges, code owns every number

Status: Accepted

## Context

Phase 2 adds the Visualizer and Critic. Vision §13 originally sketched the Visualizer emitting a full chart spec (`mark`, `x`, `y`, `data`) and the Critic reusing the Fetcher's read-only query tools to verify citations. ADR-0001 and ADR-0002 already require that every number come from Tool output and that citation validity and Caveat completeness are enforced by code.

## Decision

**The Visualizer selects; code supplies data.** The Visualizer's LLM returns a Chart Spec: mark, title, and which Evidence Entries or Findings series to plot. Code fills in the data from the Evidence Catalog and renders a PNG Chart Artifact. Marks are a closed set (`bar`, `grouped_bar`, `waterfall`), at most three charts per Report, and every series reference must resolve in the catalog. If the Visualizer fails after its bounded retry, a deterministic fallback chart set is built from the catalog and `chart_mode: fallback` is recorded.

**The Critic judges; it has no Tools.** The Critic reads the draft Report and the Evidence Catalog and returns a Critique: issues with a closed category (`causal_overreach`, `misread_evidence`, `overstated_driver`, `unsupported_emphasis`), each targeting one existing Claim or the Summary, with a short rationale. It cannot add Claims. A Critique with an invalid target gets one retry, then counts as Critic failure.

**One bounded revision pass.** A non-empty Critique triggers at most one Reporter revision, validated by the same rules as the first draft. Issues that remain are shown in the Report as Reviewer Notes. If the Critic fails, the un-reviewed Report ships with the failure recorded, never the run failing.

## Considered options

- Visualizer writes full `data` and code validates it: rejected, since "validate afterwards" is weaker than "cannot be wrong by construction" and adds a reconciliation surface.
- Critic re-queries Postgres/CSV to spot-check values: rejected, since the catalog already guarantees values are Tool outputs; re-querying duplicates that and gives the Critic a capability it does not need. This deliberately deviates from the original vision §13 line.
- Critic can block (return Fallback Report): rejected; a reviewer outage should not destroy a valid Report.

## Consequences

- No chart or Report can contain a number that is not a Tool value; this stays a structural property.
- The Critic's value is judgment the code cannot check, so its quality must be evaluated by properties (a seeded causal-overreach Claim must be flagged), not exact wording.
- Reviving a tool-using Critic later (for example for multi-source use cases) is a new decision that supersedes this one.
- `chart_mode` and `critic_outcome` (`clean | revised | unresolved | failed`) become part of run metadata.
