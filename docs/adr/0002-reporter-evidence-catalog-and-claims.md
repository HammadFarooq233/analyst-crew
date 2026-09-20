# ADR-0002: Reporter output is Claims over an Evidence Catalog

Status: Accepted

## Context

Phase 1 ends with validated `BaselineFindings` that the CLI prints as a formatted dump. No Agent writes a Report. Phase 2 slice 1 adds the Reporter (Visualizer and Critic follow in later slices). The product goal is "correct, cited numbers", and ADR-0001 established that analytical results come from Tool output, never model-authored arithmetic. The Reporter is an LLM, so the design must keep that guarantee while it writes prose, and must give the future Critic something concrete to check.

## Decision

**Claims cite Evidence Entries.** Code deterministically flattens `BaselineFindings` into an Evidence Catalog: numbered Evidence Entries, each with a stable id, raw value, display rendering, and source field. The Reporter returns a validated `Report` whose Findings section is a list of Claims, each citing one or more entry ids. A Claim citing an unknown id is invalid.

**Claim text contains no digits.** Numbers appear only as placeholders (`{E3}`) that the renderer substitutes with the entry's display string. After removing placeholders, Claim text must contain no digits. Dates, window labels, and names containing digits are catalog entries too. There is no number parsing or rounding tolerance.

**Split authorship by section.** The LLM writes the Summary and the Findings Claims (selection, order, emphasis). Code assembles Methodology (window, thresholds, bridge method, sources, integrity hash, row counts) and Caveats (every Findings limitation, every non-available optional source status, the CPI non-causality caveat, quarantine and duplicate counts). Caveats are mandatory and not selectable; the Reporter may add a Claim commenting on one but cannot replace or omit it. Charts is an explicitly empty section until the Visualizer slice.

**Reporter failure degrades, visibly.** Invalid Reporter output gets the existing one retry with validation feedback. If it still fails (or the provider fails after bounded retries), the terminal failure is recorded on the run and trace as in ADR-0001, and the Crew returns a deterministic template Report built from the Evidence Catalog with `report_mode: "fallback"`. The rendered fallback Report carries a visible notice that it was generated without the Reporter Agent.

## Consequences

- Every number in a Report is guaranteed to come from Tool output. This is a hard property, not a heuristic.
- Caveats cannot regress by omission, so the later Critic can focus on judgment (causal overreach, overstated drivers) rather than completeness.
- The template Report doubles as a deterministic baseline for tests and the eventual eval suite.
- Claim source text is stilted (`{E3}` placeholders); the rendered Report reads normally.
- The Evidence Catalog flattening is new deterministic code that must be tested offline and that decides what is citable; facts not in the catalog cannot appear in a Report.
- Two Report modes (`llm`, `fallback`) exist and must both be maintained.
