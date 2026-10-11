# ADR-0005: Deterministic, role-scoped Memory retrieval and a Crew-level Memory Update

Status: Accepted

## Context

Vision §13 says each Agent reads Memory through its own scoped `search_memory` Tool and decides when to call it. In practice the Planner already retrieves Memory deterministically, application-side, because the model did not call the Tool reliably. ADR-0001 and ADR-0002 require that every number come from a Tool and that the Handoff carry the parameters an Agent needs. Phase 3 adds persistent, per-user Memory that can change what a Report says, and automatic inference that can write it.

## Decision

**Retrieval is deterministic and application-side**, scoped by user, memory kind and Agent role. The Planner receives past-run summaries and free-text facts. The Analyst receives parameters only. The Reporter receives framing facts only. The Critic, Fetcher and Visualizer receive none. Typed preferences are resolved by the Crew into the Handoff's analysis parameters.

**Memory changes framing and parameters, never data inclusion.** An instruction to ignore a category is kept as a free-text fact and is not applied as an exclusion, so the revenue bridge still reconciles with the source data. Every Memory that influenced a run is disclosed under Applied Memories.

**Writing is a Crew-level Memory Update step, not an Agent.** Explicit statements are validated in code against a closed typed-preference vocabulary and apply immediately. Inference learns only from the user's own words and the Crew's run records, never from free-form Agent prose. Inferred free-text facts apply immediately; inferred typed preferences stay pending until the user confirms them.

**Storage is a persistent store keyed by user and kind**, so isolation between users is structural.

## Considered options

- Keep per-Agent model-callable search Tools: rejected, since use would be unreliable, untestable and hard to disclose.
- A seventh Memory Agent: rejected; it would need to be stateless and handed everything, and it adds an LLM judgment where a validated write is enough.
- Apply inferred parameters immediately: rejected, since a misreading would silently change a Report's numbers before Phase 6 adds approval flows.

## Consequences

- The vision's per-Agent `search_memory` wording is superseded.
- Memory use is testable through the Crew and always visible on the Report and the trace.
- Applied memories become part of resume keys.
- Extending the preference vocabulary is a deliberate code change.

## As shipped (Phase 3 close-out, issue #33)

Checked against the code and the live acceptance run (`docs/phase-3-acceptance-status.md`). The decision stands; these details are how it landed:

- **The preference vocabulary has two keys**: the Material Change relative threshold and the absolute-impact threshold, each range-checked in code. The "grouping dimension emphasised" preference named in the Phase 3 spec was not built: no analysis parameter selects a grouping dimension (every dimension is always computed), so it would map onto nothing. Adding it is a deliberate vocabulary change together with an Analyst parameter.
- **A question overrides a stored preference for that run.** The Applied Memories line then says the stored preference was not used; the Methodology states the threshold source (`default`, `analysis_parameters`, `user_question` or `memory`).
- **Past-run Summaries** are written by the Memory Update step from run records only (Findings, Plan Steps, Step outcomes; never Report or Critic prose) with the source file's content hash. They go to the Planner and are skipped as stale when the hash no longer matches. They are listed under Applied Memories as "Past-run Summary given to the Planner".
- **Inference** keeps the ADR rules and adds code-side grounding: quoted evidence must appear in the user's words (or, for facts, the run records), numbers must appear in the evidence, and a per-run cap applies. Pending preferences are listed by `list` and resolved with `confirm` / `reject`.
- **Storage outage** degrades to defaults and states so in Caveats; the run does not fail. Applied Memories show the Memory's kind and origin, never its stored text, so free text cannot put a number in a Report.
- **Superseded, forgotten and rejected Memories** are kept in the history (`list --history`) and are never retrieved.
