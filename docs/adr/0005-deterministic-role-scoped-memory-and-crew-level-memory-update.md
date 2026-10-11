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
