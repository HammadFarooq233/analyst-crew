# ADR-0004: Azure for development and live acceptance; free or local models for CI and evals

Status: Accepted (monthly spend ceiling not yet set)

## Context

Vision §3 makes "no paid services" a hard constraint and §11 locks OpenRouter free models, with Ollama as the fallback. Phase 1 live acceptance was blocked for weeks by free-tier limits: a 50-requests-per-day cap, upstream rate limiting, and a low rate of schema-valid structured output on the models tried. Azure OpenAI (Microsoft Foundry) was then added as a provider, and live acceptance for Phases 1 and 2 ran on it. Azure is metered, so it conflicts with the hard constraint as written.

## Decision

Azure is the provider for development and live acceptance runs. CI and evals must run on free or local models, so the project's recurring cost stays bounded and the no-surprise-bill property is kept where runs are most numerous. The provider stays a configuration choice, never a code change.

A working free or local model path that passes the structured-output tests is an entry criterion for Phase 5 (evals in CI). Evals cannot start until it exists.

A monthly Azure spend ceiling is to be set by the owner and recorded here.

## Considered options

- Azure everywhere: rejected, since evals in CI multiply model calls and the cost is open-ended.
- Free models everywhere: rejected, since Phase 1 showed live acceptance cannot be completed reliably on them.
- Amend the hard constraint and drop it: rejected; the constraint still governs the parts of the system that run unattended.

## Consequences

- Vision §3 and §11 need a note that Azure is a sanctioned exception for development and acceptance.
- Offline tests need no model and are unaffected.
- Until the ceiling is recorded, live runs are bounded only by judgment.
