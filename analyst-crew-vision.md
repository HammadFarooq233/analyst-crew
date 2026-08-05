# Analyst Crew — Project Vision & PRD

**A multi-agent data analysis copilot, built as a hands-on tour of agentic AI engineering.**

Status: Draft v1
Owner: [your name]
Type: Personal learning project / portfolio piece

---

## 1. Vision Statement

Most "AI agent" tutorials teach you to wire one model to one tool and call it done. Real agentic
systems — the ones companies actually run in production — have to plan, delegate, remember,
justify their actions, fail safely, and be observable when they don't. **Analyst Crew** is a
data analysis copilot built specifically to force those problems into existence, so that solving
them becomes the actual curriculum.

The end state: a small team of cooperating agents that takes a data question in plain English,
pulls in relevant data from files, databases, and live APIs, plans and executes an analysis,
critiques its own work, remembers what it learned for next time, and reports back — all while
being debuggable, evaluable, permissioned, and deployable like a real service rather than a
notebook.

The project is deliberately scoped around **breadth over polish**: every major pillar of modern
agent engineering (orchestration, tools, memory, observability, evals, safety, deployment) should
have a real, load-bearing home in this system — not a bolted-on demo.

## 2. Problem Statement

### The learning problem

Tutorials teach isolated concepts (a backend here, a subagent there) but rarely force you to make
them work _together_ under realistic constraints — e.g., what happens to memory when you add a
second agent? What does an eval even check when there's no single correct answer? This project
exists to surface exactly those integration problems.

### The "pretend" real-world problem

Framing it as a product, not just an exercise, keeps design decisions honest. The target scenario:

> A small team (a startup, a solo analyst, a research group) has data scattered across CSV
> exports, a database, and external APIs. They don't want to write a new script every time someone
> asks "how did revenue change last quarter and why," and they don't fully trust a black-box tool
> with their data or their money-related decisions. They want something that can investigate,
> show its work, ask for permission before doing anything risky (spending API credits, sending
> messages, writing files), and get smarter about their specific data over time.

This framing isn't marketing fluff for the README — it's a forcing function. If a design decision
doesn't make sense for that person, it's probably not worth building.

## 3. Goals

### Learning goals (primary — this is a learning project first)

- Build first-hand fluency in: backends & state, multi-agent orchestration, tool design,
  persistent memory, observability/tracing, agent evaluation, permissions & human-in-the-loop,
  and deployment.
- Be able to explain _why_ each design decision was made, not just that it was implemented
  (this document exists partly to force that articulation).
- Produce a codebase and commit history that demonstrate progression, not just a final snapshot.

### Product goals (secondary — in service of the learning goals)

- Given a dataset (file, DB, or API-backed), answer non-trivial analytical questions with
  correct, cited numbers and a supporting chart.
- Be safe to run unattended on your own machine/data without needing to fully trust the model.
- Be inspectable: any answer should be traceable back to the tool calls and data that produced it.

### Non-goals

- Not trying to compete with commercial BI tools or become a SaaS product.
- Not optimizing for the largest possible dataset/scale — correctness and clarity over
  performance engineering.
- Not building a custom UI framework — a thin, "good enough" interface is fine; the agent
  internals are the point.

### Hard constraint: no paid services

This is unpaid, personal learning time — every tool, API, and library in this project must be
either open-source/self-hosted or a genuinely free tier with **no credit card required** and no
risk of a surprise bill. "Free tier that upsells you at scale" is acceptable only if it hard-gates
(stops working) rather than silently starts billing. See §11 for the specific stack this rules
in and out.

## 4. Target Users & Use Cases

Even as a personal project, designing for a persona keeps scope honest.

**Persona: "Priya," a solo operator / small-team analyst** who has sales, marketing, or ops data
in a few different places and no dedicated data engineer.

### Use case 1 — Recurring report, no repeated scripting

> Priya uploads this month's sales CSV and asks: _"Compare this month to last month and flag
> anything unusual."_ The agent recalls (via memory) that "unusual" for Priya's business means
>
> > 15% swings by product category — established in a prior session — applies that definition,
> > and produces a short report with a chart.

### Use case 2 — Multi-source investigation

> Priya asks: _"Did last week's marketing spend actually move revenue?"_ This requires pulling
> spend data from a database and revenue data from a CSV, joining them, and reasoning about
> correlation vs. causation caveats. A planner agent breaks this into subtasks for a data-fetch
> subagent and an analysis subagent, and a critic agent flags that correlation ≠ causation in the
> final write-up.

### Use case 3 — Guarded external action

> Priya asks the agent to _"email me and the team a summary."_ Sending external messages is a
> permissioned action — the system pauses for human-in-the-loop approval before it fires, and
> the action is logged for audit.

### Use case 4 — "Why did it say that?"

> Priya doesn't trust a number in a report and asks _"where did the $42,300 come from?"_ Because
> every step is traced (self-hosted Langfuse) and tool calls are logged, the agent (or Priya, via
> the trace viewer) can point to the exact query/file/computation that produced it.

### Use case 5 — Regression safety net (for you, the developer)

> You tweak the planner's prompt to make it more concise. Before merging, your eval suite runs
> a golden set of past questions and flags that the tool now skips a caveat it used to include —
> caught in CI, not in production.

## 5. Scope — Core Capability Areas

| Area                        | What "done" looks like                                                                                                                                            |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Orchestration**           | A supervisor agent decomposes a request and delegates to ≥2 specialized subagents (e.g., data-fetch, analysis, viz, critic); handoffs are explicit and traceable. |
| **Tool use / integrations** | At least one real external API and one real database (not just CSV), wired as tools with credentials kept outside the sandbox.                                    |
| **Memory**                  | Facts/preferences persist across sessions (`StoreBackend` or equivalent) and are demonstrably used in a later session, not just written and ignored.              |
| **Observability**           | Every run is traced (self-hosted Langfuse); custom logging captures tool latency, token usage, and errors.                                                        |
| **Evals**                   | A golden set of representative tasks with automated checks (not exact-string match) runs in CI on every change.                                                   |
| **Safety / permissions**    | Declarative permission rules restrict risky filesystem/tool operations; at least one action requires human-in-the-loop approval.                                  |
| **Deployment**              | Dockerized, reproducible setup; CI runs lint + evals; a documented path to run it (locally at minimum, hosted as a stretch goal).                                 |
| **Interface**               | Start CLI-first; a minimal chat/dashboard UI is a stretch goal once the above is solid.                                                                           |

## 6. Success Criteria

This is a learning project, so "success" is measured in demonstrated understanding as much as
working software:

- [ ] Can answer each of the 5 use cases above end-to-end, live, without cherry-picking inputs.
- [ ] Can explain, for each capability area in the table, _what would break if it were removed_
      (this is the real test of whether it's load-bearing vs. decorative).
- [ ] Eval suite catches at least one real regression during development (proof it's not
      theater).
- [ ] A stranger could clone the repo, follow the README, and get a working demo in under
      15 minutes.
- [ ] The GitHub history shows a clear phase-by-phase build-up (see roadmap), not one giant
      commit.

## 7. Roadmap (Phased Delivery)

Resequenced after the architecture Q&A (§13): Phase 0 is no longer empty scaffolding — it's a
**walking skeleton**, one real agent touching every architectural layer at minimum depth, real
LLM calls included. Every phase after it _adds breadth_ (more agents, more tools) or _adds depth_
(richer memory, richer tracing) to something that already runs end-to-end — never wires a layer
for the first time. Each phase is still independently demoable and commit-worthy.

0. **Walking skeleton** — one LangGraph node (Planner only), wired to the real `CrewState`
   schema (§13), a real Postgres checkpointer, a real Langfuse trace, one memory search tool,
   one real LLM call. No mocking. Validates the Handoff contract before anything else is built
   on top of it.
1. **Real tool use** — add Fetcher + Analyst; real CSV/FRED/Postgres tools. `execute_python` is
   registered on the Analyst but denied by default per §13's safety model (no HITL yet).
2. **Full orchestration** — add Visualizer, Critic, Reporter; real multi-agent handoffs across
   all 6 agents, not just the single Phase 0 node.
3. **Memory, deepened** — write/update tools, cross-session preference learning (Use Case 1's
   actual acceptance test), extending the one search tool Phase 0 already has.
4. **Observability, deepened** — structured logging, latency/cost breakdowns, extending the one
   trace Phase 0 already has.
5. **Evals & CI testing** — golden task set with automated evaluators, wired into CI. (The
   Phase 0 Planner run already doubles as the first test case, for free.)
6. **Safety & human-in-the-loop** — turn on `execute_python` and email/side-effect tools per
   §13's risk tiers; "denied by default" becomes "approved via HITL."
7. **Deployment** — Docker Compose, CI pipeline, optional hosted demo.
8. **Interface (stretch)** — CLI polish first; chat/dashboard UI once the agent is solid.

## 8. Risks & Mitigations

| Risk                                                    | Mitigation                                                                                                                      |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Scope creep — trying to gold-plate every phase          | Each phase has an explicit "done" bar in §5/§6; resist adding features beyond it.                                               |
| Evals devolve into checking exact wording               | Design evaluators around _properties_ of a good answer (cites correct numbers, includes required caveats), not string matching. |
| Memory becomes a gimmick nobody's answer depends on     | Use case 1 is the acceptance test — a specific fact must change a real answer.                                                  |
| Multi-agent adds complexity without adding value        | Only split into subagents where a task genuinely decomposes (data-fetch vs. analysis vs. critique are naturally separable).     |
| Security shortcuts under "it's just a personal project" | Treat secrets/permissions seriously anyway — it's the whole point of Phase 7 and a core hiring signal.                          |

## 9. Open Questions — Resolved

- ~~Which live API and which DB best fit a believable "small business analyst" story?~~ →
  resolved in §11: FRED (economic data) + self-hosted Postgres.
- ~~LangSmith free tier vs. self-hosted tracing/logging?~~ → resolved in §11: self-hosted
  Langfuse, chosen specifically because it has no usage cap or billing risk at all.
- Chat UI vs. dashboard for the eventual interface layer — still open; deferred to Phase 9,
  not worth deciding before the agent itself exists.

## 10. Roadmap Status Log

A running log of what's actually true vs. planned, so this document stays honest as building
progresses. Update this table, don't just the roadmap above.

| Phase                       | Status      | Notes                                                     |
| --------------------------- | ----------- | --------------------------------------------------------- |
| 0 – Walking skeleton        | Not started | Planner only, full stack thin                             |
| 1 – Real tool use           | Not started | Fetcher + Analyst; `execute_python` registered but denied |
| 2 – Full orchestration      | Not started | Visualizer, Critic, Reporter join                         |
| 3 – Memory, deepened        | Not started |                                                           |
| 4 – Observability, deepened | Not started |                                                           |
| 5 – Evals                   | Not started |                                                           |
| 6 – Safety/HITL             | Not started | `execute_python` moves to HITL-gated                      |
| 7 – Deployment              | Not started |                                                           |
| 8 – Interface               | Not started |                                                           |

## 11. Locked Technical Stack

Decided after a dedicated research pass (see rationale notes below each row). Constraint
driving every choice here: **zero paid services, zero cards on file, zero surprise-bill risk**
(§3, Hard constraint). Where a tool has a paid tier, that's noted only to explain why it was
rejected or how it's being used safely below the paid threshold.

| Layer                           | Choice                                | Cost                                                    | Why                                                                                                                                                                                                           |
| ------------------------------- | ------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Orchestration                   | `deepagents` on LangGraph             | Free (open source)                                      | Already invested; native checkpointing, state, and human-in-the-loop primitives match this project's needs directly — no paid tier exists for the framework itself.                                           |
| LLM provider (primary)          | OpenRouter free models                | Free models available without a paid model subscription | Availability and rate limits vary by model; the configured model can be changed without code changes.                                                                                                         |
| LLM provider (fallback/offline) | Local model via Ollama                | Free                                                    | Zero rate limits, zero network dependency, useful for heavy iteration (e.g. running the eval suite repeatedly) without touching any API quota.                                                                |
| External data API               | FRED (Federal Reserve Economic Data)  | Free, unlimited, no card                                | Public-good government API with no paid tier at all — removes the "will I get rate-limited into a paywall" risk entirely.                                                                                     |
| Database                        | Postgres (Docker container)           | Free (self-hosted)                                      | Doubles as the backing store for LangGraph's checkpointer and long-term memory store — one piece of infra, two jobs.                                                                                          |
| Memory                          | LangMem, backed by the Postgres store | Free (open source)                                      | Native LangGraph long-term memory layer; no separate hosted memory product needed.                                                                                                                            |
| Observability                   | Langfuse, self-hosted via Docker      | Free (open source, self-hosted; MIT license)            | Chosen specifically over LangSmith: LangSmith's free tier hard-caps at 5,000 traces/month, which a solo dev doing heavy iterative testing can burn through quickly. Self-hosted Langfuse has no such ceiling. |
| Evals                           | DeepEval                              | Free (open source, runs locally)                        | Pytest-native, no account or cloud dependency required to run a CI eval gate.                                                                                                                                 |
| Deployment                      | Docker Compose + GitHub Actions       | Free (public/personal repo tier)                        | Standard, no paid infra needed for a project of this size.                                                                                                                                                    |
| Interface (stretch)             | FastAPI + Streamlit/Gradio            | Free (open source)                                      | No hosting commitment required to demo locally; a free-tier host (e.g. a static/community tier) can be considered later, but isn't required for the project to be "done."                                     |

**Practical note on OpenRouter free models:** Availability and rate limits can change — if a
future model swap or limit makes the free tier too restrictive mid-project, the fallback is not
"start paying," it's "switch to Ollama for that phase" or select another free OpenRouter model.
Because the orchestration layer is model-agnostic, this swap should only touch config, not code —
worth verifying that stays true as the project grows.

## 13. Architecture Decisions Log

Resolved during a deep Q&A pass with the coding agent, kept here so the reasoning isn't lost.

**Code execution (Analyst/Visualizer).** Code execution is a Tool, never an implicit LLM
capability — the LLM only produces code text; a separate process (`execute_python`, backed by
`LocalShellBackend`) runs it and returns the result as the next observation. Analyst gets this
tool; Visualizer instead uses **structured chart-spec output** (JSON: mark/x/y/data) rendered by
our own fixed code — safer and eval-friendly, since chart shape is a closed, checkable space.
Fetcher/Planner/Reporter get no execution access; Critic reuses Fetcher's read-only query tools
to verify citations, never gains execution access itself.

**Memory access.** Per-agent, via each agent's own scoped `search_memory` tool — not a single
Planner-mediated fetch. Each agent queries memory at the point it needs it, scoped to its own
role, rather than the Planner pre-guessing and injecting everything for everyone.

**Safety / permissions.** Defense-in-depth at two levels: **agent-level** (only give
`execute_python` to the Analyst — other agents never have it in their tool list) and
**tool-level** (each tool's own risk tier — read-only/sandboxed/side-effect/destructive —
governs whether a call auto-approves, sandbox-restricts, or requires HITL). Safe default before
Phase 6 (HITL) exists: **deny**, don't "sandbox and auto-approve" — an unverified sandbox isn't
a control worth trusting yet. `execute_python` is registered but denied in Phase 1, HITL-gated
starting Phase 6.

**Crew state.** Lives in a LangGraph `CrewState` schema with a small number of top-level keys,
each with the correct reducer — `current_step` (overwrite), `steps_completed` (append),
`artifacts` (keyed dict, merges on write), `messages` (LangGraph's standard `add_messages`
convention). Session metadata (`session_id`, `user`) belongs in LangGraph's `config`, not in
state — it's set once per run and is what the checkpointer already keys off.

**Phase 0 scope.** A walking skeleton, not isolated-agent testing or full scaffolding-only —
see §7's resequenced roadmap.

## 14. Appendix — Concepts This Project Exercises

Orchestration & subagents · backends & state management · tool design & credential isolation ·
short-term vs. long-term memory · tracing/observability · agent evaluation methodology ·
permissions & human-in-the-loop review · containerization & CI/CD for agentic systems.

---

_This document is a living guide, not a contract — update §10's status log as phases complete,
revisit §11 if any free tier tightens enough to become a blocker, and append new resolutions to
§13 as further architecture questions get settled._
