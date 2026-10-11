# Analyst Crew: project plan after Phase 2

Status: **Draft for team review**, 2026-10-10. Owner: Hammad Farooq.
Sources: `analyst-crew-vision.md` (§7 roadmap, §10 status log), GitHub issues #14–#22, `docs/phase-*-acceptance-status.md`, ADR-0001 to 0003.

## 1. Where we are

| Area | State |
| --- | --- |
| Phases 0 and 1 | Complete. Live acceptance passed 2026-09-19 on Azure OpenAI. |
| Phase 2 implementation | Done. All six Agents exist: Planner, Fetcher, Analyst, Visualizer, Reporter, Critic. Charts, Critique and one bounded revision are in. Merged to the `phase-2-orchestration` branch. |
| Phase 2 PR | [#22](https://github.com/HammadFarooq233/analyst-crew/pull/22) is open. QA is running in another session. |
| Phase 2 close-out | Issue #21 is open and assigned to a human: the live acceptance run and docs. Vision §10 still says "In progress". |
| Product exposure | The only interface is `python cli.py --question ... --sales-csv ...`. Only the coding agent has run it. No human has judged the output. |

**The main gap is not engineering.** We have a lot of passing tests and no human opinion on whether the Reports are good. The plan therefore starts by closing that gap.

## 2. Recommended way to proceed

1. **Run the product yourself before planning in detail (1–2 days).** See §3. This is the cheapest, highest-value step. Its findings should reshape the backlog.
2. **Close Phase 2 properly** (#21 acceptance, merge #22), then freeze it.
3. **Re-order the remaining roadmap** (§4). My recommendation is to bring minimal CI forward and to demote "memory" from a pure learning exercise to the one feature that proves the product's value.
4. **Run the team in short, labelled vertical slices.** The repo already does this well: PRDs, ADRs, issue labels, an acceptance doc per phase. Keep it.

## 3. Step 0: hands-on validation ("Phase 2.5", gate before Phase 3)

Goal: someone who did not build it uses it like Priya (the persona) would, and writes down what they saw.

| # | Task | Owner | Output |
| --- | --- | --- | --- |
| 0.1 | Clone fresh, follow the README only, time it. Target: demo in under 15 minutes (vision §6). | You | List of README or setup gaps |
| 0.2 | Run the README scenario. Open the Markdown Report and the chart PNGs. Read it as a stranger. | You | Notes: is it clear, correct, useful? |
| 0.3 | Ask 5 to 8 questions of your own, not the scripted one, including ones the Crew should refuse or caveat. | You | Question log with pass/fail and why |
| 0.4 | Bring your own CSV with different columns. | You | Records how sales-specific the code is |
| 0.5 | Break it on purpose: kill Postgres, remove the API key, give a damaged CSV. | You | Is every failure message understandable? |
| 0.6 | Open Langfuse and follow one run end to end (use case 4, "where did this number come from?"). | You | Is the trace usable by a non-author? |
| 0.7 | Triage everything into GitHub issues: `bug`, `ux`, `scope`. | You | Backlog input for Phase 3 and beyond |

Exit criteria: one written paragraph answering "would I trust this Report?", plus a triaged list. Output goes to `docs/phase-2-hands-on-findings.md`.

Treat this as a gate, not a recommendation. If the Reports are not trustworthy, building memory on top of them is wasted effort.

## 4. Roadmap

The vision's order is 3 Memory, 4 Observability, 5 Evals/CI, 6 Safety/HITL, 7 Deployment, 8 Interface. I recommend this order instead. Effort is a rough estimate, not a commitment; re-estimate once the team and capacity are known.

| Order | Phase | Why this position | Rough size |
| --- | --- | --- | --- |
| A | **Close Phase 2** (#21, merge #22, update vision §10, README) | Unblocks everything. Needs live keys, Docker and a human. | 2–3 days |
| B | **Hands-on validation** (§3) | Gate. Reshapes the rest. | 1–2 days |
| C | **Minimal CI** (pulled forward from Phase 5/7): GitHub Actions running the offline suite, lint, and a docs check | The offline suite is already large (~166 tests). It should protect every PR now, not after four more phases. Live tiers stay manual. | 1–2 days |
| D | **Phase 3: Memory, deepened** | Use case 1 is the real acceptance test: a remembered definition of "unusual" must change a later Report. This is the first feature a user would notice. | 2–3 weeks |
| E | **Phase 5: Evals** (golden set, DeepEval, regression gate) | Needs a stable Report to build on. Extends the seeded-flaw Critic eval (#20). | 1–2 weeks |
| F | **Phase 4: Observability, deepened** (latency and cost breakdowns, structured logs) | Cheap once evals give us something to measure. Cost data also informs the model-provider decision (§7). | 1 week |
| G | **Phase 6: Safety and HITL** (`execute_python`, guarded email, approvals, audit log) | Highest design risk. Do it after the core is trusted. Use cases 2 and 3 land here. | 2–3 weeks |
| H | **Phase 7: Deployment** (Compose hardening, release pipeline, optional hosted demo) | Packaging. | 1 week |
| I | **Phase 8: Interface** (FastAPI plus a simple UI) | Only after the engine is trustworthy. The CLI is enough until then. | 2 weeks, optional |

Note: the vision's roadmap lists "Visualizer, Critic, Reporter" under Phase 2 and calls the interface "Phase 9" in §9. Both are stale; fix them when updating §10.

### 4.1 Phase 3 outline (the next real build)

Draft this as a PRD and an ADR before coding, as Phases 1 and 2 did.

- Write, update and delete memory tools (the Phase 0 search tool already exists).
- Per-Agent scoped `search_memory`, per vision §13.
- Preference capture: "unusual means a swing above 15% by product category".
- **Acceptance test:** run 1 states a definition; run 2, in a new thread, applies it and the Report cites it as a Methodology or Caveat line.
- Open design questions: how a remembered threshold interacts with the configured Material Change thresholds; how to show the user what was remembered; how to forget.

## 5. Team coordination

I do not know the team's size or roles. The table below is a template; fill in names.

| Role | Responsibility | Person |
| --- | --- | --- |
| Product owner and project lead | Priorities, scope, acceptance sign-off, stakeholder updates | Hammad |
| Tech lead / reviewer | ADRs, PR review, architecture consistency | TBD |
| Implementers (humans and coding agents) | Slices from `ready-for-agent` issues | TBD |
| QA / verification | Live acceptance runs, hands-on testing, evidence docs | TBD |

### Working agreements

- **One source of truth:** GitHub issues for work, `docs/` for decisions and evidence, vision §10 for status. Chat is not a record.
- **Definition of done for a slice:** offline suite green; behaviour covered through the Crew seam; ADR or glossary updated if the design changed; no unrecorded live behaviour.
- **Definition of done for a phase:** an acceptance status doc with live evidence, stated limits ("not exercised live: X"), vision §10 updated, README accurate.
- **Human gate per phase:** AI-agent implementation is fast, so the bottleneck is human judgment. Every phase ends with a human running the product (Step 0 repeated, shorter).
- **Agent work rules:** issues carry `ready-for-agent` or `ready-for-human`. Live runs and anything needing keys stay `ready-for-human`.
- **Cadence:** a weekly 30-minute sync (status, blockers, decisions); async daily notes if the team is larger than three; a short demo at the end of each phase.
- **Decision log:** new architectural decisions get an ADR (the 0001–0003 format works well).

### Reporting

- Weekly status: shipped, in progress, blocked, decisions needed, risks.
- The existing EOD and weekly-report skills can generate these from git, issues and PRs.

## 6. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Nobody has judged output quality | High: we may be building on unreliable Reports | Step 0 gate |
| Live acceptance depends on one person, keys and Docker | Schedule slip | Document the run as a script; keep evidence in `docs/` |
| LLM provider instability (OpenRouter free tier blocked Phase 1 for weeks) | Blocks live testing | Keep the Azure/OpenRouter switch; decide on a default (§7) |
| Sales-only assumptions baked in | Product feels narrow | Step 0.4; decide whether generalising is in scope |
| Eval flakiness (model-judged checks) | CI noise | Property-based checks only, per vision §8 |
| Scope creep (UI, hosting) before the core is trusted | Wasted effort | Phase 8 stays last and optional |
| Prompt or model changes silently degrade Reports | Regression | Phase 5 gate plus pinned model ids |

## 7. Decisions needed from you

1. **Team and purpose.** The vision describes a personal, unpaid learning project with a "no paid services" rule. This brief talks about a team and a user-facing product. Which is it? The answer changes the Phase 8 priority, the documentation tone, and whether the Azure OpenAI use (which may bill) is acceptable.
2. **Default LLM provider** for the team: Azure (works, may cost money) or OpenRouter free (free, unreliable).
3. **Is generalising beyond sales data in scope?** If yes, it needs its own phase.
4. **Who runs live acceptance** when you are unavailable?
5. **Approve the re-ordering in §4**, in particular pulling minimal CI forward.

## 8. Immediate next actions (this week)

1. Finish Phase 2 QA and merge #22 (QA session, in progress).
2. Run issue #21 live acceptance and write `docs/phase-2-acceptance-status.md`.
3. Do the §3 hands-on session and triage the findings.
4. Answer the §7 decisions.
5. Open the Phase 3 PRD issue and set up the minimal CI.
6. Circulate this plan to the team once the §7 answers are in.
