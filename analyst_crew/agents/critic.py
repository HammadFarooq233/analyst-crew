"""Critic Agent and graph node (ADR-0003).

The Critic has no Tools. Its inputs are the draft Report and the Evidence
Catalog; it returns a Critique whose issues each target the Summary or an
existing Claim. It cannot add Claims. Code validates the Critique, turns the
issues into Reviewer Notes on a copy of the draft, and records ``critic_outcome``
as ``clean``, ``revised``, ``unresolved`` or ``failed``. A non-empty Critique triggers one validated
Reporter revision (see ``_apply_critique``). A Critic failure ships the
un-reviewed Report with the failure recorded; it never fails the run.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from analyst_crew.agents.agent_support import (
    catalog_prompt, format_failure, named_span, run_validated, summarize_schema_error,
)
from analyst_crew.agents.contracts import CREW_AGENTS, Crew, Handoff, Plan
from analyst_crew.agents.reporter import production_reporter_model, revise_content
from analyst_crew.agents.resume import _resume_decision, execution_key
from analyst_crew.llm import make_chat_model
from analyst_crew.reporting.catalog import EvidenceCatalog
from analyst_crew.reporting.critique import Critique, report_items, report_targets, validate_critique
from analyst_crew.reporting.report import Report, build_report, substitute
from analyst_crew.state import CrewState

CRITIC_SYSTEM_PROMPT = """You are the Critic Agent for Analyst Crew, a multi-agent data analysis copilot.

You review a draft Report for judgment errors that code cannot check. Code has already guaranteed that every
number is an Evidence Entry value, that citations resolve and that caveats are present. You have no Tools and no
data beyond the Evidence Catalog below; you never compute and you never add Claims or rewrite text.

Most draft Reports are sound, and an empty list of issues is the expected outcome for a sound Report. Flag an
issue only when you can state a concrete error: wording that asserts something the Evidence Catalog contradicts
or does not support, in one of these four categories (use exactly these):
- causal_overreach: states or implies that something caused the change (because, due to, driven by a named
  cause, so X led to Y) when the data only shows what changed. Describing which drivers account for how much
  of the change is not causal claim; naming an outside reason (a promotion, a launch, the weather) is.
- misread_evidence: the wording misreads its cited entries (wrong direction, wrong subject, wrong comparison,
  wrong period).
- overstated_driver: presents a driver as larger or more important than its entries show, for example calling a
  small contributor "the main driver".
- unsupported_emphasis: stresses or generalises beyond what the entries support, for example "dramatic",
  "all regions" when only one is shown, or a trend claimed from two points.

Before flagging anything, check the wording against the Evidence Entries, not against your own expectations:
- A superlative or comparison ("largest", "biggest", "most", "only", "fell", "higher than") is NOT an issue when
  the cited entries, or other entries in the catalog, show it. Compare the entries yourself (for instance, the
  driver with the biggest absolute change or share of the change among the catalog's drivers of that dimension
  is "the largest"); do not demand that the claim's own citations list every competitor.
- In this product "unusual", "flagged", "notable" and "stands out" mean a Material Change, and the catalog marks
  those entries [MATERIAL CHANGE]; wording that calls a marked item unusual, flagged or the main/largest one
  is supported when its absolute change or share of the change is the largest among the marked items, and
  when it is the only marked item (the "Number of Material Changes" entry says how many) it is trivially
  the main, largest and only unusual item. Do not ask for a broader ranking, a statistical test or proof of
  "unusualness".
- Decomposition language is arithmetic, not causation: "driven mainly by the volume effect", "accounted for",
  "contributed" and "explains" describe how much of the change each driver or revenue-bridge effect makes up, and
  a share-of-change entry that is positive on a decline means the driver moved the same way as the overall
  change. Flag causal_overreach only for a reason the data cannot show (a promotion, a launch, customer
  behaviour, seasonality) or for calling the decomposition proof of cause.
- Do not flag a claim for being uncertain, for not being exhaustive, or because it could have been worded more
  cautiously. Style and tone are not issues.
- Judge what the wording literally states, not what it might be taken to imply: wording the entries support is
  not an issue because a reader could infer more from it.
- Only emit an issue whose rationale describes an error. If, while writing it, you conclude the wording is
  supported, or you find yourself writing "is supported, however", drop the issue.
- Use one issue per genuine error; do not repeat the same objection on several claims. If you are unsure
  whether wording is wrong, it is not an issue.

Rules:
1. Each issue targets exactly one item: "summary" or "claim_<n>" where <n> is the Finding number shown below.
2. Give a short rationale (at most a few sentences) that says what the wording asserts and what the entries
   show. Refer to an entry by placeholder, for example {E12}, rather than restating its value, and never begin
   the rationale with a placeholder: say what the value is ("the material-change count of {E12}").
3. If nothing needs flagging, which is the usual case, return an empty list of issues. Never list a claim just
   to say it is supported.
4. Respond with a single JSON object and nothing else. First write "review", a scratch check of each item in
   one short clause ("claim_1: supported" or the exact wording that is wrong and why); it is discarded. Then
   write "issues", which holds only the items your review found wrong. The default answer for a sound Report is
   {"review": "summary: supported; claim_1: supported; claim_2: supported", "issues": []}
   and only when you found a genuine error does it take this shape:
   {"review": "...", "issues": [{"category": "causal_overreach", "target": "claim_1", "rationale": "..."}]}"""


def _report_context(report: Report, catalog: EvidenceCatalog) -> dict[str, Any]:
    return {
        "targets": report_targets(report),
        "items": [
            {"target": target, "text": claim.text, "rendered": substitute(claim.text, catalog),
             "evidence_ids": claim.evidence_ids}
            for target, claim in report_items(report)
        ],
    }


def production_critic_model() -> Any:
    """Build the live Critic seam: a plain context ``dict`` in, a ``Critique`` out. A response that does
    not parse raises ``ValueError`` so the Agent's single validation retry can feed the problem back."""
    from langchain_core.messages import HumanMessage, SystemMessage

    structured = make_chat_model(temperature=0).with_structured_output(Critique, method="json_mode", include_raw=True)

    def _invoke(context: dict[str, Any], config: Any = None) -> Critique:
        system = CRITIC_SYSTEM_PROMPT + "\n\nEvidence Catalog (id | label | display):\n" + catalog_prompt(
            context["evidence_catalog"]
        )
        draft = "\n".join(
            f"{item['target']} | {item['text']} | rendered: {item['rendered']} | cites: {', '.join(item['evidence_ids'])}"
            for item in context["report"]["items"]
        )
        user = (
            f"User question: {context['user_question']}\n"
            f"Draft Report (target | text | rendered text | cited entries):\n{draft}\n"
            f"Validation feedback from a prior attempt (empty if this is the first): "
            f"{context.get('validation_feedback') or 'none'}\n"
            "Return the JSON object now."
        )
        result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)], config=config)
        if isinstance(result, dict) and "parsed" in result:
            if result["parsed"] is None:
                raise ValueError(f"response was not valid Critique JSON: {result.get('parsing_error')}")
            return result["parsed"]
        return result

    return RunnableLambda(_invoke)


CRITIC_RUN_NAME = "Critic"
REREVIEW_RUN_NAME = "Critic (re-review)"


class CriticAgent:
    """Validated Critic loop: invalid output gets exactly one repair retry with the problems fed back;
    provider failures get a bounded retry. Terminal failure raises, and the node records it. Each review is
    its own named run (``Critic``, or ``Critic (re-review)`` after a revision) so the trace shows the Critic
    as a span with the LLM generation nested under it."""

    max_provider_attempts = 2
    backoff_seconds = 0.0

    def __init__(self, model: Any):
        self.model = model
        self.validation_retries = 0
        self.tool_retries = 0
        self.last_validation_error: str | None = None

    def run(self, handoff: Handoff, report: Report, catalog: EvidenceCatalog, *, config: Any = None,
            run_name: str = CRITIC_RUN_NAME) -> Critique:
        context = {
            "step_objective": handoff.step.objective,
            "user_question": handoff.user_question,
            "evidence_catalog": [entry.model_dump() for entry in catalog.entries],
            "report": _report_context(report, catalog),
        }

        def parse(raw: Any) -> Critique:
            try:
                critique = raw if isinstance(raw, Critique) else Critique.model_validate(raw)
            except ValidationError as exc:
                raise ValueError(f"response does not match the required JSON shape: {summarize_schema_error(exc)}") from exc
            return validate_critique(critique, report, catalog)

        return run_validated(
            self, context, parse, label="Critic", config=config,
            model=named_span(self.model, run_name, ["agent:Critic"]),
            retry_hint=("Return the corrected JSON object only, using only the targets listed, a category from "
                        "the closed set and only Evidence Entry ids inside braces."),
        )


def report_hash(report: Report) -> str:
    payload = json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def with_reviewer_notes(draft: Report, critique: Critique) -> Report:
    """A copy of the draft carrying the Critique's issues as Reviewer Notes; nothing else changes."""
    return draft.model_copy(update={"reviewer_notes": list(critique.issues)})


@dataclass(frozen=True)
class CritiqueApplication:
    """What applying a Critique produced: the Report to ship, the outcome and the revision bookkeeping."""

    report: Report
    outcome: str
    revised_critique: Critique | None = None
    revision_failure: str | None = None
    rereview_error: str | None = None


def _apply_critique(agent: "CriticAgent", handoff: Handoff, draft: Report, critique: Critique,
                    catalog: EvidenceCatalog, configurable: dict, config: Any) -> CritiqueApplication:
    """Turn a Critique of the draft into the final Report and outcome (ADR-0003).

    A clean Critique ships the draft. A non-empty one triggers exactly one Reporter revision, validated by the
    first draft's rules; the revised Report is reviewed once more, and whatever the Critic still flags becomes
    Reviewer Notes. A revision that fails keeps the draft with the original issues as Reviewer Notes. A Fallback
    Report is not revised (the Reporter model just failed to write it).
    """
    if not critique.issues:
        return CritiqueApplication(draft, "clean")
    unresolved = with_reviewer_notes(draft, critique)
    if draft.report_mode == "fallback":
        return CritiqueApplication(unresolved, "unresolved",
                                   revision_failure="not attempted: the draft is a Fallback Report")
    try:
        model = configurable.get("reporter_model") or production_reporter_model()
        content = revise_content(model, handoff, catalog, draft, critique, config=config)
        revised = build_report(content, catalog, draft.question, draft.charts, draft.chart_mode)
    except Exception as exc:  # noqa: BLE001 — a failed revision never ships; the draft stays
        return CritiqueApplication(unresolved, "unresolved", revision_failure=format_failure(exc))
    try:
        second = agent.run(handoff, revised, catalog, config=config, run_name=REREVIEW_RUN_NAME)
    except Exception as exc:  # noqa: BLE001 — cannot confirm the fix: keep the revision, keep the original issues visible
        return CritiqueApplication(with_reviewer_notes(revised, critique), "unresolved",
                                   rereview_error=format_failure(exc))
    return CritiqueApplication(with_reviewer_notes(revised, second), "unresolved" if second.issues else "revised",
                               revised_critique=second)


def _as_report(value: Any) -> Report | None:
    if value is None:
        return None
    return value if isinstance(value, Report) else Report.model_validate(value)


def critic_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    """Execute the Critic Step on the draft Report.

    The key is the Step contract plus the draft Report hash: an unchanged key reuses the stored
    Critique with no model call, and a recorded failure is retried rather than reused.
    """
    artifacts = state.get("artifacts", {})
    plan = artifacts.get("plan")
    configurable = (config or {}).get("configurable", {})
    if not isinstance(plan, Plan):
        return {"current_step": "critic_node"}
    step = next((candidate for candidate in plan.steps if candidate.agent == "Critic"), None)
    draft = _as_report(artifacts.get("draft_report") or artifacts.get("report"))
    if step is None or draft is None:
        return {"current_step": "critic_node"}

    draft_hash = report_hash(draft)
    current_key = execution_key(step, input_artifact_hashes=[draft_hash])
    resume_outcome, resume_reason = _resume_decision(state, step.step_id, current_key)
    report_reference = {"kind": "Report", "artifact_id": f"report-{draft_hash[:16]}", "content_hash": draft_hash}

    agent = None
    handoff = None
    failure: Exception | None = None
    critique: Critique | None = None
    final_report = draft
    critic_outcome = "failed"
    revised_critique: Critique | None = None
    revision_failure: str | None = None
    rereview_error: str | None = None
    try:
        catalog = artifacts.get("evidence_catalog")
        if not isinstance(catalog, EvidenceCatalog):
            catalog = EvidenceCatalog.model_validate(catalog)
        reporter_step = next((candidate for candidate in plan.steps if candidate.agent == "Reporter"), None)
        report_names = (reporter_step.declared_artifacts or [reporter_step.expected_output]) if reporter_step else []
        crew = Crew(
            available_agents=configurable.get("available_agents", CREW_AGENTS),
            data_sources=configurable.get("data_sources", {}),
        )
        handoff = crew.resolve_handoff(
            step, plan, {**artifacts.get("step_outputs", {}), **{name: report_reference for name in report_names}},
            dict(configurable.get("analysis_parameters", {})),
        )
        if resume_outcome == "reused":
            stored = artifacts.get("critique")
            stored_report = artifacts.get("reviewed_report")
            stored_outcome = artifacts.get("critic_outcome")
            if stored is None or stored_report is None or stored_outcome in (None, "failed"):
                # Defensive: never trust a reuse whose output we cannot actually produce.
                resume_outcome, resume_reason = "recomputed", "invalidated"
            else:
                critique = stored if isinstance(stored, Critique) else Critique.model_validate(stored)
                final_report = _as_report(stored_report)
                critic_outcome = stored_outcome
                second = artifacts.get("revised_critique")
                revised_critique = None if second is None else (
                    second if isinstance(second, Critique) else Critique.model_validate(second))
                revision_failure = artifacts.get("revision_failure")
                rereview_error = artifacts.get("rereview_error")
        if resume_outcome != "reused":
            model = configurable.get("critic_model") or production_critic_model()
            agent = CriticAgent(model)
            critique = agent.run(handoff, draft, catalog, config=config)
            applied = _apply_critique(agent, handoff, draft, critique, catalog, configurable, config)
            final_report, critic_outcome = applied.report, applied.outcome
            revised_critique, revision_failure = applied.revised_critique, applied.revision_failure
            rereview_error = applied.rereview_error
    except Exception as exc:  # noqa: BLE001 — a Critic failure ships the un-reviewed Report, never fails the run
        failure, critique, final_report, critic_outcome = exc, None, draft, "failed"
        revised_critique = revision_failure = rereview_error = None

    completed = failure is None
    retry_counts = (
        (artifacts.get("step_outcomes", {}).get(step.step_id, {}).get("retry_counts")
         or {"validation_retries": 0, "tool_retries": 0})
        if resume_outcome == "reused" and completed
        else {"validation_retries": getattr(agent, "validation_retries", 0),
              "tool_retries": getattr(agent, "tool_retries", 0)}
    )
    outcome: dict[str, Any] = {
        "status": "completed" if completed else "failed",
        "artifact_ids": step.declared_artifacts or [step.expected_output],
        "input_artifact_ids": [report_reference["artifact_id"]],
        "retry_counts": retry_counts,
        "execution_key": current_key,
        "resume_outcome": resume_outcome,
        "resume_reason": resume_reason,
        "critic_outcome": critic_outcome,
    }
    if revision_failure:
        outcome["revision_failure"] = revision_failure
    if rereview_error:
        outcome["rereview_error"] = rereview_error
    if failure is not None:
        outcome["error"] = format_failure(failure)
    critic_failure = None if failure is None else {
        "agent": "Critic", "step_id": step.step_id, "error": outcome["error"], "degraded_to": "unreviewed_report",
    }
    return {
        "current_step": "critic_node",
        "steps_completed": [step.step_id] if completed else [],
        "artifacts": {
            "report": final_report,
            "draft_report": draft,
            "reviewed_report": final_report,
            "critique": critique,
            "revised_critique": revised_critique,
            "revision_failure": revision_failure,
            "rereview_error": rereview_error,
            "critic_failure": critic_failure,
            "critic_outcome": critic_outcome,
            "handoffs": artifacts.get("handoffs", {}) if handoff is None else {
                **artifacts.get("handoffs", {}), step.step_id: handoff.model_dump(mode="json"),
            },
            "step_outcomes": {**artifacts.get("step_outcomes", {}), step.step_id: outcome},
            "retry_counts": {**artifacts.get("retry_counts", {}), step.step_id: retry_counts},
        },
    }
