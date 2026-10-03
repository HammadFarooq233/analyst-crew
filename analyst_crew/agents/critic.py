"""Critic Agent and graph node (ADR-0003), flag only.

The Critic has no Tools. Its inputs are the draft Report and the Evidence
Catalog; it returns a Critique whose issues each target the Summary or an
existing Claim. It cannot add Claims. Code validates the Critique, turns the
issues into Reviewer Notes on a copy of the draft, and records ``critic_outcome``
as ``clean``, ``unresolved`` or ``failed``. A Critic failure ships the
un-reviewed Report with the failure recorded; it never fails the run.
"""
import hashlib
import json
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from analyst_crew.agents.contracts import CREW_AGENTS, Crew, Handoff, Plan
from analyst_crew.agents.reporter import (
    _catalog_prompt, _is_transient_provider_error, _summarize_schema_error,
)
from analyst_crew.agents.resume import _resume_decision, execution_key
from analyst_crew.llm import make_chat_model
from analyst_crew.reporting.catalog import EvidenceCatalog
from analyst_crew.reporting.critique import Critique, report_targets, validate_critique
from analyst_crew.reporting.report import Report, _substitute
from analyst_crew.state import CrewState
from analyst_crew.tools.retry import invoke_with_optional_config, retry_with_backoff

CRITIC_SYSTEM_PROMPT = """You are the Critic Agent for Analyst Crew, a multi-agent data analysis copilot.

You review a draft Report for judgment problems that code cannot check. Code has already guaranteed that every
number is an Evidence Entry value, that citations resolve and that caveats are present. You have no Tools and no
data beyond the Evidence Catalog below; you never compute and you never add Claims or rewrite text.

Flag an issue only when the wording of the Summary or a Finding is not supported by the Evidence Entries it
cites. Categories (use exactly these):
- causal_overreach: states or implies that something caused the change; the data only shows what changed.
- misread_evidence: the wording misreads its cited entries (wrong direction, wrong subject, wrong comparison).
- overstated_driver: presents a driver as larger or more important than its entries show.
- unsupported_emphasis: stresses or generalises beyond what the cited entries support.

Rules:
1. Each issue targets exactly one item: "summary" or "claim_<n>" where <n> is the Finding number shown below.
2. Give a short rationale. Write NO digits in it; refer to an entry only by placeholder, for example {E12}.
3. If nothing needs flagging, return an empty list of issues.
4. Respond with a single JSON object and nothing else, in exactly this shape:
   {"issues": [{"category": "causal_overreach", "target": "claim_1", "rationale": "..."}]}"""


def _report_context(report: Report, catalog: EvidenceCatalog) -> dict[str, Any]:
    items = [("summary", report.summary), *[(f"claim_{n}", claim) for n, claim in enumerate(report.findings, start=1)]]
    return {
        "targets": report_targets(report),
        "items": [
            {"target": target, "text": claim.text, "rendered": _substitute(claim.text, catalog),
             "evidence_ids": claim.evidence_ids}
            for target, claim in items
        ],
    }


def production_critic_model() -> Any:
    """Build the live Critic seam: a plain context ``dict`` in, a ``Critique`` out. A response that does
    not parse raises ``ValueError`` so the Agent's single validation retry can feed the problem back."""
    from langchain_core.messages import HumanMessage, SystemMessage

    structured = make_chat_model(temperature=0).with_structured_output(Critique, method="json_mode", include_raw=True)

    def _invoke(context: dict[str, Any], config: Any = None) -> Critique:
        system = CRITIC_SYSTEM_PROMPT + "\n\nEvidence Catalog (id | label | display):\n" + _catalog_prompt(
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


class CriticAgent:
    """Validated Critic loop: invalid output gets exactly one repair retry with the problems fed back;
    provider failures get a bounded retry. Terminal failure raises, and the node records it."""

    max_provider_attempts = 2
    backoff_seconds = 0.0

    def __init__(self, model: Any):
        self.model = model
        self.validation_retries = 0
        self.tool_retries = 0

    def run(self, handoff: Handoff, report: Report, catalog: EvidenceCatalog, *, config: Any = None) -> Critique:
        context = {
            "step_objective": handoff.step.objective,
            "user_question": handoff.user_question,
            "evidence_catalog": [entry.model_dump() for entry in catalog.entries],
            "report": _report_context(report, catalog),
        }
        feedback = ""
        for attempt in range(2):
            calls = 0

            def call() -> Any:
                nonlocal calls
                calls += 1
                return invoke_with_optional_config(self.model, {**context, "validation_feedback": feedback}, config)

            try:
                try:
                    raw, _ = retry_with_backoff(
                        call, is_transient=_is_transient_provider_error,
                        max_attempts=self.max_provider_attempts, backoff_seconds=self.backoff_seconds,
                    )
                finally:
                    self.tool_retries += calls - 1
                try:
                    critique = raw if isinstance(raw, Critique) else Critique.model_validate(raw)
                except ValidationError as exc:
                    raise ValueError(f"response does not match the required JSON shape: {_summarize_schema_error(exc)}") from exc
                return validate_critique(critique, report, catalog)
            except ValueError as exc:
                if attempt:
                    raise
                self.validation_retries += 1
                feedback = (
                    f"Your previous response was invalid: {exc}. Return the corrected JSON object only, using only "
                    "the targets listed, a category from the closed set and no digits outside {E<id>} placeholders."
                )
        raise RuntimeError("Critic request loop exhausted")


def report_hash(report: Report) -> str:
    payload = json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def with_reviewer_notes(draft: Report, critique: Critique) -> Report:
    """A copy of the draft carrying the Critique's issues as Reviewer Notes; nothing else changes."""
    return draft.model_copy(update={"reviewer_notes": list(critique.issues)})


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
            if stored is None:
                # Defensive: never trust a reuse whose output we cannot actually produce.
                resume_outcome, resume_reason = "recomputed", "invalidated"
            else:
                critique = stored if isinstance(stored, Critique) else Critique.model_validate(stored)
        if resume_outcome != "reused":
            model = configurable.get("critic_model") or production_critic_model()
            agent = CriticAgent(model)
            critique = agent.run(handoff, draft, catalog, config=config)
        final_report = with_reviewer_notes(draft, critique)
    except Exception as exc:  # noqa: BLE001 — a Critic failure ships the un-reviewed Report, never fails the run
        failure, critique, final_report = exc, None, draft

    completed = failure is None
    critic_outcome = "failed" if not completed else ("unresolved" if critique.issues else "clean")
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
    if failure is not None:
        outcome["error"] = (
            str(failure) if isinstance(failure, (RuntimeError, ValueError)) else f"{type(failure).__name__}: {failure}"
        )
    return {
        "current_step": "critic_node",
        "steps_completed": [step.step_id] if completed else [],
        "artifacts": {
            "report": final_report,
            "draft_report": draft,
            "critique": critique,
            "critic_outcome": critic_outcome,
            "handoffs": artifacts.get("handoffs", {}) if handoff is None else {
                **artifacts.get("handoffs", {}), step.step_id: handoff.model_dump(mode="json"),
            },
            "step_outcomes": {**artifacts.get("step_outcomes", {}), step.step_id: outcome},
            "retry_counts": {**artifacts.get("retry_counts", {}), step.step_id: retry_counts},
        },
    }
