"""Reporter Agent and graph node (ADR-0002).

The Reporter is an LLM-driven Agent with no Tools that execute code and no data
access. Its only inputs are the Evidence Catalog, the user's question and the
Handoff; it returns a Summary and ordered Claims. Code validates every citation
and placeholder, assembles Methodology and Caveats, and degrades to a
deterministic Fallback Report if the Reporter cannot produce valid output.
"""
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from analyst_crew.agents.contracts import CREW_AGENTS, Crew, Handoff, Plan
from analyst_crew.agents.resume import _resume_decision, execution_key
from analyst_crew.llm import make_chat_model
from analyst_crew.reporting.catalog import EvidenceCatalog, build_evidence_catalog
from analyst_crew.reporting.report import (
    Report, ReporterContent, build_fallback_report, build_report, validate_reporter_content,
)
from analyst_crew.state import CrewState
from analyst_crew.tools.analysis import BaselineFindings
from analyst_crew.tools.retry import invoke_with_optional_config, retry_with_backoff

REPORTER_SYSTEM_PROMPT = """You are the Reporter Agent for Analyst Crew, a multi-agent data analysis copilot.

You write the Summary and the Findings of a Report answering the user's question. Your only source of facts is the
Evidence Catalog below: numbered Evidence Entries, each with an id, a label, a display value and the Findings field
it came from. You have no other data and you never compute anything.

Rules:
1. Write NO digits anywhere in your text. Every number, percentage, date, period label, count and any name that
   contains a digit must be written as a placeholder made of an Evidence Entry id in braces, for example {E12}.
   The renderer replaces each placeholder with the entry's display value. Never restate, round or convert a value.
   Signed changes already carry their sign, so write "a change of {E12}" and never add words such as "plus".
2. Every Claim, and the Summary, lists in evidence_ids every Evidence Entry it relies on. Use only ids from the catalog.
3. The Summary is a short headline answer to the question that states the comparison window and the net revenue change.
4. Findings are an ordered list of Claims, most important first: the biggest drivers of the change come first.
   Call out every Material Change (entries flagged material) with its size and share of the overall change, summarize
   the revenue bridge (volume, price, interaction, discount, return, new and discontinued products), and, when the
   catalog has them, report target variance by region and category and the inflation-adjusted context.
5. State only what the entries support. Describe what changed; do not claim what caused it. CPI context is never causal.
6. Methodology and Caveats are written by the system and always appear; do not restate them. You may add a Claim that
   comments on a caveat.
7. Respond with a single JSON object and nothing else, in exactly this shape:
   {"summary": {"text": "...", "evidence_ids": ["E1"]}, "claims": [{"text": "...", "evidence_ids": ["E2", "E3"]}]}"""


class TransientReporterError(Exception):
    """Retryable Reporter provider failure."""


def _is_transient_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, TransientReporterError)):
        return True
    if any(cls.__name__ in {"APITimeoutError", "APIConnectionError"} for cls in type(exc).__mro__):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status in (408, 409, 429) or status >= 500)


def _catalog_prompt(entries: list[Mapping[str, Any]]) -> str:
    lines = []
    for entry in entries:
        flag = " [MATERIAL CHANGE]" if entry.get("material") else ""
        lines.append(f"{entry['id']} | {entry['label']} | {entry['display']}{flag}")
    return "\n".join(lines)


def production_reporter_model() -> Any:
    """Build the live Reporter seam.

    ``ReporterAgent.run`` invokes this with a plain context ``dict``; this
    wraps it into system and user messages and asks for JSON that parses into
    ``ReporterContent``. A response that does not parse raises ``ValueError``
    so the Agent's single validation retry can feed the problem back.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    structured = make_chat_model(temperature=0).with_structured_output(
        ReporterContent, method="json_mode", include_raw=True,
    )

    def _invoke(context: dict[str, Any], config: Any = None) -> ReporterContent:
        system = REPORTER_SYSTEM_PROMPT + "\n\nEvidence Catalog (id | label | display):\n" + _catalog_prompt(
            context["evidence_catalog"]
        )
        user = (
            f"User question: {context['user_question']}\n"
            f"Step objective: {context['step_objective']}\n"
            f"Validation feedback from a prior attempt (empty if this is the first): "
            f"{context.get('validation_feedback') or 'none'}\n"
            "Return the JSON object now."
        )
        result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)], config=config)
        if isinstance(result, dict) and "parsed" in result:
            if result["parsed"] is None:
                raise ValueError(f"response was not valid Report JSON: {result.get('parsing_error')}")
            return result["parsed"]
        return result

    return RunnableLambda(_invoke)


def _summarize_schema_error(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())


class ReporterAgent:
    """Validated Reporter loop.

    Output that breaks the schema or the citation rules gets exactly one
    repair retry with the validation errors fed back. A provider failure
    (timeouts, connection errors, 429/5xx) gets a bounded retry; any other
    provider failure is terminal at once. ``validation_retries`` and
    ``tool_retries`` (provider retries; the Reporter has no Tools) record how
    many were used, in the same shape as the other Agents.
    """

    max_provider_attempts = 2
    backoff_seconds = 0.0

    def __init__(self, model: Any):
        self.model = model
        self.validation_retries = 0
        self.tool_retries = 0
        self.last_validation_error: str | None = None

    def run(self, handoff: Handoff, catalog: EvidenceCatalog, *, config: Any = None) -> ReporterContent:
        context = {
            "step_objective": handoff.step.objective,
            "user_question": handoff.user_question,
            "evidence_catalog": [entry.model_dump() for entry in catalog.entries],
        }
        feedback = ""
        for attempt in range(2):
            calls = 0

            def call() -> Any:
                nonlocal calls
                calls += 1
                return invoke_with_optional_config(self.model, {**context, "validation_feedback": feedback}, config)

            try:
                # A ValueError out of the model call means the response itself was unusable
                # (e.g. unparseable JSON), so it takes the validation-retry path; provider
                # failures are other exception types and are retried (or not) below.
                try:
                    raw, _ = retry_with_backoff(
                        call,
                        is_transient=_is_transient_provider_error,
                        max_attempts=self.max_provider_attempts,
                        backoff_seconds=self.backoff_seconds,
                    )
                finally:
                    # Counted even when the bounded retries are exhausted and the call raises.
                    self.tool_retries += calls - 1
                try:
                    content = raw if isinstance(raw, ReporterContent) else ReporterContent.model_validate(raw)
                except ValidationError as exc:
                    raise ValueError(f"response does not match the required JSON shape: {_summarize_schema_error(exc)}") from exc
                return validate_reporter_content(content, catalog)
            except ValueError as exc:
                self.last_validation_error = str(exc)
                if attempt:
                    raise
                self.validation_retries += 1
                feedback = (
                    f"Your previous response was invalid: {exc}. Return the corrected JSON object only, "
                    "with no digits outside {E<id>} placeholders and only ids from the catalog."
                )
        raise RuntimeError("Reporter request loop exhausted")


def findings_hash(findings: BaselineFindings) -> str:
    payload = json.dumps(findings.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reporter_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    """Execute the Reporter Step from the Analyst's Findings.

    Builds the Evidence Catalog from the Findings, resolves the Handoff (the
    Findings artifact reference, the question and the analysis parameters),
    and derives the Step's execution key from the Step, parameters and the
    Findings content hash. An unchanged key reuses the stored Report with no
    model call; changed Findings change the key and regenerate it. If the
    Reporter fails after its bounded retries, the terminal failure is recorded
    and a Fallback Report is returned so Priya still gets a cited answer.
    """
    artifacts = state.get("artifacts", {})
    plan = artifacts.get("plan")
    configurable = (config or {}).get("configurable", {})
    if not isinstance(plan, Plan):
        return {"current_step": "reporter_node", "artifacts": {"reporter_error": "No executable Plan"}}
    reporter_step = next((step for step in plan.steps if step.agent == "Reporter"), None)
    if reporter_step is None:
        return {"current_step": "reporter_node"}
    raw_findings = artifacts.get("findings")
    if raw_findings is None:
        return {"current_step": "reporter_node", "artifacts": {"reporter_error": "No Findings to report on"}}
    findings = raw_findings if isinstance(raw_findings, BaselineFindings) else BaselineFindings.model_validate(raw_findings)

    agent = None
    try:
        catalog = build_evidence_catalog(findings)
        current_hash = findings_hash(findings)
        findings_reference = {
            "kind": "Findings", "artifact_id": f"findings-{current_hash[:16]}", "content_hash": current_hash,
            "dataset_artifact_id": findings.dataset_artifact_id,
        }
        analyst_step = next((step for step in plan.steps if step.agent == "Analyst"), None)
        findings_names = (analyst_step.declared_artifacts or [analyst_step.expected_output]) if analyst_step else []
        analysis_parameters = dict(configurable.get("analysis_parameters", {}))
        crew = Crew(
            available_agents=configurable.get("available_agents", CREW_AGENTS),
            data_sources=configurable.get("data_sources", {}),
        )
        handoff = crew.resolve_handoff(
            reporter_step, plan,
            {**artifacts.get("step_outputs", {}), **{name: findings_reference for name in findings_names}},
            analysis_parameters,
        )
        current_key = execution_key(
            reporter_step, analysis_parameters=analysis_parameters, input_artifact_hashes=[current_hash],
        )
        resume_outcome, resume_reason = _resume_decision(state, reporter_step.step_id, current_key)

        prior_outcome = artifacts.get("step_outcomes", {}).get(reporter_step.step_id, {})
        report = None
        if resume_outcome == "reused":
            stored = artifacts.get("report")
            if stored is None:
                # Defensive: never trust a reuse we cannot actually produce output for.
                resume_outcome, resume_reason = "recomputed", "invalidated"
            else:
                report = stored if isinstance(stored, Report) else Report.model_validate(stored)

        failure = None
        if resume_outcome == "reused":
            retry_counts = prior_outcome.get("retry_counts") or {"validation_retries": 0, "tool_retries": 0}
        else:
            try:
                model = configurable.get("reporter_model") or production_reporter_model()
                agent = ReporterAgent(model)
                content = agent.run(handoff, catalog, config=config)
                report = build_report(content, catalog, plan.user_question)
            except Exception as exc:  # noqa: BLE001 — any Reporter failure degrades visibly, never silently
                failure = exc
                report = build_fallback_report(catalog, plan.user_question)
            retry_counts = {
                "validation_retries": getattr(agent, "validation_retries", 0),
                "tool_retries": getattr(agent, "tool_retries", 0),
            }

        completed = failure is None
        failure_message = (
            str(failure) if isinstance(failure, (RuntimeError, ValueError)) or failure is None
            else f"{type(failure).__name__}: {failure}"
        )
        return {
            "current_step": "reporter_node",
            "steps_completed": [reporter_step.step_id] if completed else [],
            "artifacts": {
                "report": report,
                "report_mode": report.report_mode,
                "evidence_catalog": catalog,
                "terminal_failure": None if completed else {
                    "agent": "Reporter",
                    "step_id": reporter_step.step_id,
                    "required": False,
                    "status": "failed",
                    "error": failure_message,
                    "retry_counts": retry_counts,
                    "degraded_to": "fallback_report",
                },
                "handoffs": {**artifacts.get("handoffs", {}), reporter_step.step_id: handoff.model_dump(mode="json")},
                "step_outcomes": {
                    **artifacts.get("step_outcomes", {}),
                    reporter_step.step_id: {
                        "status": "completed" if completed else "failed",
                        "artifact_ids": reporter_step.declared_artifacts or [reporter_step.expected_output],
                        "input_artifact_ids": [findings_reference["artifact_id"]],
                        "retry_counts": retry_counts,
                        "execution_key": current_key,
                        "resume_outcome": resume_outcome,
                        "resume_reason": resume_reason,
                        "report_mode": report.report_mode,
                    },
                },
                "retry_counts": {**artifacts.get("retry_counts", {}), reporter_step.step_id: retry_counts},
            },
        }
    except (OSError, RuntimeError, ValueError) as exc:
        # The Report could not even be set up (e.g. an invalid Handoff): no
        # Findings-derived answer is possible, so this is a required failure.
        return {
            "current_step": "reporter_node",
            "artifacts": {
                "reporter_error": str(exc),
                "terminal_failure": {
                    "agent": "Reporter", "step_id": reporter_step.step_id, "required": True,
                    "status": "failed", "error": str(exc),
                    "retry_counts": {"validation_retries": 0, "tool_retries": 0},
                },
            },
        }
