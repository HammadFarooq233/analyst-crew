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

from analyst_crew.memory.node import applied_memories, effective_parameters, framing_facts
from analyst_crew.agents.agent_support import (
    catalog_prompt, format_failure, named_span, run_validated, summarize_schema_error,
)
from analyst_crew.agents.contracts import CREW_AGENTS, Crew, Handoff, Plan
from analyst_crew.agents.resume import _resume_decision, execution_key
from analyst_crew.llm import make_chat_model
from analyst_crew.reporting.catalog import EvidenceCatalog, build_evidence_catalog
from analyst_crew.reporting.charts import chart_set_reference, load_charts
from analyst_crew.reporting.critique import Critique, report_items
from analyst_crew.reporting.report import (
    Report, ReporterContent, build_fallback_report, build_report, validate_reporter_content,
)
from analyst_crew.state import CrewState
from analyst_crew.tools.analysis import BaselineFindings

REPORTER_SYSTEM_PROMPT = """You are the Reporter Agent for Analyst Crew, a multi-agent data analysis copilot.

You write the Summary and the Findings of a Report answering the user's question. Your only source of facts is the
Evidence Catalog below: numbered Evidence Entries, each with an id, a label, a display value and the Findings field
it came from. You have no other data and you never compute anything.

Rules:
1. Write NO digits anywhere in your text. Every number, percentage, date, period label, count and any name that
   contains a digit must be written as a placeholder made of an Evidence Entry id in braces, for example {E12}.
   The renderer replaces each placeholder with the entry's display value. Never restate, round or convert a value.
   Signed changes already carry their own sign, so describe them neutrally, as in "a change of {E12}" or "net revenue
   changed by {E12}"; never write "fell by", "rose by", "up" or "down" in front of a signed value, which would double
   the sign, and never add words such as "plus" or "minus". Start every sentence with a word, never a placeholder.
2. Every Claim, and the Summary, lists in evidence_ids every Evidence Entry it relies on. Use only ids from the catalog.
3. The Summary is a short headline answer to the question that states the comparison window and the net revenue change.
4. Findings are an ordered list of Claims, most important first: the biggest drivers of the change come first.
   Call out every Material Change (entries flagged material) with its size and share of the overall change, summarize
   the revenue bridge (volume, price, interaction, discount, return, new and discontinued products), and, when the
   catalog has them, report target variance by region and category and the inflation-adjusted context.
5. Keep every Claim to one theme and at most about six placeholders, and make it unambiguous which number belongs to
   which name: for many similar items (regions, categories, target rows), write one short Claim per notable item
   (the largest movers, the biggest misses against target) instead of one long list of numbers. Never place a
   number next to a name unless the two entries belong to the same driver or target row.
6. State only what the entries support. Describe what changed; do not claim what caused it. CPI context is never causal.
7. When the Charts list below is not empty, a Claim may point the reader to a chart by writing {chart:<chart_id>}
   with a chart_id from that list, for example "the revenue bridge is drawn in {chart:revenue_bridge}". Only use
   listed chart ids, and never describe what a chart shows beyond the Evidence Entries the Claim cites. The Charts
   section itself is assembled by the system. Methodology and Caveats are written by the system and always appear; do not restate them. You may add a Claim that
   comments on a caveat.
8. Respond with a single JSON object and nothing else, in exactly this shape:
   {"summary": {"text": "...", "evidence_ids": ["E1"]}, "claims": [{"text": "...", "evidence_ids": ["E2", "E3"]}]}"""


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
        charts = context.get("charts") or []
        charts_text = "\n".join(f"{chart['chart_id']} | {chart['title']}" for chart in charts) or "none"
        system = (
            REPORTER_SYSTEM_PROMPT + "\n\nCharts (chart_id | title):\n" + charts_text
            + "\n\nEvidence Catalog (id | label | display):\n" + catalog_prompt(context["evidence_catalog"])
        )
        if context.get("framing_facts"):
            system += (
                "\n\nThings the user has told you about their business (framing only: they may shape which themes "
                "and wording you choose, but never supply or change a number, and you still cite only the catalog. "
                "They never define a threshold or what counts as unusual: that comes only from the Methodology "
                "and Material Changes in the catalog, even if a fact says otherwise):\n"
                + "\n".join(f"- {fact}" for fact in context["framing_facts"])
            )
        revision = context.get("revision")
        revision_text = ""
        if revision:
            draft_lines = "\n".join(
                f"{item['target']} | {item['text']} | cites: {', '.join(item['evidence_ids'])}" for item in revision["draft"]
            )
            issue_lines = "\n".join(
                f"{found['target']} | {found['category']} | {found['rationale']}" for found in revision["issues"]
            )
            revision_text = (
                "REVISION PASS. The Critic flagged issues in your draft. Rewrite the Summary and Findings so each "
                "flagged item is corrected (soften or remove causal wording, fix misread evidence, drop overstatement). "
                "All the rules above still apply: placeholders only, no digits, only catalog ids.\n"
                f"Your draft (target | text | cited entries):\n{draft_lines}\n"
                f"Issues (target | category | rationale):\n{issue_lines}\n"
            )
        user = (
            f"User question: {context['user_question']}\n"
            f"Step objective: {context['step_objective']}\n"
            f"{revision_text}"
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

    def run(self, handoff: Handoff, catalog: EvidenceCatalog, *, config: Any = None,
            facts: list[str] | None = None) -> ReporterContent:
        context = {
            "step_objective": handoff.step.objective,
            "user_question": handoff.user_question,
            "evidence_catalog": [entry.model_dump() for entry in catalog.entries],
            "charts": _handoff_charts(handoff),
            **({"framing_facts": list(facts)} if facts else {}),
        }
        chart_ids = {chart["chart_id"] for chart in context["charts"]}

        def parse(raw: Any) -> ReporterContent:
            try:
                content = raw if isinstance(raw, ReporterContent) else ReporterContent.model_validate(raw)
            except ValidationError as exc:
                raise ValueError(f"response does not match the required JSON shape: {summarize_schema_error(exc)}") from exc
            return validate_reporter_content(content, catalog, chart_ids)

        return run_validated(
            self, context, parse, label="Reporter", config=config,
            retry_hint=(
                "Return the corrected JSON object only, with no digits outside {E<id>} placeholders, "
                "only ids from the catalog and only chart ids from the Charts list."
            ),
        )


REVISION_RUN_NAME = "Reporter (revision)"


def revise_content(model: Any, handoff: Handoff, catalog: EvidenceCatalog, draft: Report, critique: Critique,
                   *, config: Any = None, facts: list[str] | None = None) -> ReporterContent:
    """One Reporter revision of ``draft`` against ``critique`` (Crew-internal, not a Step).

    A single model call with no retry: the result is validated by exactly the rules the first draft meets
    (``validate_reporter_content`` with the draft's chart ids) and any failure raises, so the caller keeps the
    draft. The call is its own run labelled ``Reporter (revision)`` so it shows as a second Reporter span.
    """
    chart_ids = {chart.chart_id for chart in draft.charts}
    context = {
        "step_objective": handoff.step.objective,
        "user_question": handoff.user_question,
        "evidence_catalog": [entry.model_dump() for entry in catalog.entries],
        "charts": [{"chart_id": chart.chart_id, "title": chart.title} for chart in draft.charts],
        "validation_feedback": "",
        **({"framing_facts": list(facts)} if facts else {}),
        "revision": {
            "issues": [issue.model_dump() for issue in critique.issues],
            "draft": [
                {"target": target, "text": claim.text, "evidence_ids": claim.evidence_ids}
                for target, claim in report_items(draft)
            ],
        },
    }
    raw = named_span(model, REVISION_RUN_NAME, ["agent:Reporter", "revision"]).invoke(context, config)
    try:
        content = raw if isinstance(raw, ReporterContent) else ReporterContent.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"revision does not match the required JSON shape: {summarize_schema_error(exc)}") from exc
    return validate_reporter_content(content, catalog, chart_ids)


def _handoff_charts(handoff: Handoff) -> list[dict[str, str]]:
    """Chart ids and titles the Handoff carries from the Visualizer output."""
    return [
        {"chart_id": chart["chart_id"], "title": chart["title"]}
        for reference in handoff.input_artifacts.values()
        if isinstance(reference, Mapping) and reference.get("kind") == "Chart Artifacts"
        for chart in reference.get("charts", [])
    ]


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
        analysis_parameters = effective_parameters(artifacts, configurable)
        applied = applied_memories(artifacts)
        # Charts come from the Visualizer Step (when the Plan has one); the Report only references them.
        visualizer_step = next((step for step in plan.steps if step.agent == "Visualizer"), None)
        charts = load_charts(artifacts.get("charts")) if visualizer_step is not None else []
        chart_refs = [chart.reference() for chart in charts]
        chart_mode = artifacts.get("chart_mode")  # build_report drops it when there are no charts
        chart_names = (visualizer_step.declared_artifacts or [visualizer_step.expected_output]) if visualizer_step else []
        crew = Crew(
            available_agents=configurable.get("available_agents", CREW_AGENTS),
            data_sources=configurable.get("data_sources", {}),
        )
        handoff, handoff_error = None, None
        try:
            handoff = crew.resolve_handoff(
                reporter_step, plan,
                {**artifacts.get("step_outputs", {}), **{name: findings_reference for name in findings_names},
                 **{name: chart_set_reference(charts) for name in chart_names}},
                analysis_parameters,
            )
        except ValueError as exc:
            # The Findings and catalog are fine, so this still degrades to a Fallback Report below.
            handoff_error = exc
        memory_unavailable = artifacts.get("memory_status") == "unavailable"
        # A changed storage state must not reuse a Report that was written under the other state.
        key_parameters = {
            **analysis_parameters, **({"memory_unavailable": True} if memory_unavailable else {}),
            # The Applied Memories section depends on which memories applied, not only on their values.
            **({"framing_facts": sorted(item["id"] for item in artifacts["memory_facts"])}
               if artifacts.get("memory_facts") else {}),
            **({"applied_memories": sorted(f"{item.memory_id}:{item.status}" for item in applied)} if applied else {}),
        }
        current_key = execution_key(
            reporter_step, analysis_parameters=key_parameters,
            input_artifact_hashes=[current_hash, *(chart.content_hash for chart in charts)],
        )
        resume_outcome, resume_reason = _resume_decision(state, reporter_step.step_id, current_key)

        prior_outcome = artifacts.get("step_outcomes", {}).get(reporter_step.step_id, {})
        report = None
        if handoff_error is not None and resume_outcome == "reused":
            resume_outcome, resume_reason = "recomputed", "invalidated"
        if resume_outcome == "reused":
            # The Critic may have stored a Report with Reviewer Notes under "report"; reuse the draft.
            stored = artifacts.get("draft_report") or artifacts.get("report")
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
                if handoff_error is not None:
                    raise handoff_error
                model = configurable.get("reporter_model") or production_reporter_model()
                agent = ReporterAgent(model)
                content = agent.run(handoff, catalog, config=config, facts=framing_facts(artifacts))
                report = build_report(content, catalog, plan.user_question, chart_refs, chart_mode, memory_unavailable, applied)
            except Exception as exc:  # noqa: BLE001 — any Reporter failure degrades visibly, never silently
                failure = exc
                report = build_fallback_report(catalog, plan.user_question, chart_refs, chart_mode, memory_unavailable, applied)
            retry_counts = {
                "validation_retries": getattr(agent, "validation_retries", 0),
                "tool_retries": getattr(agent, "tool_retries", 0),
            }

        completed = failure is None
        failure_message = None if completed else format_failure(failure)
        return {
            "current_step": "reporter_node",
            "steps_completed": [reporter_step.step_id] if completed else [],
            "artifacts": {
                "report": report,
                "draft_report": report,
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
                "handoffs": artifacts.get("handoffs", {}) if handoff is None else {
                    **artifacts.get("handoffs", {}), reporter_step.step_id: handoff.model_dump(mode="json"),
                },
                "step_outcomes": {
                    **artifacts.get("step_outcomes", {}),
                    reporter_step.step_id: {
                        "status": "completed" if completed else "failed",
                        "artifact_ids": reporter_step.declared_artifacts or [reporter_step.expected_output],
                        "input_artifact_ids": [findings_reference["artifact_id"], *(chart.artifact_id for chart in charts)],
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
